from functools import partial
from heapq import nlargest
import json
import os
from pprint import pprint
import torch
from torch import nn
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq, Qwen2ForCausalLM, set_seed
from accelerate import Accelerator

from src.sft.inference import print_query_and_answer

from .process_data import CustomCollator, list_of_dicts_to_dict_of_lists as dataset_list_to_dict, process_func_chatml
from .loss import compute_answer_only_saliency_masked_loss, compute_gradients_selected_attention, compute_loss_per_sample

from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from torch.nn import Parameter


class Qwen2ForCausalLMWithLastAttn(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.last_attention = None

    def forward(self, *args, save_last_attention: bool = False, **kwargs):
        self.last_attention = None
        handle = None

        if save_last_attention:
            def _hook(_module, _inputs, output):
                _, attn_weights = output
                self.last_attention = attn_weights

            # Hook only the last layer's attention
            last_attn = self.model.layers[-1].self_attn
            if not isinstance(last_attn, nn.Module):
                raise TypeError("Expected `self_attn` to be an `nn.Module`.")
            handle = last_attn.register_forward_hook(_hook)

        try:
            outputs = super().forward(*args, **kwargs)
        finally:
            if handle is not None:
                handle.remove()

        if not save_last_attention:
            return outputs

        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=(self.last_attention,) if self.last_attention is not None else None,
        )


# Unfreeze target layers
target_layer_keywords = ["embed_tokens.weight"]  # or ["model.layers.27.mlp"]

def unfreeze_params(model: torch.nn.Module):
    for name, param in model.named_parameters():
        if any(key in name for key in target_layer_keywords):
            param.requires_grad = True

def filter_params(name: str, param: Parameter):
    return any(key in name for key in target_layer_keywords) and param.requires_grad


def load_samples_from_formal_jsonl(jsonl_path: str):
    sample_list = []
    seen_inputs  = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if len(obj['messages']) > 2 and len(obj['messages'][2]['content']) > 0:
                    sys  = obj['messages'][0]['content']
                    inp  = obj['messages'][1]['content']
                    outp = obj['messages'][2]['content']
                    if inp in seen_inputs:
                        continue
                    sample_list.append(
                        {
                            'system': sys,
                            'input':  inp,
                            'output': outp
                        }
                    )
                    seen_inputs.add(inp)
            if len(sample_list) >= 1000:
                break

    return sample_list


def build_train_dataset(train_samples, convert_fn):
    train_ds = Dataset.from_dict(dataset_list_to_dict(train_samples))
    train_ds = train_ds.map(lambda x, i: {"sample_index": i}, with_indices=True)
    train_ds = train_ds.map(
        convert_fn, 
        batched=True,
        remove_columns=["input", "output", "system"]
    )
    train_ds.set_format(type="torch", columns=["input_ids", "labels", "sample_index"])

    return train_ds


def build_single_sample_query_batch(sample, convert_fn):
    # Build single-sample Dataset
    temp_ds = Dataset.from_dict({
        "input":  [sample["input"]],
        "output": [sample["output"]],
        "system": [sample["system"]],
    })
    temp_ds = temp_ds.map(
        convert_fn, 
        batched=True, 
        remove_columns=["input", "output", "system"]
    )

    return temp_ds


def load_model_and_tokenizer():
    abs_model_path = os.path.join(os.path.dirname(__file__), "sft/scripts/checkpoint-full-long-2")
    print(f"Loading model from {abs_model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(abs_model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(
        abs_model_path,
        attn_implementation="eager",
        output_attentions=False,
        use_cache=False,
    )

    model = Qwen2ForCausalLMWithLastAttn.from_pretrained(
        abs_model_path,
        config=config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).eval()

    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze some
    unfreeze_params(model)

    return model, tokenizer


def _find_subseq_start(row: torch.Tensor, subseq: tuple[int, int, int]) -> int:
    a, b, c = subseq
    for i in range(row.numel() - 1):
        if int(row[i]) == a and int(row[i + 1]) == b:
            return i
    raise ValueError("marker sequence not found")


class NewInferenceFunction:
    def __init__(
        self,
        model,
        tokenizer,
        train_loader=None,
        accelerator=None,
        param_filter_fn=None,
        ignored_token_ids=None,
        top_k=None,
        top_k_ratio=0.2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.accelerator = accelerator
        self.param_filter_fn = param_filter_fn
        self.device = accelerator.device if accelerator is not None else next(model.parameters()).device
        self.top_k = top_k
        self.top_k_ratio = top_k_ratio

        if ignored_token_ids is None:
            ignored_token_ids = []
        self.ignored_token_ids = torch.tensor(ignored_token_ids, device=self.device)

        self.base_train_results = None

    @staticmethod
    @torch.no_grad()
    def _get_param_snapshot(model, param_filter_fn):
        snapshot = []
        for name, param in model.named_parameters():
            if param.requires_grad and (param_filter_fn is None or param_filter_fn(name, param)):
                snapshot.append(param.detach().cpu().clone())
        return snapshot

    @staticmethod
    @torch.no_grad()
    def _restore_params(model, snapshot, param_filter_fn):
        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad and (param_filter_fn is None or param_filter_fn(name, param)):
                param.data.copy_(snapshot[idx].to(param.device))
                idx += 1

    @staticmethod
    @torch.no_grad()
    def _apply_gradient_update(model, grads, param_filter_fn, lr):
        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad and (param_filter_fn is None or param_filter_fn(name, param)):
                if grads[idx] is not None:
                    grad_device = grads[idx].to(param.device).to(param.dtype)
                    param.data -= lr * grad_device
                idx += 1

    @torch.no_grad()
    def _get_last_layer_attn(self, input_ids, attention_mask, target_idx):
        self.model.eval()
        self.model.set_attn_implementation("eager")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        self.model.set_attn_implementation("sdpa")

        last_layer_attn = outputs.attentions[-1]
        predicting_idx = target_idx - 1
        attn = last_layer_attn[0, :, predicting_idx, :target_idx]
        return attn.mean(0).detach().cpu()

    def _build_topk_mask(self, base_attention_mask, attn_scores, target_idx):
        new_mask = base_attention_mask.clone()
        num_prev = target_idx
        if num_prev <= 1:
            return new_mask

        if self.top_k is not None:
            k = min(self.top_k, num_prev)
        else:
            k = max(1, int(num_prev * (self.top_k_ratio or 0.2)))

        topk_indices = torch.topk(attn_scores[:num_prev], k=k, largest=True).indices
        keep = torch.zeros(num_prev, dtype=torch.bool, device=new_mask.device)
        keep[topk_indices.to(new_mask.device)] = True

        new_mask[:, :num_prev] = keep.to(new_mask.dtype)
        return new_mask

    @torch.no_grad()
    def masked_inference(self, batch, target_idx=None, gen_limit: int = 128):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        if target_idx is None:
            marker_ids = tuple(
                self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            )
            if len(marker_ids) != 3:
                raise ValueError("expected three-token marker for <|im_start|>assistant\\n")
            starts = [
                _find_subseq_start(input_ids[i], marker_ids) + 3
                for i in range(input_ids.size(0))
            ]
            target_idx = torch.tensor(starts, device=input_ids.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        _, _ , saliency_list = compute_answer_only_saliency_masked_loss(
            self.model,
            batch,
            self.device,
            target_idx
        )

        logits = outputs.logits
        batch_size = logits.size(0)

        prev_ids, prev_text, out_ids, out_text = [], [], [], []
        for i in range(batch_size):
            pos = int(target_idx[i].item()) - 1
            original_prev_ids = input_ids[i, :pos].tolist()
            out_token_id = torch.argmax(logits[i, pos], dim=-1).item()

            prev_ids.append(original_prev_ids)
            prev_text.append(self.tokenizer.decode(original_prev_ids))
            out_ids.append(out_token_id)
            out_text.append(self.tokenizer.decode(out_token_id))

        # Trim each sample to target_idx and pad to a common length for generation
        trim_lens = target_idx.to(torch.long).tolist()
        max_len = max(trim_lens)
        trimmed_ids = input_ids.new_full((batch_size, max_len), self.tokenizer.eos_token_id)
        trimmed_mask = attention_mask.new_zeros((batch_size, max_len))

        for i, tlen in enumerate(trim_lens):
            trimmed_ids[i, :tlen] = input_ids[i, :tlen]
            trimmed_mask[i, :tlen] = 1

        gen_ids = self.model.generate(
            input_ids=trimmed_ids,
            attention_mask=trimmed_mask,
            max_new_tokens=gen_limit,
            do_sample=False,
            eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.pad_token_id],
            pad_token_id=self.tokenizer.pad_token_id
        )

        pred_ids, pred_text = [], []
        full_text, answer_text = [], []
        pred_tokens, full_tokens, answer_tokens = [], [], []

        for i in range(batch_size):
            prompt_len = int(target_idx[i].item())
            continuation = gen_ids[i, prompt_len:].tolist()
            pred_ids.append(continuation)
            pred_tokens.append(self.tokenizer.convert_ids_to_tokens(continuation))
            pred_text.append(self.tokenizer.decode(continuation))

            valid_ids = input_ids[i, :int(attention_mask[i].sum().item())].tolist()
            full_tokens.append(self.tokenizer.convert_ids_to_tokens(valid_ids))
            full_text.append(self.tokenizer.decode(valid_ids))

            ans_ids = input_ids[i, prompt_len:int(attention_mask[i].sum().item())].tolist()
            answer_tokens.append(self.tokenizer.convert_ids_to_tokens(ans_ids))
            answer_text.append(self.tokenizer.decode(ans_ids))

        return {
            **batch,
            "logits": logits,
            "attention_mask": attention_mask,
            "target_idx": target_idx,
            "prev_ids": prev_ids,
            "prev_text": prev_text,
            "out_ids": out_ids,
            "out_text": out_text,
            "gen_ids": gen_ids,
            "pred_ids": pred_ids,
            "pred_text": pred_text,
            "full_text": full_text,
            "answer_text": answer_text,
            "pred_tokens": pred_tokens,
            "full_tokens": full_tokens,
            "answer_tokens": answer_tokens,
            "saliency": saliency_list
        }

    def decode_next_token(self, logits, position):
        token_id = torch.argmax(logits[0, position], dim=-1).item()
        return self.tokenizer.decode(token_id)

    def _get_train_losses(self):
        all_sum_losses, all_indices = [], []
        tokenwise_dict = {}

        self.model.eval()
        with torch.inference_mode():
            if not isinstance(self.train_loader, DataLoader):
                raise TypeError("Expected `self.train_loader` to be an `DataLoader`.")
            pbar = tqdm(self.train_loader, desc=f"Getting Train Losses", leave=False)
            loss = 1.0
            for batch in pbar:
                pbar.set_postfix(loss=f"{loss:.4f}")
                batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
                mean_loss, token_loss = compute_loss_per_sample(
                    self.model,
                    batch_gpu,
                    self.device,
                    self.ignored_token_ids
                )

                shift_labels = batch_gpu["labels"][..., 1:].contiguous()
                indices = batch_gpu["sample_index"].cpu().tolist()

                for i, idx in enumerate(indices):
                    valid_mask = shift_labels[i] != -100
                    start_idx = torch.where(valid_mask)[0][0] if valid_mask.any() else 0
                    tokenwise_dict[idx] = token_loss[i][start_idx:].cpu()

                all_sum_losses.append(mean_loss.detach().cpu())
                all_indices.append(batch_gpu["sample_index"].detach().cpu())

                loss = mean_loss.item()
                del batch_gpu, mean_loss, token_loss, shift_labels  # crucial release

        return torch.cat(all_sum_losses), torch.cat(all_indices), tokenwise_dict

    def influence_overfit_single(
        self,
        query_batch,
        lr=1e-2,
        max_steps=1000,
        loss_threshold=1e-4,
        top_k=10,
        target_idx=None,
    ):
        if self.base_train_results is None:
            self.base_train_results = self._get_train_losses()

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        def _masked_query_batch(batch, target_idx_local):
            masked = {k: v for k, v in batch.items()}
            labels = batch["labels"].clone()
            start = int(target_idx_local[0].item())
            labels[..., :start] = -100
            masked["labels"] = labels
            return masked, start

        if target_idx is None:
            raise ValueError("target_idx is required to overfit generation starting from target_idx")

        query_batch, start_q = _masked_query_batch(query_batch, target_idx)

        # Compute starting loss
        with torch.no_grad():
            scalar, tokenwise_raw = compute_loss_per_sample(
                self.model, query_batch, self.device, self.ignored_token_ids
            )
            loss_test_start = scalar.item()
            loss_test_start_tokenwise = tokenwise_raw[0][start_q:].cpu()

        snapshot = self._get_param_snapshot(self.model, self.param_filter_fn)

        # Overfitting
        curr_test_loss = loss_test_start
        pbar = tqdm(range(max_steps), desc=f"Overfitting on single sample", leave=False)
        for step in pbar:
            pbar.set_postfix(loss=f"{curr_test_loss:.4f}")

            if curr_test_loss < loss_threshold:
                break

            grads = compute_gradients_selected_attention(
                self.model, query_batch, self.param_filter_fn, self.device, self.ignored_token_ids, target_idx=target_idx
            )
            self._apply_gradient_update(self.model, grads, self.param_filter_fn, lr=lr)
            self.model.zero_grad(set_to_none=True)

            with torch.no_grad():
                scalar, _ = compute_loss_per_sample(
                    self.model, query_batch, self.device, self.ignored_token_ids
                )
                curr_test_loss = scalar.item()

        # Compute ending loss
        with torch.no_grad():
            _, l_test_des_tokenwise_raw = compute_loss_per_sample(
                self.model, query_batch, self.device, self.ignored_token_ids
            )
            loss_test_end_tokenwise = l_test_des_tokenwise_raw[0][start_q:].cpu()

        # Test loss diff
        query_token_diffs = loss_test_end_tokenwise - loss_test_start_tokenwise
        delta_test = curr_test_loss - loss_test_start

        # Train loss diff
        l_train_des_sum, _, l_train_des_tokenwise = self._get_train_losses()
        self._restore_params(self.model, snapshot, self.param_filter_fn)
        l_train_base_sum, indices_local, l_train_base_tokenwise = self.base_train_results

        # Compute scores
        local_scores = []
        local_diffs = {}
        for i, idx in enumerate(indices_local.tolist()):
            rel_delta_train = l_train_des_sum[i].item() - l_train_base_sum[i].item()
            denom = l_train_base_sum[i].item() + 1e-8
            normalized_score = delta_test * (rel_delta_train / denom)
            local_scores.append(normalized_score)
            local_diffs[idx] = l_train_des_tokenwise[idx] - l_train_base_tokenwise[idx]

        if self.accelerator is None:
            return local_scores, indices_local.tolist(), local_diffs, query_token_diffs

        all_scores = self.accelerator.gather(torch.tensor(local_scores, device=self.device))
        all_indices = self.accelerator.gather(indices_local)

        return all_scores.tolist(), all_indices.tolist(), local_diffs, query_token_diffs

class HFToTorchDataset(TorchDataset):
    def __init__(self, hf_dataset):
        self._ds = hf_dataset

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]


def main():

    accelerator = Accelerator()
    set_seed(42)

    model, tokenizer = load_model_and_tokenizer()

    convert_to_chatml_with_tokenizer = partial(process_func_chatml, tokenizer=tokenizer)

    # Load Data
    train_samples = load_samples_from_formal_jsonl("sft_train.jsonl")

    test_samples = load_samples_from_formal_jsonl("sft_test.jsonl")

    train_ds = build_train_dataset(train_samples, convert_to_chatml_with_tokenizer)

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    # Extracts sample index to the batch property
    collator = CustomCollator(base_collator)

    train_loader = DataLoader(
        HFToTorchDataset(train_ds.select(range(100))),  # only check the first 100 training samples
        batch_size=1,
        shuffle=False,
        collate_fn=collator
    )

    train_loader = accelerator.prepare(train_loader)

    # Assume: model, tokenizer already loaded
    # Assume: query_batch already built with input_ids and attention_mask

    infer = NewInferenceFunction(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        accelerator=accelerator,
        param_filter_fn=filter_params,
        top_k=20,
    )

    # Build query batch
    temp_ds = build_single_sample_query_batch(test_samples[9], convert_to_chatml_with_tokenizer)    # choose a proper length sample, or it will cuda oom
    query_batch = base_collator([temp_ds[0]])

    for k, v in query_batch.items():
        query_batch[k] = v.to(accelerator.device)

    # 1) Masked inference for one wrong sample
    result = infer.masked_inference(query_batch)
    print_query_and_answer(result["prev_text"][0], result["answer_text"][0], result["pred_text"][0])

    # 2) Rebuild query batch using prediction as new ground truth
    prompt_len = int(result["target_idx"][0].item())
    prompt_ids = query_batch["input_ids"][0, :prompt_len]
    pred_ids = torch.tensor(
        result["pred_ids"][0],
        device=prompt_ids.device,
        dtype=prompt_ids.dtype
    )
    new_input_ids = torch.cat([prompt_ids, pred_ids], dim=0).unsqueeze(0)
    new_attention_mask = torch.ones_like(new_input_ids)
    new_labels = new_input_ids.clone()
    new_labels[:, :prompt_len] = -100  # ignore prompt tokens in loss

    query_batch = {
        "input_ids": new_input_ids,
        "attention_mask": new_attention_mask,
        "labels": new_labels
    }

    # 3) Empirical influence (overfit on the new ground-truth batch)
    scores, indices, train_diffs, query_diffs = infer.influence_overfit_single(
        query_batch=query_batch,
        lr=5e-3,
        max_steps=50,
        target_idx=result["target_idx"]
    )

    result = nlargest(10, enumerate(scores), key=lambda x: x[1])
    pprint(result)

    print(">>> Top-10 related training samples\n")
    for i, sim in result:
        # Build query batch
        temp_ds = build_single_sample_query_batch(train_samples[i], convert_to_chatml_with_tokenizer)    # choose a proper length sample, or it will cuda oom
        query_batch = base_collator([temp_ds[0]])

        for k, v in query_batch.items():
            query_batch[k] = v.to(accelerator.device)

        result = infer.masked_inference(query_batch)
        print_query_and_answer(result["prev_text"][0], result["answer_text"][0], result["pred_text"][0])



if __name__ == "__main__":
    main()

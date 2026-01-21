from functools import partial
import json
import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, set_seed
from accelerate import Accelerator

from .process_data import CustomCollator, list_of_dicts_to_dict_of_lists as dataset_list_to_dict, process_func_chatml
from .loss import compute_gradients, compute_gradients_selected_attention, compute_loss_per_sample, compute_loss_per_sample_selected_attention

from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from torch.nn import Parameter


# Unfreeze target layers
target_layer_keywords = ["lm_head"]  # or ["model.layers.27.mlp"]

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
    model = AutoModelForCausalLM.from_pretrained(
        abs_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).eval()

    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

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

        self.train_batches = []
        self.base_train_results = None

        if self.train_loader is not None:
            for batch in self.train_loader:
                batch_cpu = {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                self.train_batches.append(batch_cpu)

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
    def masked_inference(self, batch, target_idx=None):
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

        attn_scores = self._get_last_layer_attn(input_ids, attention_mask, target_idx)
        masked_attention = self._build_topk_mask(attention_mask, attn_scores, target_idx)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=masked_attention,
            return_dict=True,
        )

        logits = outputs.logits
        batch_size = logits.size(0)

        prev_ids = []
        prev_text = []
        out_ids = []
        out_text = []

        for i in range(batch_size):
            pos = int(target_idx[i].item()) - 1
            original_prev_ids = input_ids[i, :pos].tolist()
            out_token_id = torch.argmax(logits[i, pos], dim=-1).item()

            prev_ids.append(original_prev_ids)
            prev_text.append(self.tokenizer.decode(original_prev_ids))
            out_ids.append(out_token_id)
            out_text.append(self.tokenizer.decode(out_token_id))

        return {
            "logits": logits,
            "masked_attention": masked_attention,
            "target_idx": target_idx,
            "prev_ids": prev_ids,
            "prev_text": prev_text,
            "out_ids": out_ids,
            "out_text": out_text,
        } 
    
    def decode_next_token(self, logits, position):
        token_id = torch.argmax(logits[0, position], dim=-1).item()
        return self.tokenizer.decode(token_id)

    def _get_train_losses(self):
        all_sum_losses = []
        all_indices = []
        tokenwise_dict = {}
        self.model.eval()

        with torch.no_grad():
            for batch in self.train_batches:
                batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
                mean_loss, token_loss = compute_loss_per_sample(
                    self.model, batch_gpu, self.device, self.ignored_token_ids
                )

                shift_labels = batch_gpu["labels"][..., 1:].contiguous()
                indices = batch_gpu["sample_index"].cpu().tolist()

                for i, idx in enumerate(indices):
                    valid_mask = shift_labels[i] != -100
                    if valid_mask.any():
                        start_idx = torch.where(valid_mask)[0][0]
                        tokenwise_dict[idx] = token_loss[i][start_idx:].cpu()
                    else:
                        tokenwise_dict[idx] = torch.tensor([], device="cpu")

                all_sum_losses.append(mean_loss)
                all_indices.append(batch_gpu["sample_index"])

        return torch.cat(all_sum_losses), torch.cat(all_indices), tokenwise_dict

    def influence_overfit_single(
        self,
        query_batch,
        lr=1e-2,
        max_steps=1000,
        loss_threshold=1e-4,
        top_k=10,
    ):
        if self.base_train_results is None:
            self.base_train_results = self._get_train_losses()

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Compute labels range
        shift_labels_q = query_batch["labels"][..., 1:].contiguous()
        valid_q = shift_labels_q[0] != -100
        start_q = torch.where(valid_q)[0][0] if valid_q.any() else 0

        # Compute starting loss
        with torch.no_grad():
            l_test_base_scalar, l_test_base_tokenwise_raw = compute_loss_per_sample_selected_attention(
                self.model, query_batch, self.device, self.ignored_token_ids, top_k=top_k
            )
            l_test_base = l_test_base_scalar.item()

            l_test_base_tokenwise = l_test_base_tokenwise_raw[0][start_q:].cpu()

        snapshot = self._get_param_snapshot(self.model, self.param_filter_fn)

        # Training
        curr_test_loss = l_test_base
        for _ in range(max_steps):
            if curr_test_loss < loss_threshold:
                break

            # SGD
            grads = compute_gradients_selected_attention(
                self.model, query_batch, self.param_filter_fn, self.device, self.ignored_token_ids
            )
            self._apply_gradient_update(self.model, grads, self.param_filter_fn, lr=lr)
            self.model.zero_grad(set_to_none=True)

            # Compute loss again
            with torch.no_grad():
                loss_s, _ = compute_loss_per_sample_selected_attention(
                    self.model, query_batch, self.device, self.ignored_token_ids, top_k=top_k
                )
                curr_test_loss = loss_s.item()

        # Compute ending loss
        with torch.no_grad():
            _, l_test_des_tokenwise_raw = compute_loss_per_sample_selected_attention(
                self.model, query_batch, self.device, self.ignored_token_ids, top_k=top_k
            )
            l_test_des_tokenwise = l_test_des_tokenwise_raw[0][start_q:].cpu()

        # Test loss diff
        query_token_diffs = l_test_des_tokenwise - l_test_base_tokenwise
        delta_test = curr_test_loss - l_test_base

        # Train loss diff
        l_train_des_sum, _, l_train_des_tokenwise = self._get_train_losses()
        self._restore_params(self.model, snapshot, self.param_filter_fn)
        l_train_base_sum, indices_local, l_train_base_tokenwise = self.base_train_results   # indices_local is train indices?

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
        batch_size=2,
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
    temp_ds = build_single_sample_query_batch(test_samples[0], convert_to_chatml_with_tokenizer)
    query_batch = base_collator([temp_ds[0]])

    for k, v in query_batch.items():
        query_batch[k] = v.to(accelerator.device)

    # 1) Masked inference for one wrong sample
    result = infer.masked_inference(query_batch)
    print(">>> Prev text:\n", result["prev_text"][0], sep="")
    print(">>> Pred token:\n", result["out_text"][0], sep="")

    # 2) Empirical influence (overfit on one wrong sample)
    scores, indices, train_diffs, query_diffs = infer.influence_overfit_single(
        query_batch=query_batch,
        lr=5e-3,
        max_steps=5,
    )

    print(scores)


if __name__ == "__main__":
    main()

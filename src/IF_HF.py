import torch
import logging
from typing import List, Dict, Callable, Optional, Any
from torch import nn
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset
from functools import partial
from accelerate import Accelerator
from accelerate.utils import set_seed
import json
from tqdm import tqdm
import re
import os
import numpy as np
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 类型别名
BatchDict = Dict[str, torch.Tensor]
ParamFilterFn = Callable[[str, nn.Parameter], bool]


def list_of_dicts_to_dict_of_lists(data_list):
    return {
        "input": [d["input"] for d in data_list],
        "output": [d["output"] for d in data_list]
    }


def extract_code_content(text):
    text = text.replace("This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. ", "")
    pattern = r"And here is the function you are asked to complete\s*([\s\S]*) Ensure that only missing codes marked as <MID> are returned"

    match = re.search(pattern, text)

    if match:
        extracted_part = match.group(1)
        return extracted_part
    else:
        return text


def process_func_chatml(examples, tokenizer, max_len=2048, system_message="You are a helpful assistant."):
    """
    将 input/output 转换为 ChatML 格式：
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {input}<|im_end|>
        <|im_start|>assistant
        {output}<|im_end|>
    并仅对 assistant 的回复部分计算 Loss。
    """
    inputs  = examples["input"]
    outputs = examples["output"]

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id  = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens  = tokenizer.encode("\n", add_special_tokens=False)

    def _build_turn(role, content, is_train=False):
        role_ids = [im_start_id] + tokenizer.encode(role, add_special_tokens=False) + nl_tokens
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        footer_ids = [im_end_id] + nl_tokens
        full_ids = role_ids + content_ids + footer_ids

        if is_train:
            # -100 表示忽略计算 Loss
            labels = [-100] * len(role_ids) + content_ids + footer_ids
        else:
            labels = [-100] * len(full_ids)
        return full_ids, labels

    new_input_ids = []
    new_labels = []
    # 注意：不再需要手动生成 attention_mask，DataCollator 会自动处理

    for inp, outp in zip(inputs, outputs):
        input_ids, labels = [], []

        # System
        sys_ids, sys_labels = _build_turn("system", system_message, is_train=False)
        input_ids += sys_ids
        labels += sys_labels

        # User
        user_ids, user_labels = _build_turn("user", inp, is_train=False)
        input_ids += user_ids
        labels += user_labels

        # Assistant
        asst_ids, asst_labels = _build_turn("assistant", outp, is_train=True)
        input_ids += asst_ids
        labels += asst_labels

        # Truncation (仅截断，不 Padding)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

        new_input_ids.append(input_ids)
        new_labels.append(labels)

    return {
        "input_ids": new_input_ids,
        "labels": new_labels
    }


class CustomCollator:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features: List[Dict[str, Any]]) -> BatchDict:
        indices = []
        for f in features:
            if "sample_index" in f:
                indices.append(f.pop("sample_index"))

        batch = self.base_collator(features)

        if indices:
            batch["sample_index"] = torch.tensor(indices, dtype=torch.long)

        return batch

def compute_loss_per_sample(model: nn.Module, batch: BatchDict, device: torch.device) -> torch.Tensor:

    inputs = {
        k: v.to(device)
        for k, v in batch.items()
        if k in ['input_ids', 'attention_mask', 'labels']
    }

    # 前向传播 (不自动计算 Loss)
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits  # [B, Seq_Len, Vocab]

    logits = logits.float()

    # Causal LM 的预测是基于前一个 token 预测下一个，所以 logits 要左移，labels 要右移
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous()

    # 不进行 reduction (mean/sum)
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    # [B * (Seq-1), Vocab] vs [B * (Seq-1)]
    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    token_losses = token_losses.view(shift_labels.size())

    valid_mask = shift_labels.ne(-100).float()

    # Sum(Loss) / Count(Valid_Tokens)
    sum_loss = (token_losses * valid_mask).sum(dim=1)
    num_valid = valid_mask.sum(dim=1)

    per_sample_loss = sum_loss / (num_valid + 1e-9)

    return per_sample_loss


# 为了兼容 compute_gradients，保留原来的标量计算函数
def compute_loss_scalar(model, batch, device):
    loss_tensor = compute_loss_per_sample(model, batch, device)
    return loss_tensor.mean()


def compute_gradients(model, batch, param_filter_fn, device):
    model.eval()  # 保持 eval 模式，我们手动计算梯度
    model.zero_grad(set_to_none=True)

    # Enable grad computation even in eval mode
    with torch.set_grad_enabled(True):
        loss = compute_loss_scalar(model, batch, device)
        params = [p for n, p in model.named_parameters() if
                  p.requires_grad and (param_filter_fn is None or param_filter_fn(n, p))]
        if not params:
            raise ValueError("No params selected for gradient computation")
        grads = torch.autograd.grad(loss, params)

    return list(grads)


class EmpiricalIF:
    def __init__(self, dl_train, model, accelerator, param_filter_fn=None):
        self.dl_train = dl_train
        self.model = model
        self.accelerator = accelerator  # 传入 accelerator 实例
        self.device = accelerator.device
        self.param_filter_fn = param_filter_fn

        # 缓存训练数据 (Cache)
        self.train_batches = []
        if self.accelerator.is_main_process:
            logger.info("Caching training data (distributed shards)...")
        for batch in self.dl_train:
            # 移回 CPU 节省显存，计算时再挪到 GPU
            batch_cpu = {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            self.train_batches.append(batch_cpu)

        local_count = sum(len(b['input_ids']) for b in self.train_batches)
        logger.info(
            f"[Rank {self.accelerator.process_index}] Cached {len(self.train_batches)} batches ({local_count} samples).")

    def _get_train_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        all_losses = []
        all_indices = []

        self.model.eval()
        with torch.no_grad():
            # 使用 tqdm 仅在 debug 时或第一次时开启，否则会刷屏
            iterator = self.train_batches
            if self.accelerator.is_main_process:
               iterator = tqdm(iterator, desc="Calc Train Loss", leave=False)

            for batch in iterator:
                if "sample_index" not in batch:
                    raise ValueError("sample_index is missing! Check CustomCollator.")

                batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
                batch_loss = compute_loss_per_sample(self.model, batch_gpu, self.device)

                all_losses.append(batch_loss)
                all_indices.append(batch_gpu["sample_index"])

        if not all_losses:
            return torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)

        return torch.cat(all_losses), torch.cat(all_indices)

    @staticmethod
    def get_param_snapshot(model: nn.Module, param_filter_fn: Optional[ParamFilterFn]) -> List[torch.Tensor]:
        snapshot = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param_filter_fn is None or param_filter_fn(name, param):
                    snapshot.append(param.detach().clone())
        return snapshot

    @staticmethod
    def restore_params(model: nn.Module, snapshot: List[torch.Tensor], param_filter_fn: Optional[ParamFilterFn]):
        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param_filter_fn is None or param_filter_fn(name, param):
                    param.data.copy_(snapshot[idx])
                    idx += 1

    @staticmethod
    def apply_gradient_update(
            model: nn.Module,
            grads: List[torch.Tensor],
            param_filter_fn: Optional[ParamFilterFn],
            lr: float
    ):
        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param_filter_fn is None or param_filter_fn(name, param):
                    if grads[idx] is not None:
                        grad_device = grads[idx].to(param.device).to(param.dtype)
                        param.data -= lr * grad_device
                    idx += 1

    def query_influence(self, query_batch: BatchDict, lr: float = 1e-4, max_steps: int = 1000, loss_threshold: float = 0.1) -> List[float]:
        """
        Implementation of:
        Term 1: (L_test' - L_test) * (L_train' - L_train)  [from Test Descent]
        Term 2: (L_test'' - L_test) * (L_train'' - L_train) [from Test Ascent]
        Score = (Term 1 + Term 2) / 2
        """

        # 1. 计算 Test Sample 的梯度 (Perturbation Source)
        if self.accelerator.is_main_process:
            logger.info("Computing gradient for query...")

        # 2. 计算 Base Loss (L_test, L_train) - 更新前
        if self.accelerator.is_main_process:
            logger.info("Calculating Base Losses...")

        self.model.eval()
        with torch.no_grad():
            l_test_base = compute_loss_scalar(self.model, query_batch, self.device).item()
        l_train_base, indices_local = self._get_train_losses()  # Vector [N]

        # 备份参数
        snapshot = self.get_param_snapshot(self.model, self.param_filter_fn)

        # Part A: Descent on Test (L')
        steps_taken = 0
        current_test_loss = l_test_base

        if self.accelerator.is_main_process:
            logger.info(f"Step A: Descent (Target Loss < {loss_threshold} or Max Steps {max_steps})...")

        # 使用 while 循环：只要 Loss 还很大且没到最大步数，就继续降
        while current_test_loss > loss_threshold and steps_taken < max_steps:
            grads = compute_gradients(self.model, query_batch, self.param_filter_fn, self.device)
            self.apply_gradient_update(self.model, grads, self.param_filter_fn, lr=lr)

            with torch.no_grad():
                current_test_loss = compute_loss_scalar(self.model, query_batch, self.device).item()

            steps_taken += 1
            if self.accelerator.is_main_process and steps_taken % 10 == 0:
                logger.info(f"  >> Step {steps_taken}: Loss = {current_test_loss:.6f}")

        if self.accelerator.is_main_process:
            logger.info(f"  >> Descent finished in {steps_taken} steps. Final Loss: {current_test_loss:.6f}")

        # L_test'
        l_test_des = current_test_loss
        l_train_des, _ = self._get_train_losses()  # Vector [N]

        # 恢复参数
        self.restore_params(self.model, snapshot, self.param_filter_fn)

        # # Final Score Calculation
        if self.accelerator.is_main_process:
            logger.info("Computing final scores and gathering...")

        # # Scalar deltas
        delta_test_des = l_test_des - l_test_base
        #
        # # Vector deltas [N]
        rel_delta_train_des = (l_train_des - l_train_base)

        # local_scores = ((delta_test_des * rel_delta_train_des) + (delta_test_asc * rel_delta_train_asc)) / 2
        local_scores = delta_test_des * rel_delta_train_des
        all_scores  = self.accelerator.gather(local_scores)
        all_indices = self.accelerator.gather(indices_local)
        return all_scores.tolist(), all_indices.tolist()


def print_sample_detail(train_texts, rank_name, idx, score):
    # 从原始文本列表 train_texts 中获取内容
    content = train_texts[idx]
    print(f"\n>>> {rank_name} (Index: {idx}, Score: {score:.6e})")
    print(f"Input:  {content['input'][:200]}..." if len(
        content['input']) > 200 else f"Input:  {content['input']}")
    print(f"Output: {content['output']}")

def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # 路径配置
    JSONL_PATH = "/home/ruofan/git_space/Empirical-Influence-Function/small_data.jsonl"  # 请确保文件存在
    TEST_JSONL_PATH = "/home/ruofan/git_space/Empirical-Influence-Function/perturbed_small_data.jsonl"  # 请确保文件存在
    MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    accelerator = Accelerator()
    set_seed(42)

    if accelerator.is_main_process:
        logger.info(f"Processes: {accelerator.state.num_processes}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Model
    # 注意：在多卡环境下，显式指定 device_map={"": device} 有助于避免 Accelerate 自动分配冲突
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": accelerator.device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Clone Head for Independent Optimization
    model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.detach().clone())
    model.config.tie_word_embeddings = False

    # Freeze & Unfreeze
    target_layer_keyword = "lm_head"
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if target_layer_keyword in name:
            param.requires_grad = True

    def filter_params(n, p):
        return target_layer_keyword in n and p.requires_grad

    # Load Data
    train_texts = []
    seen_inputs = set()
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if len(obj['messages']) > 2 and len(obj['messages'][2]['content']) > 0:
                    inp = extract_code_content(obj['messages'][1]['content'])
                    if inp in seen_inputs:
                        continue
                    train_texts.append(
                        {
                            'input': inp,
                            'output': obj['messages'][2]['content']
                        }
                    )
                    seen_inputs.add(inp)

    test_texts = []
    for idx in range(len(train_texts)):
        perturbed_idx = (idx + 1) % len(train_texts)
        test_texts.append(
            {
                "input":  train_texts[idx]["input"],
                "output": "Today is a good day",
            }
        )

    # Dataset Preparation
    process_func = partial(process_func_chatml, tokenizer=tokenizer)
    train_ds = Dataset.from_dict(list_of_dicts_to_dict_of_lists(train_texts))
    # 这一步很关键：保留原始 index
    train_ds = train_ds.map(lambda x, i: {"sample_index": i}, with_indices=True)
    train_ds = train_ds.map(process_func, batched=True, remove_columns=["input", "output"])
    train_ds.set_format(type="torch", columns=["input_ids", "labels", "sample_index"])

    # Collator Setup
    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )
    collator = CustomCollator(base_collator)

    BATCH_SIZE = 1
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 建议 False 以便 debug，若 True 也不影响 index tracking
        collate_fn=collator
    )

    # Prepare DataLoader
    train_loader = accelerator.prepare(train_loader)
    total_samples = len(train_texts)

    eif = EmpiricalIF(train_loader, model, accelerator, filter_params)

    self_ranks = []
    for i in tqdm(range(20), desc="Running Experiments"):

        test_sample_dict = test_texts[i]

        # 临时处理 Query Batch
        temp_ds = Dataset.from_dict({"input": [test_sample_dict["input"]], "output": [test_sample_dict["output"]]})
        temp_ds = temp_ds.map(process_func, batched=True, remove_columns=["input", "output"])

        # 这里的 collator 调用不会经过 DataLoader，所以手动处理 device
        query_batch = base_collator([temp_ds[0]])
        for k, v in query_batch.items():
            query_batch[k] = v.to(accelerator.device)

        # 运行 Influence Analysis
        # lr 可以适当大一点，因为我们只更新很少的步数
        scores, indices = eif.query_influence(query_batch, lr=1e-2, max_steps=1000, loss_threshold=1e-4)

        if accelerator.is_main_process:
            unique_results = {}
            for s, idx in zip(scores, indices):
                unique_results[int(idx)] = s

            # Sort Descending: Top = Helpful, Bottom = Harmful
            sorted_results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)

            rank_pos = -1
            self_score = 0.0

            for rank, (idx, score) in enumerate(sorted_results):
                if idx == i:
                    rank_pos = rank
                    self_score = score
                    break

            if rank_pos != -1:
                self_ranks.append(rank_pos)
                percentile = rank_pos / total_samples

                logger.info("=" * 40)
                logger.info(f"Influence Score: {self_score:.6e}")
                logger.info(f"Rank: {rank_pos} / {total_samples}")
                logger.info(f"Percentile: {percentile:.2%} (Lower is more helpful, Higher is more harmful)")
                logger.info("=" * 40)

                # Sanity Check for your hypothesis
                if percentile > 0.9:
                    logger.info("SUCCESS: The sample was identified as highly harmful (conflicting)!")
                elif percentile < 0.1:
                    logger.info("UNEXPECTED: The sample was identified as helpful?")
                else:
                    logger.info("NEUTRAL: The sample did not stand out significantly.")

            else:
                logger.error(f"Sample {i} not found in results!")

            # logger.info("=" * 60)
            # logger.info("DETAILED TOP HARMFUL SAMPLES")

            # # 打印 Rank 最大 (最有害)
            # idx_max, score_max = sorted_results[-1]
            # print_sample_detail(train_texts, "MOST HARMFUL", idx_max, score_max)
            #
            # # 打印 Rank 次大
            # if len(sorted_results) > 1:
            #     idx_second, score_second = sorted_results[-2]
            #     print_sample_detail(train_texts, "SECOND HARMFUL", idx_second, score_second)

            # logger.info("=" * 60)

    if accelerator.is_main_process:

        if not self_ranks:
            logger.error("No valid ranks collected!")
            return

        # 转换为 numpy 数组方便计算
        ranks_arr = np.array(self_ranks)

        # 计算统计量
        min_rank = np.min(ranks_arr)
        max_rank = np.max(ranks_arr)
        median_rank = np.median(ranks_arr)

        print("\n" + "=" * 60)
        print("🧪 EXPERIMENT REPORT: Mismatched Query (Self-Input + Other-Output)")
        print(f"Target Layer       : {target_layer_keyword}")
        print(f"Total Samples Tested : {len(self_ranks)}")
        print("-" * 60)
        print(f"📉 Min Rank          : {min_rank:.0f}")
        print(f"📈 Max Rank          : {max_rank:.0f}")
        print(f"⚖️  Median Rank       : {median_rank:.1f}")


if __name__ == '__main__':
    main()

    # pip install accelerate
    # accelerate config
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 IF_HF.py
    # 本来长什么样子的，一fitting之后长什么样子


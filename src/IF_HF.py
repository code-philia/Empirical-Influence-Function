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
from accelerate.utils import gather_object
import re
import shutil
import re
from process_data import *
from loss import compute_loss_per_sample, compute_gradients
import gc
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 类型别名
ParamFilterFn = Callable[[str, nn.Parameter], bool]

@torch.inference_mode()
def generate_response(model, tokenizer, input_ids, device):
    input_ids = input_ids.to(device)
    # 找到 assistant 的起始位置，截取输入部分
    start_token_id     = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")

    prompt_end_idx = len(input_ids)
    for i in range(len(input_ids) - 1):
        if input_ids[i] == start_token_id and input_ids[i + 1] == assistant_token_id:
            # 输入应包含到 <|im_start|>assistant\n 为止
            prompt_end_idx = i + 2
            # 尝试跳过换行
            if prompt_end_idx < len(input_ids):
                next_id = input_ids[prompt_end_idx].item()
                if tokenizer.decode(next_id) in ['\n', 'Ċ', 'Ġ\n']:
                    prompt_end_idx += 1
            break

    prompt_ids = input_ids[:prompt_end_idx].unsqueeze(0).to(device)  # 增加 batch 维度

    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=20,  # 可根据需要调整
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
        )

    # 只解码新生成的 token
    new_tokens = generated_ids[0][prompt_end_idx:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated_text


class EmpiricalIF:
    def __init__(self, dl_train, model, tokenizer, accelerator, param_filter_fn=None, debug_on=False):
        self.dl_train = dl_train
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.device = accelerator.device
        self.param_filter_fn = param_filter_fn
        self.ignored_token_ids = torch.tensor([], device=self.device)
        self.debug_on = debug_on

        # 1. 缓存训练数据至 CPU，防止显存占用过多
        self.train_batches = []
        if self.accelerator.is_main_process:
            logger.info("Step 1: Caching training shards to CPU...")
        for batch in self.dl_train:
            batch_cpu = {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            self.train_batches.append(batch_cpu)

        # 2. 预计算 Train Base Losses (一次性任务)
        if self.accelerator.is_main_process:
            logger.info("Step 2: Pre-calculating Train Base Losses (Global Reference)...")
        # 这里计算的是该进程负责的那部分样本的 base loss
        self.base_train_results = self._get_train_losses()
        gc.collect()
        torch.cuda.empty_cache()

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

    def _get_train_losses(self):
        """内部方法：计算当前进程分片内所有样本的 Loss"""
        all_sum_losses, all_indices, tokenwise_dict = [], [], {}
        self.model.eval()

        with torch.no_grad():
            for batch in self.train_batches:
                batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
                mean_loss, token_loss = compute_loss_per_sample(self.model, batch_gpu, self.device)

                shift_labels = batch_gpu["labels"][..., 1:].contiguous()
                indices = batch_gpu["sample_index"].cpu().tolist()

                for i, idx in enumerate(indices):
                    # 仅保留 Response 部分的 Loss 用于可视化
                    valid_mask = (shift_labels[i] != -100)
                    if valid_mask.any():
                        start_idx = torch.where(valid_mask)[0][0]
                        tokenwise_dict[idx] = token_loss[i][start_idx:].cpu()
                    else:
                        tokenwise_dict[idx] = torch.tensor([], device='cpu')

                all_sum_losses.append(mean_loss)
                all_indices.append(batch_gpu["sample_index"])

        return torch.cat(all_sum_losses), torch.cat(all_indices), tokenwise_dict

    def query_influence(self, query_batch: BatchDict, lr: float = 1e-2, max_steps: int = 1000, loss_threshold: float = 1e-4):

        if self.base_train_results is None:
            self.base_train_results = self._get_train_losses()
            gc.collect()
            torch.cuda.empty_cache()

        if self.accelerator.is_main_process:
            logger.info("Computing Influence Function...")

        self.model.eval()

        # ================== 1. 计算 Query Base Loss (含 Token-level) ==================
        with torch.no_grad():
            l_test_base_scalar, l_test_base_tokenwise_raw = compute_loss_per_sample(
                self.model, query_batch, self.device
            )
            l_test_base = l_test_base_scalar.item()

            # 对齐 Query 的 Token-level diffs
            shift_labels_q = query_batch["labels"][..., 1:].contiguous()
            valid_q = (shift_labels_q[0] != -100)
            start_q = torch.where(valid_q)[0][0] if valid_q.any() else 0
            l_test_base_tokenwise = l_test_base_tokenwise_raw[0][start_q:].cpu()

        if self.accelerator.is_main_process:
            logger.info(f"Initial Testing Loss = {l_test_base}...")

        # 备份参数
        snapshot = self.get_param_snapshot(self.model, self.param_filter_fn)

        # ================== 3. 在 Query 上执行梯度下降 ==================
        # 3. 对 Query 执行梯度下降 (扰动)
        curr_test_loss = l_test_base
        for step in range(max_steps):
            if curr_test_loss < loss_threshold:
                break

            # 使用加速后的梯度计算
            grads = compute_gradients(self.model, query_batch, self.param_filter_fn, self.device)
            self.apply_gradient_update(self.model, grads, self.param_filter_fn, lr=lr)

            with torch.no_grad():
                loss_s, _ = compute_loss_per_sample(self.model, query_batch, self.device)
                curr_test_loss = loss_s.item()

            if self.accelerator.is_main_process:
                logger.info(f"Gradient Descent for {step} Steps, the Testing Loss Decreased to {curr_test_loss}...")

        # ================== 4. 计算 Query Descent Loss (含 Token-level) ==================
        with torch.no_grad():
            _, l_test_des_tokenwise_raw = compute_loss_per_sample(self.model, query_batch, self.device)
            l_test_des_tokenwise = l_test_des_tokenwise_raw[0][start_q:].cpu()

        query_token_diffs = l_test_des_tokenwise - l_test_base_tokenwise
        delta_test = curr_test_loss - l_test_base

        # ================== 6. 计算 Train Descent Losses ==================
        l_train_des_sum, _, l_train_des_tokenwise = self._get_train_losses()

        # 恢复参数
        self.restore_params(self.model, snapshot, self.param_filter_fn)

        # ================== 7. 计算最终分数 ==================
        l_train_base_sum, indices_local, l_train_base_tokenwise = self.base_train_results

        local_scores = []
        local_diffs = {}
        for i, idx in enumerate(indices_local.tolist()):
            # 核心计算：Query 的 Loss 变化 * 训练样本的 Loss 变化
            rel_delta_train = l_train_des_sum[i].item() - l_train_base_sum[i].item()
            denom = l_train_base_sum[i].item() + 1e-8
            normalized_score = delta_test * (rel_delta_train / denom)  # 关注符号而非绝对大小
            local_scores.append(normalized_score)
            local_diffs[idx] = (l_train_des_tokenwise[idx] - l_train_base_tokenwise[idx])

        # 8. 全局通信汇总
        all_scores  = self.accelerator.gather(torch.tensor(local_scores, device=self.device))
        all_indices = self.accelerator.gather(indices_local)
        all_diffs_list = gather_object([local_diffs])

        global_train_diffs = {}
        if self.accelerator.is_main_process:
            for d in all_diffs_list:
                global_train_diffs.update(d)

        # 9. 显存清理：彻底释放中间变量
        del snapshot, grads, l_train_des_sum, l_train_des_tokenwise
        gc.collect()
        torch.cuda.empty_cache()

        # 返回四个值：Scores, Indices, Train Diffs, Query Diffs
        return all_scores.tolist(), all_indices.tolist(), \
               global_train_diffs, query_token_diffs

    def query_resonance_influence(self, query_batch: Dict[str, torch.Tensor], lr: float = 1e-4, max_steps: int = 5):
        """
        双向条件共振实现:
        特意不恢复 Stage A 更新，以探测在 Q 背景下 i 的逻辑冲突。
        """
        # 0. 准备原始基准 (θ0)
        l_train_base_vec, indices_local, _ = self.base_train_results

        with torch.no_grad():
            l_q_base_t, _ = compute_loss_per_sample(self.model, query_batch, self.device)
            l_q_base = l_q_base_t.item()

        # 备份原始参数 θ0
        snapshot_theta0 = self.get_param_snapshot(self.model, self.param_filter_fn)

        # --- Stage A: Query 驱动 (θ0 -> θQ) ---
        for step in range(max_steps):
            grads_q = compute_gradients(self.model, query_batch, self.param_filter_fn, self.device)
            self.apply_gradient_update(self.model, grads_q, self.param_filter_fn, lr=lr)

        # 记录 Q 更新后的基准
        snapshot_after_q = self.get_param_snapshot(self.model, self.param_filter_fn)

        with torch.no_grad():
            # l_q_after_q: 学习 Q 后的 Q loss
            l_q_after_q_t, _ = compute_loss_per_sample(self.model, query_batch, self.device)
            l_q_after_q = l_q_after_q_t.item()

            # l_train_at_q_vec: 学习 Q 后的所有训练样本 Loss (用于 Stage B 的起点)
            l_train_at_q_vec, _, _ = self._get_train_losses()

            # ΔL_i|q: Q 对 i 的原生拉动力 (相对于 θ0)
            delta_train_by_q = (l_train_at_q_vec - l_train_base_vec).cpu()
            # ΔL_q|q: Q 的自优化量
            delta_q_by_q = l_q_after_q - l_q_base

        # --- Stage B: 条件探测 (θQ -> θQ+i) ---
        local_polarized_scores = []
        idx_counter = 0

        for batch in tqdm(self.train_batches, desc="Conditioned Resonance Probing"):
            batch_size = batch['sample_index'].size(0)
            for j in range(batch_size):
                single_sample = {k: v[j:j + 1].to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                # 当前起点：已经学过 Q 后的样本 i 的 Loss
                current_l_i_start = l_train_at_q_vec[idx_counter].item()

                # 模拟学习样本 i (Stage B 建议 1-2 步)
                for _ in range(1):
                    grads_i = compute_gradients(self.model, single_sample, self.param_filter_fn, self.device)
                    self.apply_gradient_update(self.model, grads_i, self.param_filter_fn, lr=lr)

                with torch.no_grad():
                    # 1. 条件自优化: ΔL_i|(i,q)
                    l_i_after_i, _ = compute_loss_per_sample(self.model, single_sample, self.device)
                    delta_i_conditioned = l_i_after_i.item() - current_l_i_start

                    # 2. 条件对 Q 影响: ΔL_q|(i,q)
                    l_q_at_i_t, _ = compute_loss_per_sample(self.model, query_batch, self.device)
                    delta_q_conditioned = l_q_at_i_t.item() - l_q_after_q

                # --- 效能计算 ---
                delta_i_by_q = delta_train_by_q[idx_counter].item()

                # 归一化：帮了对方多少 / 自己能跑多远
                eff_q_to_i = delta_i_by_q / (abs(delta_i_conditioned) + 1e-8)
                eff_i_to_q = delta_q_conditioned / (abs(delta_q_by_q) + 1e-8)

                # 判定：如果 delta_q_conditioned > 0，说明 i 导致 Q 的 Loss 反弹，即 Harmful
                resonance_sign = 1.0 if delta_q_conditioned < 0 else -1.0

                magnitude = torch.sqrt(torch.tensor(abs(eff_q_to_i * eff_i_to_q)))
                final_score = resonance_sign * magnitude.item()

                # 一致性门控：若 Q 帮 i 的方向与 i 帮 Q 的方向相反（一个降一个升），判定为噪声
                if (delta_i_by_q < 0) != (delta_q_conditioned < 0):
                    final_score = 0.0

                local_polarized_scores.append(final_score)

                # 恢复到 Stage A 的终点 (θQ)
                self.restore_params(self.model, snapshot_after_q, self.param_filter_fn)
                self.model.zero_grad(set_to_none=True)
                idx_counter += 1

        # 恢复原始参数 θ0 (不影响后续其他 Query 的探测)
        self.restore_params(self.model, snapshot_theta0, self.param_filter_fn)

        # 全局通信汇总 (修正索引对齐问题)
        all_scores = self.accelerator.gather(torch.tensor(local_polarized_scores, device=self.device))
        all_indices = self.accelerator.gather(indices_local)

        return all_scores.tolist(), all_indices.tolist(), None, None


# --- 内部辅助函数：根据 input_ids 和 diffs 生成着色 HTML ---
def get_colored_html_from_ids(
        tokenizer: AutoTokenizer,
        input_ids: torch.Tensor,
        diffs_tensor: Optional[torch.Tensor] = None,
        enable_coloring: bool=False
):

    # 1. Logic Check: If coloring is disabled or diffs are missing, return plain text
    if not enable_coloring or diffs_tensor is None:
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        # Find the assistant response part to keep the report focused
        try:
            response_part = full_text.split("assistant\n")[-1]
            return response_part.replace('\n', '<br>').replace(' ', '&nbsp;')
        except:
            return full_text.replace('\n', '<br>')

    html_content = ""
    threshold = 1e-4
    diffs_list = diffs_tensor.tolist() if diffs_tensor is not None else []

    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
    start_token_id     = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")

    # Default to starting from the beginning if not found (fallback)
    output_start_idx = 0
    input_ids_cpu = input_ids.cpu()

    for i in range(len(input_ids_cpu) - 1):
        if input_ids_cpu[i] == start_token_id and input_ids_cpu[i + 1] == assistant_token_id:
            # Found <|im_start|>assistant. The content starts 2 tokens later.
            output_start_idx = i + 2

            if output_start_idx < len(input_ids_cpu):
                next_token_id = input_ids_cpu[output_start_idx].item()
                decoded_next = tokenizer.decode(next_token_id)
                if decoded_next in ['\n', 'Ċ', 'Ġ\n']: # Check if it's a newline character (common in Qwen tokenizer)
                    output_start_idx += 1
            break

    # [Modification]: Start iterating directly from output_start_idx
    diff_idx = 0
    for i in range(output_start_idx, len(tokens)):
        t = tokens[i]
        if t in [tokenizer.pad_token, tokenizer.eos_token, '<|im_end|>']:
            break  # Stop coloring upon reaching end token

        # Replace Qwen's special whitespace/newline tokens
        display_t = t.replace('Ċ', '\n').replace('Ġ', ' ').replace('Ä‰', '\t')
        display_t = display_t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        display_t = display_t.replace('\n', '<br>').replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;').replace(' ', '&nbsp;')

        # Core Visualization Logic
        if diff_idx < len(diffs_list):
            d = diffs_list[diff_idx]
            if d > threshold:
                # Red highlight: Loss Increased
                html_content += f'<span style="background-color: #ffebee; color: #c62828; font-weight: bold; border-radius: 2px; border-bottom: 1px solid #ffcdd2;">{display_t}</span>'
            elif d < -threshold:
                # Blue highlight: Loss Decreased
                html_content += f'<span style="background-color: #e3f2fd; color: #1565c0; font-weight: bold; border-radius: 2px; border-bottom: 1px solid #bbdefb;">{display_t}</span>'
            else:
                html_content += display_t
            diff_idx += 1
        else:
            # This branch should theoretically not be reached if diffs are aligned with Output
            html_content += display_t

    return html_content

def save_query_report_html(
        query_idx: int,
        query_batch: BatchDict,
        train_dataset: Dataset,
        rank_pos: int,
        percentile: float,
        score: float,
        tokenizer: AutoTokenizer,
        # Now handles None safely
        train_token_diffs_dict: Optional[Dict[int, torch.Tensor]],
        query_token_diffs: Optional[torch.Tensor],
        top_5_harmful_indices: List[int],
        top_5_harmful_scores: List[float],
        top_5_helpful_indices: List[int],
        top_5_helpful_scores: List[float],
        model: nn.Module,
        param_filter_fn,
        lr,
        max_steps,
        output_dir: str = "reports",
        enable_coloring: bool = True  # New Option: Toggle coloring on/off
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def get_gen_results(indices):
        results = {}
        model.eval()
        for idx in indices:
            sample = train_dataset[idx]
            results[idx] = generate_response(model, tokenizer, sample["input_ids"], model.device)
        return results

    all_indices = [query_idx] + top_5_helpful_indices + top_5_harmful_indices
    logger.info(f"Generating 'Before' responses for query {query_idx}...")
    before_gens = get_gen_results(all_indices)

    # 临时重演扰动
    snapshot = EmpiricalIF.get_param_snapshot(model, param_filter_fn)
    model.eval()
    for _ in range(max_steps):
        grads = compute_gradients(model, query_batch, param_filter_fn, model.device)
        EmpiricalIF.apply_gradient_update(model, grads, param_filter_fn, lr=lr)

    logger.info(f"Generating 'After' responses for query {query_idx}...")
    after_gens = get_gen_results(all_indices)
    EmpiricalIF.restore_params(model, snapshot, param_filter_fn)

    # 渲染 Top 样本的辅助逻辑
    def render_samples(indices, scores, title):
        nonlocal html_template
        html_template += f"<h2>{title}</h2>"
        for i, (idx, h_score) in enumerate(zip(indices, scores)):
            sample_data  = train_dataset[idx]
            colored_code = get_colored_html_from_ids(
                tokenizer=tokenizer,
                input_ids=sample_data["input_ids"].cpu(),
                diffs_tensor=train_token_diffs_dict.get(idx, None) if train_token_diffs_dict else None,
                enable_coloring=enable_coloring
            )

            # 直接从传入的字典获取生成结果
            base_gen = before_gens.get(idx, "No generation found")
            des_gen  = after_gens.get(idx,  "No generation found")

            # 提取 Input Snippet
            full_text     = tokenizer.decode(sample_data["input_ids"], skip_special_tokens=True)
            input_snippet = full_text.split("assistant\n")[0].strip()

            html_template += f"""
            <div class="card">
                <h3>Index {idx} (Score: {h_score:.6e})</h3>
                <span class="label">Input:</span> <pre style="background: #eee; color: #333;">{input_snippet}</pre>
                <span class="label">Ground Truth:</span> <div class="code-box">{colored_code}</div>

                <span class="label">Generation Comparison:</span>
                <div class="gen-container">
                    <div class="gen-sub-box base-gen">
                        <strong>Before Perturbation (Base):</strong><br>{base_gen}
                    </div>
                    <div class="gen-sub-box descent-gen">
                        <strong>After Perturbation (Descent):</strong><br>{des_gen}
                    </div>
                </div>
            </div>
            """

    # --- 生成 Query 的着色 HTML ---
    query_input_ids = query_batch["input_ids"][0].cpu()
    query_full_text = tokenizer.decode(query_input_ids, skip_special_tokens=True)
    try:
        query_input_text = query_full_text.split("assistant\n")[0].strip()
    except:
        query_input_text = "Error extracting input text."

    query_colored_code = get_colored_html_from_ids(
        tokenizer=tokenizer,
        input_ids=query_input_ids,
        diffs_tensor=query_token_diffs,
        enable_coloring=enable_coloring
    )

    # ... (HTML 头部样式保持不变) ...
    html_template = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 20px auto; padding: 20px; background-color: #f8f9fa; }}
            .card {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
            h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .metrics {{ display: flex; gap: 20px; font-weight: bold; color: #e74c3c; }}
            pre {{ background: #2d3436; color: #dfe6e9; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: 'Consolas', monospace; }}
            .code-box {{
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace; /* 程序员最爱的等宽字体 */
                white-space: pre-wrap; /* 保留空格和换行 */
                word-break: break-all;
                line-height: 1.5;
                background: #ffffff;
                padding: 20px;
                border: 1px solid #eaeeef;
                border-radius: 8px;
            }}
            .label {{ font-weight: bold; margin-top: 10px; display: block; color: #7f8c8d; }}
            .gen-container {{ display: flex; gap: 10px; margin-top: 10px; }}
            .gen-sub-box {{ flex: 1; padding: 10px; border-radius: 5px; border: 1px solid #ddd; font-size: 0.9em; }}
            .base-gen {{ background-color: #fff3e0; border-color: #ffe0b2; }}
            .descent-gen {{ background-color: #e8f5e9; border-color: #c8e6c9; }}
        </style>
    </head>
    <body>
        <h1>Experiment Report: Query Sample {query_idx}</h1>

        <div class="card">
            <h2>Query Information (Test)</h2>
            <div class="metrics">
                <span>Rank: {rank_pos+1} / {len(train_dataset)}</span> | <span>Percentile: {percentile:.2%}</span> | <span>Score: {score:.6e}</span>
            </div>
            <span class="label">Input Text (Approximation):</span>
            <pre>{query_input_text}</pre>
            <span class="label">Full Output (Tokenized & Colored):</span>
            <div class="code-box">{query_colored_code}</div>
            <p style="font-size: 0.9em; color: #666;">(Red = Loss Increased, Blue = Loss Decreased)</p>
        </div>
    """

    # --- 生成自身的着色 HTML ---
    render_samples([query_idx], [score], "Original Target in Training Set")

    # 渲染 Harmful 和 Helpful
    render_samples(top_5_harmful_indices, top_5_harmful_scores, "Top 5 Most Harmful Samples")
    render_samples(top_5_helpful_indices, top_5_helpful_scores, "Top 5 Most Helpful Samples")

    html_template += "</body></html>"
    with open(f"{output_dir}/query_{query_idx}.html", "w", encoding="utf-8") as f:
        f.write(html_template)


def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # 路径配置
    JSONL_PATH = "/mnt/nvme0n1/ruofan/git_space/Empirical-Influence-Function/sft.jsonl"  # 请确保文件存在
    MODEL_ID   = "/mnt/nvme0n1/ruofan/git_space/Empirical-Influence-Function/src/sft/scripts/checkpoint-full"
    RESULTS_JSON_PATH = "/mnt/nvme0n1/ruofan/git_space/Empirical-Influence-Function/experiment_results.jsonl"

    accelerator = Accelerator()
    set_seed(42)

    if accelerator.is_main_process:
        logger.info(f"Processes: {accelerator.state.num_processes}")

    # Load Model
    # 注意：在多卡环境下，显式指定 device_map={"": device} 有助于避免 Accelerate 自动分配冲突
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": accelerator.device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # target_layer_keyword =
    model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.detach().clone())
    model.config.tie_word_embeddings = False

    # Freeze & Unfreeze
    target_layer_keywords = ["lm_head"]
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(key in name for key in target_layer_keywords):
            param.requires_grad = True

    if model.lm_head.weight.requires_grad:
        logger.info("Weight Tying confirmed: lm_head is automatically unfrozen via embed_tokens.")

    def filter_params(n, p):
        return any(key in n for key in target_layer_keywords) and p.requires_grad

    # Load Data
    train_texts = []
    seen_inputs  = set()
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if len(obj['messages']) > 2 and len(obj['messages'][2]['content']) > 0:
                    sys  = obj['messages'][0]['content']
                    inp  = obj['messages'][1]['content']
                    outp = obj['messages'][2]['content']
                    if inp in seen_inputs:
                        continue
                    train_texts.append(
                        {
                            'system': sys,
                            'input':  inp,
                            'output': outp
                        }
                    )
                    seen_inputs.add(inp)
            if len(train_texts) >= 1000:
                break

    test_texts = []
    for idx in range(len(train_texts)):
        perturbed_idx = (idx + 1) % len(train_texts)
        test_texts.append(
            {
                "system": train_texts[idx]["system"],
                "input":  train_texts[idx]["input"],
                "output": train_texts[perturbed_idx]["output"],
            }
        )
        if len(test_texts) >= 100:
            break

    # Dataset Preparation
    process_func = partial(process_func_chatml, tokenizer=tokenizer)
    train_ds = Dataset.from_dict(list_of_dicts_to_dict_of_lists(train_texts))
    # 保留原始 index!
    train_ds = train_ds.map(lambda x, i: {"sample_index": i}, with_indices=True)
    train_ds = train_ds.map(process_func, batched=True, remove_columns=["input", "output", "system"])
    train_ds.set_format(type="torch", columns=["input_ids", "labels", "sample_index"])

    # Collator Setup
    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )
    collator = CustomCollator(base_collator)

    BATCH_SIZE = 2
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 建议 False 以便 debug，若 True 也不影响 index tracking
        collate_fn=collator
    )

    # Prepare DataLoader
    train_loader  = accelerator.prepare(train_loader)
    total_samples = len(train_texts)

    eif = EmpiricalIF(train_loader, model, tokenizer, accelerator, filter_params,
                      debug_on=False)

    self_ranks = []
    # if accelerator.is_main_process:
    #     if os.path.exists(RESULTS_JSON_PATH):
    #         shutil.move(RESULTS_JSON_PATH, f"{RESULTS_JSON_PATH}.bak")
    #     logger.info(f"Results will be streamed to {RESULTS_JSON_PATH}")

    for i in tqdm(range(len(test_texts)), desc="Running Experiments"):

        if i not in [89, 60, 98, 32, 67, 37, 86]:
            continue
        test_sample_dict = test_texts[i]

        # 临时处理 Query Batch
        temp_ds = Dataset.from_dict({
            "input":  [test_sample_dict["input"]],
            "output": [test_sample_dict["output"]],
            "system": [test_sample_dict["system"]]
        })
        temp_ds = temp_ds.map(process_func, batched=True, remove_columns=["input", "output", "system"])

        query_batch = base_collator([temp_ds[0]])
        for k, v in query_batch.items():
            query_batch[k] = v.to(accelerator.device)

        global_train_diffs = None
        query_token_diffs = None

        # 运行 Influence Analysis
        # lr 可以适当大一点，因为我们只更新很少的步数
        # scores, indices, global_train_diffs, query_token_diffs = eif.query_influence(
        #     query_batch, lr=5e-4, max_steps=10, loss_threshold=0.6
        # )

        scores, indices, *_ = eif.query_resonance_influence(
            query_batch, lr=5e-4, max_steps=5,
        )

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
                # if idx == 0:

            self_ranks.append(rank_pos + 1)
            percentile = (rank_pos + 1) / total_samples
            entry = {
                "query_index": i,
                "test_text": test_texts[i],  # 包含 system, input, output
                "self_rank": rank_pos + 1,
                "percentile": percentile,
                "score": float(self_score)  # 确保转为 float 以便 JSON 序列化
            }
            # with open(RESULTS_JSON_PATH, "a", encoding="utf-8") as f:
            #     f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # # 准备 Top 5 Harmful 数据
            top_5_harmful = sorted_results[-5:][::-1]
            top_5_indices = [idx for idx, score in top_5_harmful]
            top_5_scores = [score for idx, score in top_5_harmful]
            # 准备 Top 5 Helpful 数据
            top_5_helpful = sorted_results[:5]
            top_5_helpful_indices = [idx for idx, score in top_5_helpful]
            top_5_helpful_scores = [score for idx, score in top_5_helpful]

            # 保存 HTML 报告
            save_query_report_html(
                query_idx=i,
                query_batch=query_batch,  # 传入 Query Batch
                train_dataset=train_ds,  # 传入训练集 Dataset
                rank_pos=rank_pos,
                percentile=percentile,
                score=self_score,
                tokenizer=tokenizer,
                train_token_diffs_dict=global_train_diffs,
                query_token_diffs=query_token_diffs,
                top_5_harmful_indices=top_5_indices,  # Top 5 索引
                top_5_harmful_scores=top_5_scores,  # Top 5 分数
                top_5_helpful_indices=top_5_helpful_indices,  # Top 5 索引
                top_5_helpful_scores=top_5_helpful_scores,  # Top 5 分数
                output_dir="influence_reports",
                # output_dir="influence_reports_good",
                lr=5e-4,
                max_steps=5,
                model=model,
                param_filter_fn=filter_params,
                enable_coloring = False,
            )

            # 转换为 numpy 数组方便计算
            ranks_arr = np.array(self_ranks)

            # 计算统计量
            min_rank = np.min(ranks_arr)
            max_rank = np.max(ranks_arr)
            median_rank = np.median(ranks_arr)

            print("\n" + "=" * 60)
            print("🧪 EXPERIMENT REPORT: Mismatched Query (Self-Input + Other-Output)")
            print(f"Target Layer         : {target_layer_keywords}")
            print(f"Total Samples Tested : {len(self_ranks)}")
            print("-" * 60)
            print(f"📉 Min Rank          : {min_rank:.0f}")
            print(f"📈 Max Rank          : {max_rank:.0f}")
            print(f"⚖️  Median Rank      : {median_rank:.1f}")


    if accelerator.is_main_process:

        if not self_ranks:
            logger.error("No valid ranks collected!")
            return

        # 转换为 numpy 数组方便计算
        ranks_arr = np.array(self_ranks)

        # 计算统计量
        min_rank    = np.min(ranks_arr)
        max_rank    = np.max(ranks_arr)
        median_rank = np.median(ranks_arr)

        print("\n" + "=" * 60)
        print("🧪 EXPERIMENT REPORT: Mismatched Query (Self-Input + Other-Output)")
        print(f"Target Layer         : {target_layer_keywords}")
        print(f"Total Samples Tested : {len(self_ranks)}")
        print("-" * 60)
        print(f"📉 Min Rank          : {min_rank:.0f}")
        print(f"📈 Max Rank          : {max_rank:.0f}")
        print(f"⚖️  Median Rank      : {median_rank:.1f}")


if __name__ == '__main__':
    main()

    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 IF_HF.py



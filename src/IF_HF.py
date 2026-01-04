import torch
import logging
from typing import List, Dict, Callable, Optional, Any, Callable
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
import random
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 类型别名
ParamFilterFn = Callable[[str, nn.Parameter], bool]


class EmpiricalIF:

    def __init__(self, dl_train, model, tokenizer, accelerator, param_filter_fn=None, debug_on=False):
        self.dl_train = dl_train
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.device = accelerator.device
        self.param_filter_fn = param_filter_fn
        ids_list = tokenizer.convert_tokens_to_ids(['Ċ', 'Ġ', 'Ä'])  # 这里填入你想忽略的 token 字符串
        self.ignored_token_ids = torch.tensor(ids_list, device=self.device)
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
        # self.base_train_results = self._get_train_losses()
        self.base_train_results = None
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def get_param_snapshot(model: nn.Module, param_filter_fn: Optional[ParamFilterFn]) -> List[torch.Tensor]:
        snapshot = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param_filter_fn is None or param_filter_fn(name, param):
                    snapshot.append(param.detach().cpu().clone())
        return snapshot

    @staticmethod
    def restore_params(model: nn.Module, snapshot: List[torch.Tensor], param_filter_fn: Optional[ParamFilterFn]):
        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param_filter_fn is None or param_filter_fn(name, param):
                    param.data.copy_(snapshot[idx].to(param.device))
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
                mean_loss, token_loss = compute_loss_per_sample(self.model, batch_gpu, self.device, self.ignored_token_ids)

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
        self.model.zero_grad(set_to_none=True)

        # ================== 1. 计算 Query Base Loss (含 Token-level) ==================
        with torch.no_grad():
            l_test_base_scalar, l_test_base_tokenwise_raw = compute_loss_per_sample(
                self.model, query_batch, self.device, self.ignored_token_ids
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
            grads = compute_gradients(self.model, query_batch, self.param_filter_fn, self.device, self.ignored_token_ids)
            self.apply_gradient_update(self.model, grads, self.param_filter_fn, lr=lr)
            self.model.zero_grad(set_to_none=True)

            with torch.no_grad():
                loss_s, _ = compute_loss_per_sample(self.model, query_batch, self.device, self.ignored_token_ids)
                curr_test_loss = loss_s.item()

            if self.accelerator.is_main_process:
                logger.info(f"Gradient Descent for {step} Steps, the Testing Loss Decreased to {curr_test_loss}...")

        # ================== 4. 计算 Query Descent Loss (含 Token-level) ==================
        with torch.no_grad():
            _, l_test_des_tokenwise_raw = compute_loss_per_sample(self.model, query_batch, self.device, self.ignored_token_ids)
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
        if self.base_train_results is None:
            self.base_train_results = self._get_train_losses()
            gc.collect()
            torch.cuda.empty_cache()

        # 备份原始参数
        snapshot_theta0 = self.get_param_snapshot(self.model, self.param_filter_fn)

        l_train_theta0, indices_local, _ = self.base_train_results

        with torch.no_grad():
            l_query_theta0, _ = compute_loss_per_sample(self.model, query_batch, self.device, self.ignored_token_ids)
            l_query_theta0 = l_query_theta0.item()

        # --- Stage A: Query 驱动 (θ0 -> θQ) ---
        self.model.zero_grad(set_to_none=True)
        for step in range(max_steps):
            grads_q = compute_gradients(self.model, query_batch, self.param_filter_fn, self.device, self.ignored_token_ids)
            self.apply_gradient_update(self.model, grads_q, self.param_filter_fn, lr=lr)
            self.model.zero_grad(set_to_none=True)

        # 记录 Q 更新后的基准
        snapshot_thetaQ = self.get_param_snapshot(self.model, self.param_filter_fn)

        with torch.no_grad():
            # 学习 Q 后的 Q loss
            l_query_thetaQ, _ = compute_loss_per_sample(self.model, query_batch, self.device, self.ignored_token_ids)
            l_query_thetaQ = l_query_thetaQ.item()

            # 学习 Q 后的所有训练样本 Loss (用于 Stage B 的起点)
            l_train_thetaQ, _, _ = self._get_train_losses()

            # ΔL_i|q: Q 对 i 的原生拉动力 (相对于 θ0)
            delta_i_from_theta0_to_thetaQ = (l_train_thetaQ - l_train_theta0).cpu()
            # ΔL_q|q: Q 的自优化量
            delta_q_by_q = l_query_thetaQ - l_query_theta0

        # --- Stage B: 条件探测 (θQ -> θQ+i) ---
        local_polarized_scores = []
        idx_counter = 0

        for batch in tqdm(self.train_batches, desc="Conditioned Resonance Probing"):
            batch_size = batch['sample_index'].size(0)
            for j in range(batch_size):
                single_sample = {k: v[j:j + 1].to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                # 当前起点：已经学过 Q 后的样本 i 的 Loss
                l_train_thetaQ_i = l_train_thetaQ[idx_counter].item()

                # 模拟学习样本 i fixme: I only update for 1 step, but timed learning rate by max_steps
                self.model.zero_grad(set_to_none=True)
                for step in range(1):
                    grads_i = compute_gradients(self.model, single_sample, self.param_filter_fn, self.device, self.ignored_token_ids)
                    self.apply_gradient_update(self.model, grads_i, self.param_filter_fn, lr=lr * max_steps)
                    self.model.zero_grad(set_to_none=True)

                with torch.no_grad():
                    # 1. 条件自优化: ΔL_i|(i,q)
                    l_train_thetai, _ = compute_loss_per_sample(self.model, single_sample, self.device, self.ignored_token_ids)
                    delta_i_by_i = l_train_thetai.item() - l_train_thetaQ_i

                    # 2. 条件对 Q 影响: ΔL_q|(i,q)
                    l_query_thetai, _ = compute_loss_per_sample(self.model, query_batch, self.device, self.ignored_token_ids)
                    delta_q_by_i = l_query_thetai.item() - l_query_thetaQ

                # --- 效能计算 ---
                delta_i_by_q = delta_i_from_theta0_to_thetaQ[idx_counter].item()

                # 归一化：帮了对方多少 / 自己能跑多远
                eff_q_to_i = delta_i_by_q / (abs(delta_i_by_i) + 1e-8)
                eff_i_to_q = delta_q_by_i / (abs(delta_q_by_q) + 1e-8)

                # 判定：如果 delta_q_by_i > 0，说明 i 导致 Q 的 Loss 反弹，即 Harmful
                resonance_sign = 1.0 if delta_q_by_i < 0 else -1.0

                magnitude   = torch.sqrt(torch.tensor(abs(eff_q_to_i * eff_i_to_q)))
                final_score = resonance_sign * magnitude.item()

                # 一致性门控：若 Q 帮 i 的方向与 i 帮 Q 的方向相反（一个降一个升），判定为噪声
                if (delta_i_by_q < 0) != (delta_q_by_i < 0):
                    final_score = 0.0

                local_polarized_scores.append(final_score)

                # 恢复到 Stage A 的终点 (θQ)
                self.restore_params(self.model, snapshot_thetaQ, self.param_filter_fn)
                idx_counter += 1

            gc.collect()
            torch.cuda.empty_cache()

        # 恢复原始参数 θ0 (不影响后续其他 Query 的探测)
        self.restore_params(self.model, snapshot_theta0, self.param_filter_fn)
        gc.collect()
        torch.cuda.empty_cache()

        # 全局通信汇总 (修正索引对齐问题)
        all_scores = self.accelerator.gather(torch.tensor(local_polarized_scores, device=self.device))
        all_indices = self.accelerator.gather(indices_local)

        return all_scores.cpu().tolist(), all_indices.cpu().tolist(), None, None


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
        try:
            response_part = full_text.split("assistant\n")[-1]
            return response_part.replace('\n', '<br>').replace(' ', '&nbsp;')
        except:
            return full_text.replace('\n', '<br>')

    html_content = ""
    threshold = 0  # 只要有变化就着色
    diffs_list = diffs_tensor.tolist() if diffs_tensor is not None else []

    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
    assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")

    # 找到 Output 开始的位置
    output_start_idx = 0
    input_ids_cpu = input_ids.cpu()

    for i in range(len(input_ids_cpu) - 1):
        if input_ids_cpu[i] == assistant_token_id:
            output_start_idx = i + 1
            if output_start_idx < len(input_ids_cpu):
                next_token_id = input_ids_cpu[output_start_idx].item()
                decoded_next = tokenizer.decode(next_token_id)
                if decoded_next in ['\n', 'Ċ', 'Ġ\n']:
                    output_start_idx += 1
            break

    for i in range(output_start_idx, len(tokens)):
        t = tokens[i]
        if t in [tokenizer.pad_token, tokenizer.eos_token, '<|im_end|>']:
            break

        # 处理特殊字符显示
        display_t = t.replace('Ċ', '\n').replace('Ġ', ' ').replace('Ä‰', '\t')
        display_t = display_t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        display_t = display_t.replace('\n', '<br>').replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;').replace(' ', '&nbsp;')

        # --- 获取 Loss Diff (关键修改) ---
        # 你的 input_ids 长度是 N
        # 你的 diffs_list (即 token_losses) 长度是 N-1
        # token[i] 的 loss 对应的是 diffs_list[i-1]

        current_diff = 0.0
        diff_index = i - 1  # Shift logic

        if 0 <= diff_index < len(diffs_list):
            current_diff = diffs_list[diff_index]

        # 着色逻辑
        if current_diff > threshold:
            # Red: Loss Increased
            html_content += f'<span style="background-color: #ffebee; color: #c62828; font-weight: bold; border-radius: 2px; border-bottom: 1px solid #ffcdd2;">{display_t}</span>'
        elif current_diff < threshold:  # 注意这里要是负的 threshold
            # Blue: Loss Decreased
            html_content += f'<span style="background-color: #e3f2fd; color: #1565c0; font-weight: bold; border-radius: 2px; border-bottom: 1px solid #bbdefb;">{display_t}</span>'
        else:
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
        top_5_harmful_indices: List[int],
        top_5_harmful_scores: List[float],
        top_5_helpful_indices: List[int],
        top_5_helpful_scores: List[float],
        model: nn.Module,
        collator: Callable,
        param_filter_fn,
        lr,
        max_steps,
        output_dir: str = "reports",
):
    # ================= 1. 准备数据阶段 =================
    model.eval()
    all_indices = [query_idx] + top_5_helpful_indices + top_5_harmful_indices
    # 对应的分数列表，用于展示
    all_scores = [score] + top_5_helpful_scores + top_5_harmful_scores
    train_raw_samples = [train_dataset[i] for i in all_indices]
    ids_list = tokenizer.convert_tokens_to_ids(['Ċ','Ġ', 'Ä'])  # 这里填入你想忽略的 token 字符串
    ignored_token_ids = torch.tensor(ids_list, device=self.device)

    def compute_loss_in_minibatches(samples_list, batch_size=2):
        all_samples_loss_list = []  # 存储每个样本的 1D Tensor

        # 手动分批
        for i in range(0, len(samples_list), batch_size):
            batch_samples = samples_list[i: i + batch_size]
            batch = collator(batch_samples)

            # 移到 GPU
            batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                # token_loss shape: [Current_Batch_Size, Current_Seq_Len]
                _, token_loss = compute_loss_per_sample(model, batch, model.device, ignored_token_ids)

            # 立即上 CPU
            token_loss_cpu = token_loss.detach().cpu()

            # 【关键】把 Batch 拆开，按样本存入列表
            all_samples_loss_list.extend(token_loss_cpu.unbind(0))
            del batch

        return all_samples_loss_list

    # 临时重演扰动
    snapshot_theta0 = EmpiricalIF.get_param_snapshot(model, param_filter_fn)
    with torch.no_grad():
        # 【修正】这里得到的是 List[Tensor]，列表长度 = 11 (样本数)
        l_train_token_loss_theta0 = compute_loss_in_minibatches(train_raw_samples, batch_size=2)

    # 1. Query 驱动 (θ0 -> θQ) ---
    for step in range(max_steps):
        model.zero_grad(set_to_none=True)
        grads_q = compute_gradients(model, query_batch, param_filter_fn, model.device, ignored_token_ids)
        EmpiricalIF.apply_gradient_update(model, grads_q, param_filter_fn, lr=lr)
        del grads_q

    # 2. 计算 Theta Q 状态下的 Training Loss (Batch 推理)
    model.eval()
    with torch.no_grad():
        l_train_token_loss_thetaQ = compute_loss_in_minibatches(train_raw_samples, batch_size=2)

        _, l_query_token_loss_thetaQ = compute_loss_per_sample(model, query_batch, model.device, ignored_token_ids)
        l_query_token_loss_thetaQ = l_query_token_loss_thetaQ.detach()  # 保持在 GPU 但切断梯度

    # 3. 计算 Delta train from theta0 to thetaQ
    delta_train_token_loss_theta0_to_thetaQ = [
        loss_q - loss_0
        for loss_q, loss_0 in zip(l_train_token_loss_thetaQ, l_train_token_loss_theta0)
    ]

    # 4. 计算 Delta query from thetaQ to thetai
    snapshot_thetaQ = EmpiricalIF.get_param_snapshot(model, param_filter_fn)
    delta_query_token_loss_thetaQ_to_thetai = []

    for j, raw_sample in tqdm(enumerate(train_raw_samples), total=len(train_raw_samples), desc="Report Probing"):
        # 显式清理
        model.zero_grad(set_to_none=True)

        # 构造单样本 Batch
        single_batch = collator([raw_sample])
        single_batch = {k: v.to(model.device) for k, v in single_batch.items() if isinstance(v, torch.Tensor)}

        # 模拟学习
        grads_i = compute_gradients(model, single_batch, param_filter_fn, model.device, ignored_token_ids)
        EmpiricalIF.apply_gradient_update(model, grads_i, param_filter_fn, lr=lr * max_steps)
        del grads_i  # 立即删梯度

        with torch.no_grad():
            _, l_query_token_loss_thetai = compute_loss_per_sample(model, query_batch, model.device, ignored_token_ids)

        diff = l_query_token_loss_thetai[0] - l_query_token_loss_thetaQ[0]
        delta_query_token_loss_thetaQ_to_thetai.append(diff.detach().cpu()) # 【关键】上 CPU

        # 恢复到 θQ
        EmpiricalIF.restore_params(model, snapshot_thetaQ, param_filter_fn)
        # 7. 【关键】清理本轮产生的中间变量
        del single_batch
        del l_query_token_loss_thetai
        del diff

        # 8. 激进的显存清理 (每 N 轮清理一次，或者每轮清理防止 OOM)
        if j % 1 == 0:
            torch.cuda.empty_cache()

    delta_query_token_loss_thetaQ_to_thetai = torch.stack(delta_query_token_loss_thetaQ_to_thetai)
    EmpiricalIF.restore_params(model, snapshot_theta0, param_filter_fn) # restore to original
    # 最后的清理
    del snapshot_theta0
    del snapshot_thetaQ
    gc.collect()
    torch.cuda.empty_cache()
    # ================= 2. 渲染 HTML 报告阶段 =================

    # 2.1 准备 Query 自身的静态信息
    query_input_ids = query_batch["input_ids"][0].cpu()
    query_full_text = tokenizer.decode(query_input_ids, skip_special_tokens=True)
    try:
        query_input_text = query_full_text.split("assistant\n")[0].strip()
    except:
        query_input_text = "Error extracting input text."

    # Query 的输出文本（不做 Diff 着色，仅展示原文，或者你可以选择用 θQ 的 loss 着色，这里暂存原文）
    # 为了保持一致性，这里我们展示无着色的 Query 输出作为 Reference
    query_output_html_plain = get_colored_html_from_ids(
        tokenizer=tokenizer,
        input_ids=query_input_ids,
        diffs_tensor=None,  # 不着色
        enable_coloring=False
    )

    # HTML Header
    html_template = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1400px; margin: 20px auto; padding: 20px; background-color: #f8f9fa; }}
                .card {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 30px; border-left: 5px solid #3498db; }}
                .card.harmful {{ border-left-color: #e74c3c; }}
                .card.helpful {{ border-left-color: #2ecc71; }}
                h1 {{ color: #2c3e50; text-align: center; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; margin-top: 0; }}
                h3 {{ color: #7f8c8d; font-size: 1.1em; margin-bottom: 5px; }}
                .metrics {{ display: flex; justify-content: center; gap: 30px; font-weight: bold; color: #555; background: #fff; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
                pre {{ background: #2d3436; color: #dfe6e9; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: 'Consolas', monospace; white-space: pre-wrap; word-break: break-all; }}
                .code-box {{
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                    white-space: pre-wrap;
                    word-break: break-all;
                    line-height: 1.6;
                    background: #ffffff;
                    padding: 15px;
                    border: 1px solid #dfe6e9;
                    border-radius: 5px;
                    margin-top: 5px;
                }}
                .label {{ font-weight: bold; margin-top: 15px; display: block; color: #2c3e50; text-transform: uppercase; font-size: 0.85em; letter-spacing: 0.5px; }}
                .row {{ display: flex; gap: 20px; }}
                .col {{ flex: 1; min-width: 0; }} /* min-width 0 prevents flex item overflow */
                .sub-title {{ font-size: 0.9em; color: #7f8c8d; margin-bottom: 5px; font-weight: bold; }}
                .legend {{ text-align: center; font-size: 0.9em; color: #666; margin-bottom: 20px; }}
                .legend span.red {{ color: #c0392b; font-weight: bold; background: #ffebee; padding: 2px 5px; border-radius: 3px; }}
                .legend span.blue {{ color: #2980b9; font-weight: bold; background: #e3f2fd; padding: 2px 5px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h1>Influence Analysis Report</h1>

            <div class="metrics">
                <span>Query Index: {query_idx}</span>
                <span>Self Rank: {rank_pos + 1} / {len(train_dataset)}</span>
                <span>Percentile: {percentile:.2%}</span>
                <span>Self Score: {score:.6e}</span>
            </div>

            <div class="legend">
                Color Legend: <span class="red">Red (Loss Increased / Harmful)</span> | <span class="blue">Blue (Loss Decreased / Helpful)</span>
            </div>

            <div class="card">
                <h2>Target Query Sample</h2>
                <span class="label">Query Input:</span>
                <pre>{query_input_text}</pre>
                <span class="label">Query Output (Reference):</span>
                <div class="code-box">{query_output_html_plain}</div>
            </div>

            <h2>Detailed Influence Analysis</h2>
        """

    # 2.2 循环渲染每个样本
    # all_indices 顺序: [Query_Itself, Helpful_1...5, Harmful_1...5]
    # 我们需要区分类型给 Card 加不同的 CSS class

    for j, (idx, current_score) in enumerate(zip(all_indices, all_scores)):

        # 确定卡片类型 (用于样式)
        card_class = ""
        type_label = ""
        if j == 0:
            card_class = ""
            type_label = "Self (Original Target)"
        elif j <= 5:
            card_class = "helpful"
            type_label = f"Top Helpful #{j}"
        else:
            card_class = "harmful"
            type_label = f"Top Harmful #{j - 5}"

        # --- 准备 Train Sample 数据 ---
        raw_sample = train_raw_samples[j]
        train_input_ids = raw_sample["input_ids"]  # 已经是 Tensor
        if isinstance(train_input_ids, torch.Tensor):
            train_input_ids = train_input_ids.cpu()

        train_full_text = tokenizer.decode(train_input_ids, skip_special_tokens=True)
        try:
            train_input_str = train_full_text.split("assistant\n")[0].strip()
        except:
            train_input_str = train_full_text

        # 1. Train Output Coloring (Effect of Query on Train Sample)
        # Source: delta_train_token_loss_theta0_to_thetaQ[j]
        train_diffs = delta_train_token_loss_theta0_to_thetaQ[j].cpu()
        train_output_colored = get_colored_html_from_ids(
            tokenizer, train_input_ids, train_diffs, enable_coloring=True
        )

        # 2. Query Output Coloring (Effect of Train Sample on Query)
        # Source: delta_query_token_loss_thetaQ_to_thetai[j]
        # 注意：这里我们要再次用到 Query 的 Input IDs
        query_diffs = delta_query_token_loss_thetaQ_to_thetai[j].cpu()
        query_output_colored_by_train = get_colored_html_from_ids(
            tokenizer, query_input_ids, query_diffs, enable_coloring=True
        )

        # 拼接 HTML
        html_template += f"""
            <div class="card {card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <h2>{type_label} <span style="font-size:0.7em; color:#999;">(Index {idx})</span></h2>
                    <span style="font-weight:bold; color:#333;">Influence Score: {current_score:.6e}</span>
                </div>

                <span class="label">1. Training Sample Input:</span>
                <pre>{train_input_str}</pre>

                <div class="row">
                    <div class="col">
                        <span class="label">2. Effect of Query on This Sample</span>
                        <div class="sub-title">How did learning the Query change this sample's loss?</div>
                        <div class="code-box">{train_output_colored}</div>
                    </div>
                    <div class="col">
                        <span class="label">3. Effect of This Sample on Query</span>
                        <div class="sub-title">How did learning this sample change the Query's loss?</div>
                        <div class="code-box">{query_output_colored_by_train}</div>
                    </div>
                </div>
            </div>
            """

    html_template += "</body></html>"

    # 保存文件
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/query_{query_idx}.html", "w", encoding="utf-8") as f:
        f.write(html_template)

    logger.info(f"Report saved to {output_dir}/query_{query_idx}.html")

def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # 路径配置
    JSONL_PATH = "/mnt/nvme0n1/ruofan/git_space/Empirical-Influence-Function/big_data_L_2_5.jsonl"  # 请确保文件存在
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
        if len(test_texts) >= 50:
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
    LR = 5e-3
    MAX_STEPS = 5

    self_ranks = []
    self_scores = []
    if accelerator.is_main_process:
        if os.path.exists(RESULTS_JSON_PATH):
            shutil.move(RESULTS_JSON_PATH, f"{RESULTS_JSON_PATH}.bak")
        logger.info(f"Results will be streamed to {RESULTS_JSON_PATH}")

    for i in tqdm(range(len(test_texts)), desc="Running Experiments"):

        # if i not in [48, 19, 5, 22, 38, 39, 41, 12, 29, 3]:
        #     continue
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

        scores, indices, *_ = eif.query_resonance_influence(
            query_batch, lr=LR, max_steps=MAX_STEPS,
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

            self_ranks.append(rank_pos + 1)
            self_scores.append(self_score)
            percentile = (rank_pos + 1) / total_samples
            entry = {
                "query_index": i,
                "test_text": test_texts[i],  # 包含 system, input, output
                "self_rank": rank_pos + 1,
                "percentile": percentile,
                "score": float(self_score)  # 确保转为 float 以便 JSON 序列化
            }
            with open(RESULTS_JSON_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # # # 准备 Top 5 Harmful 数据
            # top_5_harmful = sorted_results[-5:][::-1]
            # top_5_indices = [idx for idx, score in top_5_harmful]
            # top_5_scores = [score for idx, score in top_5_harmful]
            # # 准备 Top 5 Helpful 数据
            # top_5_helpful = sorted_results[:5]
            # top_5_helpful_indices = [idx for idx, score in top_5_helpful]
            # top_5_helpful_scores = [score for idx, score in top_5_helpful]
            #
            # # 保存 HTML 报告
            # save_query_report_html(
            #     query_idx=i,
            #     query_batch=query_batch,  # 传入 Query Batch
            #     train_dataset=train_ds,  # 传入训练集 Dataset
            #     rank_pos=rank_pos,
            #     percentile=percentile,
            #     score=self_score,
            #     tokenizer=tokenizer,
            #     top_5_harmful_indices=top_5_indices,  # Top 5 索引
            #     top_5_harmful_scores=top_5_scores,  # Top 5 分数
            #     top_5_helpful_indices=top_5_helpful_indices,  # Top 5 索引
            #     top_5_helpful_scores=top_5_helpful_scores,  # Top 5 分数
            #     output_dir="influence_reports",
            #     lr=LR,
            #     max_steps=MAX_STEPS,
            #     model=model,
            #     collator=collator,
            #     param_filter_fn=filter_params,
            # )

            # 转换为 numpy 数组方便计算
            ranks_arr = np.array(self_ranks)

            # 计算统计量
            min_rank = np.min(ranks_arr)
            max_rank = np.max(ranks_arr)
            median_rank = np.median(ranks_arr)
            median_score = np.median(np.array(self_scores))

            print("\n" + "=" * 60)
            print("🧪 EXPERIMENT REPORT: Mismatched Query (Self-Input + Other-Output)")
            print(f"Target Layer         : {target_layer_keywords}")
            print(f"Total Samples Tested : {len(self_ranks)}")
            print("-" * 60)
            print(f"📉 Min Rank          : {min_rank:.0f}")
            print(f"📈 Max Rank          : {max_rank:.0f}")
            print(f"⚖️  Median Rank      : {median_rank:.1f}")
            print(f"Median score         : {median_score:.1f} ")


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
        median_score = np.median(np.array(self_scores))

        print("\n" + "=" * 60)
        print("🧪 EXPERIMENT REPORT: Mismatched Query (Self-Input + Other-Output)")
        print(f"Target Layer         : {target_layer_keywords}")
        print(f"Total Samples Tested : {len(self_ranks)}")
        print("-" * 60)
        print(f"📉 Min Rank          : {min_rank:.0f}")
        print(f"📈 Max Rank          : {max_rank:.0f}")
        print(f"⚖️  Median Rank      : {median_rank:.1f}")
        print(f"Median score         : {median_score:.1f} ")

if __name__ == '__main__':
    main()

    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 IF_HF.py



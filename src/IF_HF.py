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
from loss import compute_loss_per_sample, compute_gradients, compute_token_specific_update, compute_loss_in_minibatches, get_first_response_token
from vis import get_colored_html_from_ids, get_attention_html, get_query_html_with_highlight
from attribution import attention_attribution_static, attention_attribution_on_generation, gradient_saliency_static, gradient_saliency_on_generation, model_generation
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
        # ids_list = tokenizer.convert_tokens_to_ids(["\\t", "\\n"])  # 这里填入你想忽略的 token 字符串
        ids_list = []
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
    @torch.no_grad()
    def get_param_snapshot(model: nn.Module, param_filter_fn: Optional[ParamFilterFn]) -> List[torch.Tensor]:
        snapshot = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param_filter_fn is None or param_filter_fn(name, param):
                    snapshot.append(param.detach().cpu().clone())
        return snapshot

    @staticmethod
    @torch.no_grad()
    def restore_params(model: nn.Module, snapshot: List[torch.Tensor], param_filter_fn: Optional[ParamFilterFn]):
        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param_filter_fn is None or param_filter_fn(name, param):
                    param.data.copy_(snapshot[idx].to(param.device))
                    idx += 1

    @staticmethod
    @torch.no_grad()
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


def save_query_report_html_attn(
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
    all_indices = [query_idx] + top_5_harmful_indices + top_5_helpful_indices
    all_scores  = [score] + top_5_harmful_scores + top_5_helpful_scores
    train_raw_samples = [train_dataset[i] for i in all_indices]

    ids_list = []
    ignored_token_ids = torch.tensor(ids_list, device=model.device)

    # record theta0
    snapshot_theta0 = EmpiricalIF.get_param_snapshot(model, param_filter_fn)
    # get all training loss at theta0 (tokenwise)
    l_train_token_loss_theta0 = compute_loss_in_minibatches(
        model, collator, train_raw_samples, ignored_token_ids, batch_size=2
    )

    # locate the 1st response token in query
    query_response_start_idx, query_response_len = get_first_response_token(query_batch, ignored_token_ids)
    if query_response_start_idx is None:
        return
    inference_train_tokens_theta0 = []

    for j, raw_sample in enumerate(train_raw_samples):
        single_batch = collator([raw_sample])
        single_batch = {k: v.to(model.device) for k, v in single_batch.items() if isinstance(v, torch.Tensor)}

        # locate the 1st response token in training
        train_response_start_idx, _ = get_first_response_token(single_batch, ignored_token_ids)

        # do real inference on training
        train_new_generated_tokens = model_generation(
            model,
            tokenizer,
            single_batch,
            train_response_start_idx,
            max_new_tokens=50,
        )

        generated_ids_cpu = train_new_generated_tokens[0]
        inference_train_tokens_theta0.append(generated_ids_cpu)

    # ================= 2. Loop per Query Token (i) =================
    for i in tqdm(range(min(query_response_len, 5)), desc=f"Analyzing Query {query_idx} Tokens"):
        target_sequence_idx = query_response_start_idx + i + 1  # Now I am predicting query_response_start_idx + i + 1

        # Reset to Theta_0
        EmpiricalIF.restore_params(model, snapshot_theta0, param_filter_fn)

        # --- A. Update to Theta_{Q,i} ---
        for _ in range(max_steps):
            grads_q_i = compute_token_specific_update(
                model=model,
                batch=query_batch,
                param_filter_fn=param_filter_fn,
                device=model.device,
                ignored_token_ids=ignored_token_ids,
                target_sequence_idx=target_sequence_idx,
                lr=lr
            )
            EmpiricalIF.apply_gradient_update(model, grads_q_i, param_filter_fn, lr=lr)
            del grads_q_i

        snapshot_thetaQi = EmpiricalIF.get_param_snapshot(model, param_filter_fn)

        # --- B. Compute Baseline Loss at Theta_{Q,i} ---
        model.eval()
        with torch.no_grad():
            l_train_token_loss_thetaQi = compute_loss_in_minibatches(
                model, collator, train_raw_samples, ignored_token_ids, batch_size=2
            )
            _, l_query_token_loss_thetaQi = compute_loss_per_sample(
                model, query_batch, model.device, ignored_token_ids
            )
            l_query_token_loss_thetaQi = l_query_token_loss_thetaQi.detach()

        # --- C. Compute Loss Diff (Train Samples) ---
        delta_train_token_loss_theta0_to_thetaQi = [
            loss_q - loss_0
            for loss_q, loss_0 in zip(l_train_token_loss_thetaQi, l_train_token_loss_theta0)
        ]

        # =========================================================
        # Compute Query Attention (Static Reference)
        # =========================================================
        query_attn_cpu = gradient_saliency_static(
            model,
            query_batch,
            target_sequence_idx
        )

        # 生成静态 HTML
        query_context_ids      = query_batch["input_ids"][0, :target_sequence_idx].cpu()
        static_query_attn_html = get_attention_html(
            tokenizer,
            query_context_ids,
            query_attn_cpu
        )

        # =========================================================
        # [Step D] Inference & Visualization on Training Samples
        # =========================================================
        inference_vis_htmls = []
        inference_train_tokens_thetaQi = []

        for j, raw_sample in enumerate(train_raw_samples):
            # 准备单样本 Batch
            single_batch = collator([raw_sample])
            single_batch = {k: v.to(model.device) for k, v in single_batch.items() if isinstance(v, torch.Tensor)}

            train_response_start_idx, _ = get_first_response_token(single_batch, ignored_token_ids)

            # Generate (Inference)
            train_new_generated_tokens, step_wise_attentions = gradient_saliency_on_generation(
                model,
                tokenizer,
                single_batch,
                train_response_start_idx,
                max_new_tokens=50,
            )

            # Build HTML Components
            train_input_ids_cpu = single_batch["input_ids"][0, :train_response_start_idx].cpu()
            generated_ids_cpu = train_new_generated_tokens[0]
            inference_train_tokens_thetaQi.append(generated_ids_cpu)

            unique_sample_id = f"sample_{j}_token_{i}"
            buttons_html = ""
            frames_html = ""
            current_context_ids = train_input_ids_cpu.tolist()

            for step_k in range(len(step_wise_attentions)):
                gen_token_id = generated_ids_cpu[step_k].item()
                gen_token_str = tokenizer.decode(gen_token_id)
                display_token = gen_token_str.replace('\n', '\\n').replace(' ', ' ')
                if not display_token.strip():
                    display_token = "&nbsp;"

                active_class = "active" if step_k == 0 else ""

                # Button
                buttons_html += f"""
                <div class="gen-token-btn {active_class}" onclick="showStep('{unique_sample_id}', {step_k})">
                     {step_k + 1}. {display_token}
                </div>
                """

                # Left Heatmap (Dynamic Train)
                attn_weights = step_wise_attentions[step_k]
                train_heatmap_html = get_attention_html(
                    tokenizer,
                    torch.tensor(current_context_ids),
                    attn_weights
                )

                # Frame (Side-by-Side)
                frames_html += f"""
                <div class="heatmap-frame {active_class}">
                    <div style="margin-bottom:10px; border-bottom:1px solid #eee; padding-bottom:5px;">
                        <strong>Step {step_k + 1}</strong>: Generating <span style="background:#e8f0fe; color:#1967d2; padding:2px 6px; border-radius:4px;">"{display_token}"</span>
                    </div>

                    <div class="comparison-row">
                        <div class="comparison-col">
                            <div class="sub-label">Training Sample Attention (Dynamic)</div>
                            <div class="code-box full-height">{train_heatmap_html}</div>
                        </div>

                        <div class="comparison-col">
                            <div class="sub-label">Query Target Attention (Static Reference)</div>
                            <div class="code-box full-height">{static_query_attn_html}</div>
                        </div>
                    </div>
                </div>
                """

                current_context_ids.append(gen_token_id)

            vis_html = f"""
            <div id="{unique_sample_id}" style="margin-top: 15px; border-top:1px dashed #ccc; padding-top:10px;">
                <div class="sub-title">Interactive Inference Analysis (Side-by-Side)</div>
                <div class="generation-stream">{buttons_html}</div>
                <div class="attention-viewer">{frames_html}</div>
            </div>
            """
            inference_vis_htmls.append(vis_html)

            # Restore params for next sample iteration
            EmpiricalIF.restore_params(model, snapshot_thetaQi, param_filter_fn)
            del single_batch
            torch.cuda.empty_cache()

        # ================= 3. HTML Assembly =================

        # Prepare Header Info
        query_input_ids = query_batch["input_ids"][0].cpu()
        query_output_html = get_query_html_with_highlight(
            tokenizer,
            query_input_ids,
            target_sequence_idx
        )

        html_template = f"""
            <html>
            <head>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1600px; margin: 20px auto; padding: 20px; background-color: #f8f9fa; }}
                .card {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 30px; border-left: 5px solid #3498db; }}
                .card.harmful {{ border-left-color: #e74c3c; }}
                .card.helpful {{ border-left-color: #e74c3c; }}

                h1 {{ color: #2c3e50; text-align: center; }}
                .metrics {{ display: flex; justify-content: center; gap: 30px; font-weight: bold; background: #fff; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}

                .code-box {{ font-family: 'Consolas', monospace; white-space: pre-wrap; word-break: break-all; background: #ffffff; padding: 15px; border: 1px solid #dfe6e9; border-radius: 5px; }}
                .code-box.full-height {{ height: 100%; box-sizing: border-box; }}

                .label {{ font-weight: bold; margin-top: 15px; display: block; color: #2c3e50; font-size: 0.85em; text-transform: uppercase; }}
                .sub-title {{ font-size: 0.9em; color: #7f8c8d; margin-bottom: 5px; font-weight: bold; }}

                .generation-stream {{ display: flex; flex-wrap: wrap; gap: 5px; padding: 10px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; margin-bottom: 10px; }}
                .gen-token-btn {{ background: white; border: 1px solid #ced4da; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-family: monospace; font-size: 0.85em; transition: all 0.2s; }}
                .gen-token-btn:hover {{ border-color: #3498db; background: #ebf5fb; }}
                .gen-token-btn.active {{ background: #3498db; color: white; border-color: #2980b9; }}

                .attention-viewer {{ border: 1px solid #dee2e6; border-radius: 6px; background: white; }}
                .heatmap-frame {{ display: none; padding: 15px; }}
                .heatmap-frame.active {{ display: block; animation: fadeIn 0.3s; }}

                .comparison-row {{ display: flex; gap: 20px; align-items: stretch; }}
                .comparison-col {{ flex: 1; display: flex; flex-direction: column; }}
                .sub-label {{ text-align: center; font-weight: bold; color: #555; margin-bottom: 8px; padding-bottom: 5px; border-bottom: 2px solid #eee; }}

                @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
            </style>
            </head>
            <body>
                <h1>Influence Analysis: Query {query_idx} (Token {i + 1}) Rank {rank_pos} Percentile {percentile}</h1>
                <div class="metrics">
                    <span>Token Index: {i}</span>
                    <span>Query Loss: {l_query_token_loss_thetaQi[0].mean().item():.6e}</span>
                </div>

                <div class="card">
                    <h2>Target Query Context ($\\\\theta_{{Q,i}}$ State)</h2>
                    <span class="label">Target Token in Context:</span>
                    <div class="code-box">{query_output_html}</div>
                </div>

                <h2>Training Sample Analysis</h2>
        """

        for j, (idx, current_score) in enumerate(zip(all_indices, all_scores)):
            if j > 0 and j < len(top_5_harmful_scores):
                card_class = "harmful"
                type_label = f"Top Harmful #{j}"
            elif j >= len(top_5_harmful_scores):
                card_class = "helpful"
                type_label = f"Top Helpful #{j-len(top_5_harmful_scores)}"
            else:
                card_class = ""
                type_label = "Self (Target)"

            train_input_ids = train_raw_samples[j]["input_ids"].cpu()
            train_diffs = delta_train_token_loss_theta0_to_thetaQi[j].cpu()
            train_loss_html = get_colored_html_from_ids(tokenizer, train_input_ids, train_diffs, enable_coloring=True)

            train_generated_html_theta0 = get_colored_html_from_ids(tokenizer, inference_train_tokens_theta0[j], None, enable_coloring=False)

            train_generated_html = get_colored_html_from_ids(tokenizer, inference_train_tokens_thetaQi[j], None, enable_coloring=False)

            html_template += f"""
                <div class="card {card_class}">
                    <div style="margin-top: 20px; border-top: 1px dashed #ccc; padding-top: 15px;">
                        <span class="label">Training: Ground-truth Output</span>
                        <div class="code-box">{train_loss_html}</div>
                    </div>
                     <div style="margin-top: 20px; border-top: 1px dashed #ccc; padding-top: 15px;">
                        <span class="label">Training: Generated Response at $theta_0$</span>
                        <div class="code-box">{train_generated_html_theta0}</div>
                    </div>
                    <div style="margin-top: 20px; border-top: 1px dashed #ccc; padding-top: 15px;">
                        <span class="label">Training: Generated Response at $theta_Q$</span>
                        <div class="code-box">{train_generated_html}</div>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <h2>{type_label} <span style="font-size:0.7em; color:#999;">(Index {idx})</span></h2>
                        <span style="font-weight:bold;">Influence: {current_score:.4e}</span>
                    </div>

                    {inference_vis_htmls[j]}

                </div>
            """

        html_template += """
            <script>
                function showStep(sampleId, stepIndex) {
                    const container = document.getElementById(sampleId);
                    const buttons = container.querySelectorAll('.gen-token-btn');
                    const frames = container.querySelectorAll('.heatmap-frame');
                    buttons.forEach(b => b.classList.remove('active'));
                    frames.forEach(f => f.classList.remove('active'));
                    buttons[stepIndex].classList.add('active');
                    frames[stepIndex].classList.add('active');
                }
            </script>
            </body></html>
        """

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        final_path = f"{output_dir}/query_{query_idx}_token_{i}.html"
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(html_template)

    EmpiricalIF.restore_params(model, snapshot_theta0, param_filter_fn)
    gc.collect()
    torch.cuda.empty_cache()

def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # 路径配置
    JSONL_PATH = "/mnt/nvme0n1/ruofan/git_space/Empirical-Influence-Function/sft.jsonl"  # 请确保文件存在
    MODEL_ID   = "/mnt/nvme0n1/ruofan/git_space/Empirical-Influence-Function/src/sft/scripts/checkpoint-full-long"
    RESULTS_JSON_PATH = "/mnt/nvme0n1/ruofan/git_space/Empirical-Influence-Function/experiment_results.jsonl"
    VIS_DIR = "influence_reports_v3"
    # RESULTS_JSON_PATH = "/mnt/nvme0n1/ruofan/git_space/Empirical-Influence-Function/experiment_results_last_decoder.jsonl"

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
    # target_layer_keywords = ["model.layers.27.mlp"]

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
    # if accelerator.is_main_process:
    #     if os.path.exists(RESULTS_JSON_PATH):
    #         shutil.move(RESULTS_JSON_PATH, f"{RESULTS_JSON_PATH}.bak")
    #     logger.info(f"Results will be streamed to {RESULTS_JSON_PATH}")

    for i in tqdm(range(len(test_texts)), desc="Running Experiments"):

        # if i not in [15, 10, 25, 38, 17, 1, ]:
        # if i not in [7, 14, 36, 5, 2, 21]: #
        # if i not in [23, 34, 3]: #
        # if i not in [32, ]: # good examples , 43, 0
        #     continue
        # if i not in [17, 3, 34, 13, 38]:
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

        # scores = [random.random() for _ in range(len(train_texts))]
        # indices = [i for i in range(len(train_texts))]

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
            # top_1_harmful = sorted_results[-1:][::-1]
            # top_1_indices = [idx for idx, score in top_1_harmful]
            # top_1_scores = [score for idx, score in top_1_harmful]
            #
            # top_1_helpful = sorted_results[-1:]
            # top_1_helpful_indices = [idx for idx, score in top_1_helpful]
            # top_1_helpful_scores = [score for idx, score in top_1_helpful]
            #
            # # 保存 HTML 报告
            # save_query_report_html_attn(
            #     query_idx=i,
            #     query_batch=query_batch,  # 传入 Query Batch
            #     train_dataset=train_ds,  # 传入训练集 Dataset
            #     rank_pos=rank_pos,
            #     percentile=percentile,
            #     score=self_score,
            #     tokenizer=tokenizer,
            #     top_5_harmful_indices=top_1_indices,  # Top 5 索引
            #     top_5_harmful_scores=top_1_scores,  # Top 5 分数
            #     top_5_helpful_indices=top_1_helpful_indices,  # Top 5 索引
            #     top_5_helpful_scores=top_1_helpful_scores,  # Top 5 分数
            #     output_dir=VIS_DIR,
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



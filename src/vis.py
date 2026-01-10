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

import torch
from typing import Optional
from transformers import AutoTokenizer


def get_attention_html(tokenizer, input_ids, attention_weights):
    """
    渲染 Attention Heatmap (橙色高亮) 的 HTML。
    input_ids: Tensor or List
    attention_weights: Tensor or List (0.0~1.0)
    """
    if attention_weights is None or len(attention_weights) == 0:
        return "No Attention Data"

    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
    if isinstance(attention_weights, torch.Tensor):
        attn_scores = attention_weights.tolist()
    else:
        attn_scores = attention_weights

    # Normalize
    max_score = max(attn_scores) if attn_scores else 1.0
    if max_score <= 1e-9: max_score = 1.0

    html_content = ""
    for t, score in zip(tokens, attn_scores):
        intensity = score / max_score

        display_t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        display_t = display_t.replace('Ċ', '\n').replace('Ġ', ' ').replace('Ä‰', '\t')
        display_t = display_t.replace('\n', '<br>').replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;').replace(' ', '&nbsp;')

        # Gold/Orange color scheme
        alpha = 0.1 + (0.9 * intensity)
        style = f'background-color: rgba(255, 180, 0, {alpha:.2f});'

        html_content += f'<span style="{style}" title="Attn: {score:.4f}">{display_t}</span>'

    return html_content

def get_colored_html_from_ids(
        tokenizer: AutoTokenizer,
        input_ids: torch.Tensor,
        diffs_tensor: Optional[torch.Tensor] = None,
        enable_coloring: bool = False
) -> str:
    """
    渲染 Loss Diff (红色/蓝色) 的 HTML。
    """
    if not enable_coloring or diffs_tensor is None:
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        try:
            response_part = full_text.split("assistant\n")[-1]
            return response_part.replace('\n', '<br>').replace(' ', '&nbsp;')
        except:
            return full_text.replace('\n', '<br>')

    html_content = ""
    diffs_list = diffs_tensor.tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
    assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")

    # 找到 Output 开始的位置
    output_start_idx = 0
    input_ids_cpu = input_ids.cpu()
    for i in range(len(input_ids_cpu) - 1):
        if input_ids_cpu[i] == assistant_token_id:
            output_start_idx = i + 1
            if output_start_idx < len(input_ids_cpu):
                decoded_next = tokenizer.decode(input_ids_cpu[output_start_idx].item())
                if decoded_next in ['\n', 'Ċ', 'Ġ\n']:
                    output_start_idx += 1
            break

    # 计算最大 Diff 用于归一化
    visible_diffs = []
    for i in range(output_start_idx, len(tokens)):
        diff_idx = i - 1
        if 0 <= diff_idx < len(diffs_list):
            visible_diffs.append(abs(diffs_list[diff_idx]))
    max_abs_diff = max(visible_diffs) if visible_diffs and max(visible_diffs) > 0 else 1.0

    for i in range(output_start_idx, len(tokens)):
        t = tokens[i]
        if t in [tokenizer.pad_token, tokenizer.eos_token, '<|im_end|>']:
            break

        # 转义 HTML 字符
        display_t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        display_t = display_t.replace('Ċ', '\n').replace('Ġ', ' ').replace('Ä‰', '\t')
        display_t = display_t.replace('\n', '<br>').replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;').replace(' ', '&nbsp;')

        current_diff = 0.0
        diff_index = i - 1
        if 0 <= diff_index < len(diffs_list):
            current_diff = diffs_list[diff_index]

        # 颜色逻辑
        normalized_score = min(abs(current_diff) / max_abs_diff, 1.0)
        max_lightness = 96; min_lightness = 70
        lightness = max_lightness - (normalized_score * (max_lightness - min_lightness))

        style = ""
        if abs(current_diff) > 1e-6:
            if current_diff > 0:
                style = f'background-color: hsl(0, 90%, {lightness:.1f}%); color: #4a0f0f;' # Red
            else:
                style = f'background-color: hsl(210, 90%, {lightness:.1f}%); color: #0d2b4a;' # Blue
            html_content += f'<span style="{style}">{display_t}</span>'
        else:
            html_content += display_t

    return html_content

def get_query_html_with_highlight(tokenizer, input_ids, target_idx):
    """
    渲染 Query HTML，除了 target_idx 位置的 Token 被高亮圈出外，其余为普通文本。
    """
    html_content = ""
    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
    assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")

    # 找到 Output 开始的位置
    output_start_idx = 0
    input_ids_cpu = input_ids.cpu()
    for i in range(len(input_ids_cpu) - 1):
        if input_ids_cpu[i] == assistant_token_id:
            output_start_idx = i + 1
            if output_start_idx < len(input_ids_cpu):
                decoded_next = tokenizer.decode(input_ids_cpu[output_start_idx].item())
                if decoded_next in ['\n', 'Ċ', 'Ġ\n']:
                    output_start_idx += 1
            break

    for i in range(output_start_idx, len(tokens)):
        t = tokens[i]
        # --- 字符清洗 (保持和你之前的一致) ---
        display_t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        display_t = display_t.replace('Ċ', '\n').replace('Ġ', ' ').replace('Ä‰', '\t')
        display_t = display_t.replace('\n', '<br>').replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;').replace(' ', '&nbsp;')

        # --- 核心逻辑: 判断是否是当前的目标 Token ---
        if i == target_idx:
            # 样式：红色边框 + 浅红背景 + 加粗，模拟“圈出来”的效果
            # position: relative 是为了如果需要加 Tooltip 可以加
            style = (
                "border: 2px solid #e74c3c; "       # 红色实线边框
                "background-color: #fadbd8; "       # 浅粉色背景
                "color: #c0392b; "                  # 深红色文字
                "font-weight: bold; "               # 加粗
                "border-radius: 4px; "              # 圆角
                "padding: 0 2px; "                  # 稍微撑开一点
                "box-shadow: 0 0 5px rgba(231, 76, 60, 0.4);" #以此强调
            )
            html_content += f'<span style="{style}" title="Target Token (Gradient calculated here)">{display_t}</span>'
        else:
            # 普通样式 (灰色，弱化背景)
            html_content += f'<span style="color: #555;">{display_t}</span>'

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
    all_indices = [query_idx] + top_5_harmful_indices + top_5_helpful_indices
    # 对应的分数列表，用于展示
    all_scores = [score] + top_5_harmful_scores + top_5_helpful_scores
    train_raw_samples = [train_dataset[i] for i in all_indices]
    # ids_list = tokenizer.convert_tokens_to_ids([["\\t", "\\n"]])  # 这里填入你想忽略的 token 字符串
    ids_list = []
    ignored_token_ids = torch.tensor(ids_list, device=model.device)

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
            card_class = "harmful"
            type_label = f"Top Harmful #{j}"
        else:
            card_class = "helpful"
            type_label = f"Top Helpful #{j - 5}"

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


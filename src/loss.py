import torch
import torch.nn as nn
import random
import torch
import logging
from typing import List, Dict, Callable, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset
from functools import partial
import json
from tqdm import tqdm
import re
import os
import numpy as np
import re
import shutil
import re
import torch
import random
from typing import Optional, Dict, List

def compute_loss_per_sample(model, batch, device, ignored_token_ids):
    """
    核心 Loss 计算 (优化版)：
    直接修改 labels 为 -100 来屏蔽 loss。
    """
    ignored_token_ids = ignored_token_ids.to(device)
    inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits.float()

    # 1. 进行错位
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous().clone()  # clone 一份，避免修改原始数据

    # 2. 找出 shift_labels 中属于需要忽略的 token 的位置
    mask_to_ignore = torch.isin(shift_labels, ignored_token_ids)

    # 3. 将这些位置的 label 直接修改为 -100
    shift_labels[mask_to_ignore] = -100
    # ============================================

    # 4. 计算 Loss
    # reduction='none' 确保返回的是每个 token 的 loss
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    # 计算出来的 token_losses 在被忽略的位置上已经是 0 了
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
        shift_labels.size())

    # 5. 计算有效 token 数量 (即 label 不为 -100 的位置)
    valid_mask = shift_labels.ne(-100).float()
    num_valid  = valid_mask.sum(dim=1)

    # 6. 计算 sum 和 mean loss
    sum_loss = token_losses.sum(dim=1)

    # 避免除以 0
    mean_loss = sum_loss / (num_valid + 1e-9)

    # 返回的 token_losses 已经是经过屏蔽的了（被屏蔽处为0）
    # token_losses 的形状是 [batch_size, seq_len - 1]
    return mean_loss, token_losses


def compute_independent_sliding_loss_scalar(
        model,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        ignored_token_ids: torch.Tensor,
        tokenizer,
        num_samples_in: int = 3,
        num_samples_out: int = 3,
        min_ratio_in: float = 0.4,
        min_ratio_out: float = 0.4
):
    # 1. 前置准备 (假设 query batch size 为 1)
    if batch["input_ids"].size(0) != 1:
        return compute_loss_scalar_for_grad(model, batch, device, ignored_token_ids)

    original_input_ids = batch["input_ids"][0].to(device)
    original_labels = batch["labels"][0].to(device)

    # 获取关键结构 Token 的 ID (针对 Qwen ChatML 调整)
    # 注意：不同分词器的特殊 token 表示可能不同，这里需要根据实际情况调整
    try:
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        # Qwen 的 system/user/assistant 通常是普通 token
        system_id = tokenizer.convert_tokens_to_ids("system")
        user_id = tokenizer.convert_tokens_to_ids("user")
        assistant_id = tokenizer.convert_tokens_to_ids("assistant")
        # 换行符也很重要
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    except:
        # Fallback: 如果获取失败，打印警告并退化为不采样
        print("Warning: Failed to get special structure token IDs. Skipping sliding window.")
        return compute_loss_scalar_for_grad(model, batch, device, ignored_token_ids)

    structure_tokens = {im_start_id, im_end_id, system_id, user_id, assistant_id}
    structure_tokens.update(newline_ids)
    # 移除 None 或未定义的 ID
    structure_tokens = {tid for tid in structure_tokens if tid is not None}

    # 2. 分离输入和输出
    output_mask = (original_labels != -100)
    output_indices = torch.where(output_mask)[0]
    if len(output_indices) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    output_start_idx = output_indices[0].item()

    full_input_part_ids = original_input_ids[:output_start_idx]
    full_output_part_ids = original_input_ids[output_start_idx:]
    full_output_part_labels = original_labels[output_start_idx:]

    # --- 定义结构化采样辅助函数 (改进版：更小窗口，更剧烈移动) ---
    def sample_structured_subsequence(
            sequence,
            labels=None,
            min_ratio=0.2,  # 降低默认最小比例，允许更短的窗口
            min_absolute_len=5,  # 设置一个绝对最小长度，防止窗口过短
            num_samples=3
    ):
        seq_len = len(sequence)
        # 找出非结构化内容 Token 的索引
        structure_tensor = torch.tensor(list(structure_tokens), device=device)
        is_structure_mask = torch.isin(sequence, structure_tensor)
        content_indices = torch.where(~is_structure_mask)[0]

        # 如果总内容长度本身就太短，则不进行采样，直接返回原始序列
        if len(content_indices) <= min_absolute_len:
            return [sequence], ([labels] if labels is not None else None)

        # 确定内容的起始和结束索引
        content_start = content_indices[0].item()
        content_end = content_indices[-1].item()  # 包含
        content_len = content_end - content_start + 1

        # 分离前缀、内容、后缀
        prefix = sequence[:content_start]
        content = sequence[content_start: content_end + 1]
        suffix = sequence[content_end + 1:]

        if labels is not None:
            prefix_lbl = labels[:content_start]
            content_lbl = labels[content_start: content_end + 1]
            suffix_lbl = labels[content_end + 1:]

        sampled_seqs = []
        sampled_lbls = [] if labels is not None else None

        # 计算实际的最小窗口大小
        # 取 "按比例计算出的最小长度" 和 "绝对最小长度" 中的较大值
        min_window = max(min_absolute_len, int(content_len * min_ratio))
        # 确保最小窗口不超过内容总长度
        min_window = min(min_window, content_len)

        # 为了增加移动的剧烈性，我们可以尝试生成分布更广的起始点
        # 一种方法是将可能的起始范围划分为 num_samples 个区间，在每个区间内随机取点
        max_start_offset = content_len - min_window

        # 如果可移动范围太小，退化为简单随机
        if max_start_offset <= num_samples:
            start_offsets = [random.randint(0, max_start_offset) for _ in range(num_samples)]
        else:
            # 分层采样，确保起始点分布更均匀/剧烈
            interval_size = max_start_offset // num_samples
            start_offsets = []
            for k in range(num_samples):
                # 定义区间的边界
                interval_start = k * interval_size
                # 最后一个区间延伸到末尾
                interval_end = (k + 1) * interval_size if k < num_samples - 1 else max_start_offset + 1

                # 在当前区间内随机选择一个起始偏移量
                offset = random.randint(interval_start, interval_end - 1)
                start_offsets.append(offset)

            # 为了增加随机性，可以再 shuffle 一下
            random.shuffle(start_offsets)

        for start_offset in start_offsets:
            # 在确定了起始点后，窗口大小也可以是随机的
            # 最大可能的窗口大小取决于起始点位置
            max_possible_window = content_len - start_offset
            # 窗口大小在 [min_window, max_possible_window] 之间随机
            window_size = random.randint(min_window, max_possible_window)

            sub_content = content[start_offset: start_offset + window_size]

            # 重新拼装：前缀 + 子内容 + 后缀
            new_seq = torch.cat([prefix, sub_content, suffix])
            sampled_seqs.append(new_seq)

            if labels is not None:
                sub_content_lbl = content_lbl[start_offset: start_offset + window_size]
                new_lbl = torch.cat([prefix_lbl, sub_content_lbl, suffix_lbl])
                sampled_lbls.append(new_lbl)

        return sampled_seqs, sampled_lbls

    # 3 & 4. 分别进行结构化采样
    # Input 只需要采样 IDs
    input_samples_ids, _ = sample_structured_subsequence(
        full_input_part_ids, min_ratio=min_ratio_in, num_samples=num_samples_in
    )
    # Output 需要同时采样 IDs 和 Labels
    output_samples_ids, output_samples_labels = sample_structured_subsequence(
        full_output_part_ids, labels=full_output_part_labels, min_ratio=min_ratio_out, num_samples=num_samples_out
    )

    combined_input_ids_list = []
    combined_labels_list = []

    # 5. 排列组合 (Kin * Kout 种组合)
    for i in range(len(input_samples_ids)):
        for j in range(len(output_samples_ids)):
            in_seq = input_samples_ids[i]
            out_seq_ids = output_samples_ids[j]
            out_seq_labels = output_samples_labels[j]

            # 拼接 Input IDs
            combined_ids = torch.cat([in_seq, out_seq_ids])
            combined_input_ids_list.append(combined_ids)

            # 拼接 Labels (输入部分设为 -100)
            new_in_labels = torch.full((len(in_seq),), -100, dtype=original_labels.dtype, device=device)
            combined_lbls = torch.cat([new_in_labels, out_seq_labels])
            combined_labels_list.append(combined_lbls)

    # 6. 构建大 Batch 并 Padding (保持不变)
    padded_inputs = tokenizer.pad(
        {"input_ids": combined_input_ids_list}, padding=True, return_tensors="pt"
    ).to(device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    temp_labels_list = []
    for lbl in combined_labels_list:
        lbl_copy = lbl.clone()
        lbl_copy[lbl_copy == -100] = pad_id
        temp_labels_list.append(lbl_copy)

    padded_labels_info = tokenizer.pad(
        {"input_ids": temp_labels_list}, padding=True, return_tensors="pt"
    ).to(device)

    final_padded_labels = padded_labels_info["input_ids"]
    attention_mask = padded_inputs["attention_mask"]
    final_padded_labels[attention_mask == 0] = -100
    if pad_id != -100:
        final_padded_labels[final_padded_labels == pad_id] = -100

    combined_batch = {
        "input_ids": padded_inputs["input_ids"],
        "attention_mask": padded_inputs["attention_mask"],
        "labels": final_padded_labels
    }

    # 7. 计算平均 Loss
    # 使用你现有的 compute_loss_per_sample 函数
    per_sample_mean_losses, _ = compute_loss_per_sample(model, combined_batch, device, ignored_token_ids)

    # 返回所有组合的平均 Loss (标量)
    return per_sample_mean_losses.mean()

def compute_loss_scalar_for_grad(model, batch, device, ignored_token_ids):
    """使用 Mean 保证梯度稳定"""
    mean_loss, _ = compute_loss_per_sample(model, batch, device, ignored_token_ids)
    return mean_loss

# def compute_gradients(model, batch, param_filter_fn, device, ignored_token_ids):
#     model.eval()
#     model.zero_grad(set_to_none=True)
#     with torch.set_grad_enabled(True):
#         loss = compute_loss_scalar_for_grad(model, batch, device, ignored_token_ids)
#         params = [p for n, p in model.named_parameters() if p.requires_grad and (param_filter_fn is None or param_filter_fn(n, p))]
#         grads = torch.autograd.grad(loss, params)
#     return list(grads)

def compute_gradients(
        model,
        batch,
        param_filter_fn,
        device,
        ignored_token_ids,
        tokenizer,  # 需要传入 tokenizer
        use_independent_sliding: bool = True,  # 新增开关：是否使用你的独立滑动策略
        num_samples_in: int = 4,
        num_samples_out: int = 4,
        min_ratio_in: float = 0.4,
        min_ratio_out: float = 0.4
):
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(True):
        # if use_independent_sliding:
        #     # 使用你想法的实现
        #     loss = compute_independent_sliding_loss_scalar(
        #         model, batch, device, ignored_token_ids, tokenizer,
        #         num_samples_in=num_samples_in, num_samples_out=num_samples_out,
        #         min_ratio_in=min_ratio_in, min_ratio_out=min_ratio_out
        #     )
        # else:
        #     # 使用原始 Loss
        loss   = compute_loss_scalar_for_grad(model, batch, device, ignored_token_ids)

        params = [p for n, p in model.named_parameters() if
                  p.requires_grad and (param_filter_fn is None or param_filter_fn(n, p))]

        # 确保 loss 是标量
        if loss.numel() > 1:
            loss = loss.mean()

        grads = torch.autograd.grad(loss, params)
    return list(grads)
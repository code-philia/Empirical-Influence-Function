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
    # 确保 ignored_token_ids 是 Tensor 且在正确的设备上
    if ignored_token_ids is not None and not isinstance(ignored_token_ids, torch.Tensor):
        ignored_token_ids = torch.tensor(ignored_token_ids, device=device)
    elif ignored_token_ids is not None:
        ignored_token_ids = ignored_token_ids.to(device)

    inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits.float()

    # 1. 进行错位
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous().clone()  # clone 一份，避免修改原始数据

    if ignored_token_ids is not None and len(ignored_token_ids) > 0:
        mask_to_ignore = torch.isin(shift_labels, ignored_token_ids)
        shift_labels[mask_to_ignore] = -100

    # 4. 计算 Loss
    # reduction='none' 确保返回的是每个 token 的 loss
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    # 计算出来的 token_losses 在被忽略的位置上已经是 0 了
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())

    # 5. 计算有效 token 数量 (即 label 不为 -100 的位置)
    valid_mask = shift_labels.ne(-100).float()
    num_valid  = valid_mask.sum(dim=1)

    # 6. 计算 sum 和 mean loss
    sum_loss = token_losses.sum(dim=1)

    # 避免除以 0
    mean_loss = sum_loss / (num_valid + 1e-9)

    return mean_loss, token_losses

@torch.no_grad()
def compute_loss_in_minibatches(model, collator, samples_list, ignored_token_ids, batch_size=2):
    all_samples_loss_list = []  # 存储每个样本的 1D Tensor
    for i in range(0, len(samples_list), batch_size):
        batch_samples = samples_list[i: i + batch_size]
        batch = collator(batch_samples)
        batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)} # 移到 GPU

        with torch.no_grad():
            _, token_loss = compute_loss_per_sample(model, batch, model.device, ignored_token_ids)

        # 立即上 CPU
        token_loss_cpu = token_loss.detach().cpu()
        all_samples_loss_list.extend(token_loss_cpu.unbind(0))
        del batch

    return all_samples_loss_list

def compute_gradients(
        model,
        batch,
        param_filter_fn,
        device,
        ignored_token_ids
):
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(True):
        #     # 使用原始 Loss
        mean_loss,_   = compute_loss_per_sample(model, batch, device, ignored_token_ids)
        loss = mean_loss.mean()

        # 核心优化：只提取需要更新的参数（如 lm_head）
        params = [p for n, p in model.named_parameters() if
                  p.requires_grad and (param_filter_fn is None or param_filter_fn(n, p))]

        # 确保 loss 是标量
        if loss.numel() > 1:
            loss = loss.mean()

        grads = torch.autograd.grad(loss, params, create_graph=False)
    return list(grads)


def compute_token_specific_update(
        model,
        batch,
        param_filter_fn,
        device,
        ignored_token_ids,
        target_sequence_idx: int,
        lr: float
):
    """
    针对 query_batch 中特定序列索引的 token 计算梯度，并应用一次更新。
    """
    model.zero_grad(set_to_none=True)

    # 1. 前向传播
    inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits.float()

    # 2. 错位和屏蔽 (与 compute_loss_per_sample 逻辑相似)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous().clone()

    if ignored_token_ids is not None and len(ignored_token_ids) > 0:
        mask_to_ignore = torch.isin(shift_labels, ignored_token_ids.cpu())  # 确保在 CPU 上比较
        shift_labels[mask_to_ignore] = -100

    # 3. 提取单 Token Loss
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    token_losses_flat = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses_flat.view(shift_labels.size())

    # 4. 选取目标 Token 的损失并进行 Backward
    # 确保索引在范围内
    loss_index_in_shifted = target_sequence_idx - 1

    if loss_index_in_shifted >= shift_logits.shape[1] or loss_index_in_shifted < 0:
        raise IndexError(f"Warning: Token index {loss_index_in_shifted} out of bounds.")

    # 仅对该 Token 的损失进行反向传播
    single_token_loss = token_losses[0, loss_index_in_shifted]  # 假设 batch_size=1

    # 仅在损失有效时才反向传播（避免对 -100 的位置求导）
    if single_token_loss.item() != 0 or shift_labels[0, target_sequence_idx].item() != -100:
        single_token_loss.backward()

        # 5. 收集梯度并应用更新
        params = [p for n, p in model.named_parameters() if
                  p.requires_grad and (param_filter_fn is None or param_filter_fn(n, p))]
        grads = [p.grad for p in params]  # 直接使用 .grad

        return grads

    raise IndexError(f"Warning: Token index has label mask as -100.")


@torch.inference_mode()
def get_first_response_token(
        batch,
        ignored_token_ids,
):
    # 找到第一个未被忽略（即需要计算损失）的 Token 索引
    labels_shifted = batch["labels"][0, 1:].cpu()

    effective_ignored_ids = ignored_token_ids.cpu() if ignored_token_ids is not None and ignored_token_ids.numel() > 0 else torch.tensor([])
    is_valid = labels_shifted.ne(-100)
    if effective_ignored_ids.numel() > 0:
        is_valid = is_valid & ~torch.isin(labels_shifted, effective_ignored_ids)

    # 找到第一个为 True 的索引
    valid_indices = torch.where(is_valid)[0]
    if valid_indices.numel() == 0:
        print(f"Could not find any valid response token. Skipping report generation.")
        return None, None

    # query_response_start_idx_in_shifted_labels 是响应在 shift_labels 中的起始索引
    response_start_idx_in_shifted_labels = valid_indices[0].item()

    # 序列总长度 (shift_labels 的长度)
    total_shifted_len = labels_shifted.shape[0]

    # 响应的有效长度
    query_response_len = total_shifted_len - response_start_idx_in_shifted_labels
    return response_start_idx_in_shifted_labels, query_response_len

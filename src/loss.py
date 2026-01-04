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
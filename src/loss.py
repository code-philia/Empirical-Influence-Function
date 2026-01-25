import torch
from torch import Tensor, nn
from collections.abc import Callable, Iterable

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
    outputs = model(**inputs, return_dict=True, output_attentions=False, use_cache=False)
    logits = outputs.logits.float()

    del outputs

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


def compute_answer_only_union_topk_loss(
    model: torch.nn.Module,
    batch: dict[str, Tensor],
    device: torch.device,
    target_idx: Tensor,
    top_k: int = 10,
    ignored_token_ids: Iterable[int] | Tensor | None = None,
    *,
    enable_grad: bool = False,
    renormalize: bool = True,
) -> tuple[Tensor, Tensor]:
    '''
    Deprecated. Compute loss on a union of top-k correlated tokens of answer tokens.
    '''
    if ignored_token_ids is not None and not isinstance(ignored_token_ids, torch.Tensor):
        ignored_token_ids = torch.tensor(ignored_token_ids, device=device)
    elif ignored_token_ids is not None:
        ignored_token_ids = ignored_token_ids.to(device)

    inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
    labels = inputs["labels"].clone()
    start = int(target_idx[0].item())
    labels[..., :start] = -100

    with torch.set_grad_enabled(enable_grad):
        outputs = model(
            **inputs,
            return_dict=True,
            save_last_attention=True,
            use_cache=False,
        )

    logits = outputs.logits
    attn = outputs.attentions[-1].detach()
    del outputs

    bsz, n_heads, q_len, k_len = attn.shape
    k = min(top_k, k_len)
    attn_avg = attn.mean(dim=1)  # [B, Q, K]

    q_from = max(start, 0)
    if q_from >= q_len:
        raise ValueError("target_idx is beyond sequence length.")

    topk_indices = torch.topk(attn_avg[:, q_from:, :], k=k, dim=-1).indices
    union_mask = torch.zeros((bsz, k_len), device=attn.device, dtype=torch.bool)
    union_mask.scatter_(1, topk_indices.reshape(bsz, -1), True)

    masked_attn = attn_avg * union_mask[:, None, :]
    if renormalize:
        masked_attn = masked_attn / (masked_attn.sum(dim=-1, keepdim=True) + 1e-9)

    token_weights = torch.zeros_like(attn_avg)
    token_weights[:, q_from:, :] = masked_attn[:, q_from:, :].detach()

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    if ignored_token_ids is not None and len(ignored_token_ids) > 0:
        shift_labels[torch.isin(shift_labels, ignored_token_ids)] = -100

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size())

    weights = token_weights[..., :-1].contiguous()
    mask = shift_labels.ne(-100)
    weights = weights * mask

    weighted_token_losses = token_losses * weights
    denom = weights.sum(dim=1).clamp_min(1e-9)
    mean_loss = weighted_token_losses.sum(dim=1) / denom
    return mean_loss, weighted_token_losses


def compute_answer_only_saliency_masked_loss(
    model: torch.nn.Module,
    batch: dict[str, Tensor],
    device: torch.device,
    target_idx: Tensor,
    top_k: int = 10,
    ignored_token_ids: Iterable[int] | Tensor | None = None,
    *,
    enable_grad: bool = False,
) -> tuple[Tensor, Tensor, list[list[dict[str, object]]]]:
    '''
    Compute loss on answer part (>= `target_idx`), with gradient,
    attention-masked by `top_k` most relative tokens.

    Returns a tuple:
    - `mean_loss`:                of shape `(batch,)`.
    - `weighted_token_losses`:    of shape `(batch, token)`.
    - `saliency_list`:            list[list[dict]], of shape `(batch, from_token, to_token)`.
    '''
    if ignored_token_ids is not None and not isinstance(ignored_token_ids, torch.Tensor):
        ignored_token_ids = torch.tensor(ignored_token_ids, device=device)
    elif ignored_token_ids is not None:
        ignored_token_ids = ignored_token_ids.to(device)

    inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
    labels = inputs["labels"].clone()
    start = int(target_idx[0].item())
    labels[..., :start] = -100

    with torch.set_grad_enabled(enable_grad):
        outputs = model(
            **inputs,
            return_dict=True,
            save_last_attention=True,
            use_cache=False,
        )

    logits = outputs.logits
    attn = outputs.attentions[-1].detach()
    del outputs

    bsz, n_heads, q_len, k_len = attn.shape
    token_weights = torch.zeros((bsz, q_len), device=device, dtype=attn.dtype)

    saliency_list = []
    for i in range(bsz):
        saliency_list.append([])

    if not isinstance(model.get_input_embeddings, Callable):
        raise ValueError("Expect model.get_input_embeddings to be torch.nn.Module")

    for t in range(max(start, 1), q_len):
        curr_input_ids = inputs["input_ids"][:, :t]
        target_vocab_id = inputs["input_ids"][:, t]

        embeddings = model.get_input_embeddings()(curr_input_ids).detach()
        embeddings.requires_grad_(True)

        with torch.enable_grad():
            step_outputs = model(inputs_embeds=embeddings)
            target_logits = step_outputs.logits[:, -1, :]

            picked = target_logits.gather(1, target_vocab_id[:, None]).sum()
            # gradients from target logits to input embeddings
            grads = torch.autograd.grad(picked, embeddings, retain_graph=False, create_graph=False)[0]

        saliency = (embeddings * grads).abs().sum(dim=-1)   # [B, T, E]
        k = min(top_k, saliency.size(-1))

        # get top-k saliency token indices
        topk_indices = torch.topk(saliency, k=k, dim=-1).indices
        


        # build masks
        mask = torch.zeros((bsz, k_len), device=device, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)

        # apply attention mask to token t-1
        masked_attn = attn[:, :, t - 1, :] * mask[:, None, :]
        token_weights[:, t - 1] = masked_attn.sum(dim=-1).mean(dim=1).detach()
        # token_weights[:, t - 1] = torch.ones_like(masked_attn.sum(dim=-1).mean(dim=1).detach())

        for i in range(bsz):
            saliency_list[i].append({
                "index": t,
                "saliency": saliency[i].tolist()
            })
        del embeddings, grads, step_outputs

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    if ignored_token_ids is not None and len(ignored_token_ids) > 0:
        shift_labels[torch.isin(shift_labels, ignored_token_ids)] = -100

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size())

    weights = token_weights[..., : token_losses.size(-1)]
    mask = shift_labels.ne(-100)
    weights = weights * mask

    weighted_token_losses = token_losses * weights
    denom = weights.sum(dim=1).clamp_min(1e-9)
    mean_loss = weighted_token_losses.sum(dim=1) / denom
    return mean_loss, weighted_token_losses, saliency_list


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


def compute_gradients_selected_attention(
    model,
    batch,
    param_filter_fn,
    device,
    ignored_token_ids,
    *,
    target_idx
):
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Disable torch.inference_mode() before calling this function.")

    model.eval()
    model.zero_grad(set_to_none=True)

    params = [p for n, p in model.named_parameters()
              if (param_filter_fn is None or param_filter_fn(n, p))]
    if not params:
        raise RuntimeError("No parameters selected by param_filter_fn.")

    orig_flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)

    with torch.enable_grad():
        mean_loss, _, saliency = compute_answer_only_saliency_masked_loss(
            model,
            batch,
            device,
            target_idx,
            ignored_token_ids=ignored_token_ids,
            enable_grad=True,
        )
        loss = mean_loss.mean()

        if not loss.requires_grad:
            raise RuntimeError("Loss is detached. Check outer contexts and model freezing.")

        grads = torch.autograd.grad(loss, params, create_graph=False, allow_unused=False)

    for p, flag in zip(params, orig_flags):
        p.requires_grad_(flag)

    return grads, saliency

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

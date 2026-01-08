import torch
from typing import Optional


def model_generation(
        model,
        tokenizer,
        batch,
        response_start_idx: Optional[int] = None,
        max_new_tokens: int = 20,
):
    model.eval()
    if response_start_idx is None:
        response_start_idx = batch["input_ids"].shape[1]

    # Run standard generation to get the tokens
    with torch.no_grad():
        model.set_attn_implementation('sdpa')  # Use fast attention for generation
        outputs = model.generate(
            input_ids=batch["input_ids"][:, :response_start_idx],
            attention_mask=batch["attention_mask"][:, :response_start_idx],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
        )

    # The full sequence (Prompt + New Tokens)
    full_sequence_ids = outputs  # (Batch, Total_Len)
    generated_sequences = full_sequence_ids.detach().cpu()

    # Extract only the new tokens for return
    newly_generated_tokens = generated_sequences[:, response_start_idx:]
    return newly_generated_tokens

## 方法一：Attention Weights (最直观、最快)这是最常用且计算成本最低的方法。
# 在 Transformer 中，预测 $t_i$ 的决策主要发生在序列的最后一个位置 ($t_{i-1}$)。
# 我们需要查看这一位置在计算时，分配给前面所有 token 的注意力权重。
# 原理：Decoder 模型在预测 $t_i$ 时，实际上是利用 $t_{i-1}$ 位置的 Hidden State 进行计算的。
# 该位置的 Query 向量 ($Q_{i-1}$) 会去查询之前所有位置 ($0$ 到 $i-1$) 的 Key 向量 ($K$)。
# $$\text{Contribution}(t_j) \approx \text{AttentionWeight}(Q_{i-1}, K_j)$$

@torch.no_grad()
def attention_attribution_static(
        model,
        batch,
        target_idx: int,
):
    model.eval()
    q_input_ids = batch["input_ids"].to(model.device)
    q_attention_mask = batch["attention_mask"].to(model.device)

    model.set_attn_implementation('eager')
    query_outputs = model(
        input_ids=q_input_ids,
        attention_mask=q_attention_mask,
        output_attentions=True,
        return_dict=True
    )
    model.set_attn_implementation('sdpa')

    # Extract Attention: Row = (target_idx - 1) -> Predicting Token
    last_layer_attn = query_outputs.attentions[-1] # (Layers, B, Num_Heads, Q, K)
    predicting_idx  = target_idx - 1 # we are predicting target_idx, so we need to take the (target_idx - 1) as query

    # 截取到 target_sequence_idx (包含 context，不包含 target token 本身)
    query_attn_weights = last_layer_attn[0, :, predicting_idx, :target_idx]
    query_attn_cpu = query_attn_weights.mean(0).cpu() # Average over all heads

    del query_outputs
    return query_attn_cpu


@torch.no_grad()
def attention_attribution_on_generation(
        model,
        tokenizer,
        batch,
        response_start_idx: Optional[int] = None,
        max_new_tokens: int = 20,
):
    model.eval()
    if response_start_idx is None:
        response_start_idx = batch["input_ids"].shape[1]

    model.set_attn_implementation('eager')
    outputs = model.generate(
        input_ids=batch["input_ids"][:, :response_start_idx],
        attention_mask=batch["attention_mask"][:, :response_start_idx],
        output_attentions=True,
        return_dict_in_generate=True,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
    )
    model.set_attn_implementation('sdpa')

    generated_sequences = outputs.sequences.detach().cpu()
    # attentions is a nested Tuple (length=generated_steps) -> Tuple (length=num_layers) -> Tensor
    # Step 0 Tensor Shape: (B, Num_Heads, Prompt_Len, Prompt_Len)
    # Step >0 Tensor Shape: (B, Num_Heads, 1, Total_Past_Len + 1)
    attentions = outputs.attentions

    newly_generated_tokens = generated_sequences[:, response_start_idx:].cpu()  # (B, generated_steps)

    # Extract Step-wise Attention
    # This will be a List of Tensors, where each tensor has shape (B, K_len).
    # Note: K_len grows by 1 at each step.
    newly_generated_tokens_attentions = []

    for step_idx, layer_attentions in enumerate(attentions):

        last_layer_attn = layer_attentions[-1].detach().cpu()
        # layer_attentions is a Tuple of length `num_layers`
        # Shape Step 0: (B, Num_Heads, Prompt_Len, Prompt_Len)
        # Shape Step >0: (B, Num_Heads, 1, Total_Len)

        del layer_attentions
        query_len = last_layer_attn.shape[2]

        if query_len > 1:  # Prefill (First Token Generation)
            # We take the LAST row of the query dimension (the token predicting the first new token)
            current_step_attn = last_layer_attn[:, :, -1, :] # Shape: (B, Num_Heads, Prompt_Len)
        else:  # Decoding (Subsequent Tokens)
            current_step_attn = last_layer_attn.squeeze(2) # Shape: (B, Num_Heads, Total_Len)

        avg_current_step_attn = current_step_attn.mean(1).cpu()  # Shape: (B, Total_Len), Total_Len = Prompt_Len + step_idx

        newly_generated_tokens_attentions.append(avg_current_step_attn)

    del outputs
    return newly_generated_tokens, newly_generated_tokens_attentions


## 方法二：Gradient-based Saliency
# Attention 权重高不代表贡献一定大（可能 Value 向量很小）。基于梯度的方法通过计算“输出 Logit 对输入 Embedding 的梯度”来衡量敏感度。
# 原理：计算目标 token $t_i$ 的 Logit (分数) 相对于输入 token $t_j$ 的 Embedding 的偏导数。
# $$\text{Score}(t_j) = \left| \frac{\partial \text{Logit}(t_i)}{\partial \text{Emb}(t_j)} \times \text{Emb}(t_j) \right|$$(通常使用 Input $\times$ Gradient 以消除量纲影响)


def gradient_saliency_static(
        model,
        batch,
        target_idx: int,
):
    full_input_ids = batch["input_ids"].to(model.device)

    target_vocab_id = full_input_ids[:, target_idx]
    input_ids = full_input_ids[:, :target_idx]
    model.eval()

    # 1. Get Input Embeddings
    embeddings = model.get_input_embeddings()(input_ids).detach()
    embeddings.requires_grad = True

    # 2. Forward Pass
    outputs = model(inputs_embeds=embeddings)

    # 3. Target Selection
    # outputs.logits shape: (Batch, Seq_Len, Vocab_Size)
    # We look at the last token's logits, and pick the specific target class.
    target_logit = outputs.logits[0, -1, target_vocab_id]

    # 4. Backward Pass
    model.zero_grad()
    target_logit.backward()
    grads = embeddings.grad # Shape: (Batch, Seq_Len, Hidden_Dim)

    # 6. Compute Input * Gradient
    input_x_grad = embeddings * grads

    # 7. Aggregate over Hidden Dimension
    token_scores = input_x_grad.sum(dim=-1).abs()  # Shape: (Batch, Seq_Len)

    # Remove Batch dim
    token_scores = token_scores[0]  # Shape: (Seq_Len,)

    return token_scores


def gradient_saliency_on_generation(
        model,
        tokenizer,
        batch,
        response_start_idx: Optional[int] = None,
        max_new_tokens: int = 20,
):
    """
    Computes Input * Gradient saliency for each token generated by the model.
    """
    model.eval()

    # Phase 1: Generation (Fast Inference)
    if response_start_idx is None:
        response_start_idx = batch["input_ids"].shape[1]

    # Run standard generation to get the tokens
    with torch.no_grad():
        model.set_attn_implementation('sdpa')  # Use fast attention for generation
        outputs = model.generate(
            input_ids=batch["input_ids"][:, :response_start_idx],
            attention_mask=batch["attention_mask"][:, :response_start_idx],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
        )

    # The full sequence (Prompt + New Tokens)
    full_sequence_ids = outputs  # (Batch, Total_Len)
    generated_sequences = full_sequence_ids.detach().cpu()

    # Extract only the new tokens for return
    newly_generated_tokens = generated_sequences[:, response_start_idx:]

    # =========================================================
    # Phase 2: Saliency Calculation Loop (Step-by-Step)
    # =========================================================
    saliency_results = []

    # We iterate through each NEW token we just generated.
    # Logic: To explain the token at `current_target_idx`, we look at inputs `[:current_target_idx]`
    num_new_tokens = newly_generated_tokens.shape[1]

    for i in range(num_new_tokens):
        # The index of the token we want to explain (the target)
        target_seq_idx = response_start_idx + i

        # Prepare Input for this step (Context = Prompt + Previous Generated Tokens)
        curr_input_ids = full_sequence_ids[:, :target_seq_idx]

        # The specific Token ID (Word) that was generated at this position
        target_vocab_id = full_sequence_ids[0, target_seq_idx]  # Assume Batch=1

        # --- Standard Gradient Saliency Logic ---

        # 1. Get Embeddings & Enable Gradient
        # We must create a new computation graph for this specific step
        embeddings = model.get_input_embeddings()(curr_input_ids).detach()
        embeddings.requires_grad = True

        # 2. Forward Pass
        # Note: We cannot use KV cache here easily because we need grads w.r.t input embeddings
        outputs = model(inputs_embeds=embeddings)

        # 3. Select Target Logit
        # We look at the LAST position (-1) because that is what predicted the 'target_vocab_id'
        target_logit = outputs.logits[0, -1, target_vocab_id]

        # 4. Backward Pass
        model.zero_grad()
        target_logit.backward()

        # 5. Compute Input * Gradient
        grads = embeddings.grad
        input_x_grad = embeddings * grads

        # 6. Aggregate
        # Sum over hidden dim and take Abs
        token_scores = input_x_grad.sum(dim=-1).abs()  # (Batch, Current_Seq_Len)

        # Store result on CPU to save GPU memory
        saliency_results.append(token_scores[0].cpu())

        # Cleanup to prevent OOM in loop
        del embeddings, grads, outputs, input_x_grad
        torch.cuda.empty_cache()

    return newly_generated_tokens, saliency_results
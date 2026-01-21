import torch
import random
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import training_datasets
import textwrap
import difflib
import json

class Color:
    GREEN = '\033[92m'   # 绿色
    BLUE = '\033[94m'    # 蓝色
    CYAN = '\033[96m'    # 青色 (可选，用于 Index)
    BOLD = '\033[1m'     # 加粗
    END = '\033[0m'      # 重置颜色


def restore_escaped_characters(text: str):
    text = text.encode().decode('unicode_escape')
    text = text.replace('\t', '  ')
    return text


def print_side_by_side(gt_text, model_text, width=50):
    """
    将两条文本并排打印。
    width: 每一列的字符宽度
    """
    # 1. 处理换行，确保每行不超过指定宽度
    gt_lines = []
    for line in gt_text.splitlines():
        gt_lines.extend(textwrap.wrap(line, width=width) if line.strip() else [""])

    model_lines = []
    for line in model_text.splitlines():
        model_lines.extend(textwrap.wrap(line, width=width) if line.strip() else [""])

    # 2. 补齐行数，使其对齐
    max_len = max(len(gt_lines), len(model_lines))
    gt_lines    += [""] * (max_len - len(gt_lines))
    model_lines += [""] * (max_len - len(model_lines))

    # 3. 打印标题
    header_gt    = f"{Color.BOLD}{Color.GREEN}{'GROUND TRUTH'.center(width)}{Color.END}"
    header_model = f"{Color.BOLD}{Color.BLUE}{'MODEL RESPONSE'.center(width)}{Color.END}"
    print(f"\n{header_gt} | {header_model}")
    print("-" * (width * 2 + 3))

    # 4. 逐行并排打印
    for gt, model in zip(gt_lines, model_lines):
        # 使用 ljust 确保左侧列对齐
        left_col  = f"{Color.GREEN}{gt.ljust(width)}{Color.END}"
        right_col = f"{Color.BLUE}{model.ljust(width)}{Color.END}"
        print(f"{left_col} | {right_col}")

def print_query_and_answer(query, gt_text, model_text="", width=70):
    print(f"{Color.BOLD}[[[ Query Example ]]]{Color.END}")
    print(restore_escaped_characters(query))

    gt_text = restore_escaped_characters(gt_text)
    model_text = restore_escaped_characters(model_text)

    # 1. 处理换行，确保每行不超过指定宽度
    gt_lines = []
    for line in gt_text.splitlines():
        gt_lines.extend(textwrap.wrap(line, width=width) if line.strip() else [""])

    model_lines = []
    for line in model_text.splitlines():
        model_lines.extend(textwrap.wrap(line, width=width) if line.strip() else [""])

    # 2. 补齐行数，使其对齐
    max_len = max(len(gt_lines), len(model_lines))
    gt_lines    += [""] * (max_len - len(gt_lines))
    model_lines += [""] * (max_len - len(model_lines))

    # 3. 打印标题
    header_gt    = f"{Color.BOLD}{Color.GREEN}{'GROUND TRUTH'.center(width)}{Color.END}"
    header_model = f"{Color.BOLD}{Color.BLUE}{'MODEL RESPONSE'.center(width)}{Color.END}"
    print(f"\n{header_gt} | {header_model}")
    print("-" * (width * 2 + 3))

    # 4. 逐行并排打印
    for gt, model in zip(gt_lines, model_lines):
        # 使用 ljust 确保左侧列对齐
        left_col  = f"{Color.GREEN}{gt.ljust(width)}{Color.END}"
        right_col = f"{Color.BLUE}{model.ljust(width)}{Color.END}"
        print(f"{left_col} | {right_col}")
    
    print("")

def inference_samples(num_samples):

    model_path = "./src/sft/scripts/checkpoint-full-long"
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()

    dataset = training_datasets.SupervisedDataset(
        tokenizer=tokenizer,
        data_path="./sft_processed_test.jsonl",
        args=argparse.Namespace(**{
            "model_max_length": 1280,
            "truncate_source": False
        })
    )

    total_data     = len(dataset)
    sample_indices = random.sample(range(total_data), min(num_samples, total_data))
    print(f"Total dataset size: {total_data}. Sampling {len(sample_indices)} cases...\n")

    # 编码识别符
    assistant_start_token  = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    assistant_start_tensor = torch.tensor(assistant_start_token)

    for idx in sample_indices:
        data_item = dataset[idx]
        input_ids_full = data_item["input_ids"]

        # 1. 寻找切分点
        break_idx = -1
        for i in range(len(input_ids_full) - len(assistant_start_tensor) + 1):
            if torch.equal(input_ids_full[i:i + len(assistant_start_tensor)], assistant_start_tensor):
                break_idx = i + len(assistant_start_tensor)
                break

        if break_idx == -1:
            continue

        # 2. 提取 Prompt
        prompt_ids      = input_ids_full[:break_idx].unsqueeze(0).to(model.device)
        readable_prompt = tokenizer.decode(prompt_ids[0], skip_special_tokens=False)

        # 3. 提取 Ground Truth (从 break_idx 到末尾，忽略填充部分)
        # 在 SupervisedDataset 中，labels 之外通常是 -100，input_ids 则是对应的 token
        # 我们直接从原始 input_ids 中截取后半部分
        gt_ids = input_ids_full[break_idx:]
        # 过滤掉可能的 padding (假设 pad_token_id 存在)
        if tokenizer.pad_token_id is not None:
            gt_ids = gt_ids[gt_ids != tokenizer.pad_token_id]
        # 过滤掉结束符以保持清洁
        gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True).strip()

        # 4. 模型生成
        with torch.no_grad():
            generated_ids = model.generate(
                prompt_ids,
                max_new_tokens=512,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        response_ids = generated_ids[0][prompt_ids.shape[-1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        # 5. 结构化打印对比
        # 只有在打印的时候还原所有的 \n \t
        print("=" * 60)
        print(f"{Color.BOLD}【Sample Index】: {idx}{Color.END}")
        print(f"【Input Prompt】:\n{restore_escaped_characters(readable_prompt)}")
        print("-" * 30)

        print_side_by_side(restore_escaped_characters(gt_text), restore_escaped_characters(response_text), width=60)

        print("=" * 60 + "\n")

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

def calculate_similarity(text_a, text_b):
    """
    Calculates a similarity score between 0.0 and 1.0.
    1.0 means exact match.
    """
    return difflib.SequenceMatcher(None, text_a, text_b).ratio()

def find_error_samples(output_file="bad_cases.json", similarity_threshold=0.7):
    """
    Scans the dataset, generates responses, and saves cases where
    similarity(GT, Prediction) < threshold.
    """
    model_path = "./src/sft/scripts/checkpoint-full-long"
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()

    dataset = training_datasets.SupervisedDataset(
        tokenizer=tokenizer,
        data_path="./sft_processed_test.jsonl",
        args=argparse.Namespace(**{
            "model_max_length": 8192,
            "truncate_source": False
        })
    )

    print(f"Scanning {len(dataset)} samples for errors (Threshold < {similarity_threshold})...")

    # Prepare token tensors for splitting
    assistant_start_token = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    assistant_start_tensor = torch.tensor(assistant_start_token)

    bad_cases = []

    # Iterate through ALL samples (or use tqdm(range(100)) for a quick test)
    for idx in tqdm(range(len(dataset))):
        data_item = dataset[idx]
        input_ids_full = data_item["input_ids"]

        # --- 1. Find the split point between Prompt and GT ---
        break_idx = -1
        # Optimized search: usually the assistant token is near the end of prompt,
        # but a linear scan is fine for inference
        for i in range(len(input_ids_full) - len(assistant_start_tensor) + 1):
            if torch.equal(input_ids_full[i:i + len(assistant_start_tensor)], assistant_start_tensor):
                break_idx = i + len(assistant_start_tensor)
                break

        if break_idx == -1:
            continue  # Skip malformed data

        prompt_ids = input_ids_full[:break_idx].unsqueeze(0).to(model.device)

        # Extract Ground Truth Text
        gt_ids = input_ids_full[break_idx:]
        if tokenizer.pad_token_id is not None:
            gt_ids = gt_ids[gt_ids != tokenizer.pad_token_id]
        gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True).strip()

        # --- 2. Generate Prediction ---
        with torch.no_grad():
            generated_ids = model.generate(
                prompt_ids,
                max_new_tokens=512,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        response_ids = generated_ids[0][prompt_ids.shape[-1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        # --- 3. Compare and Filter ---
        score = calculate_similarity(gt_text, response_text)

        if score < similarity_threshold:
            # Decode prompt for readability in the log file
            readable_prompt = tokenizer.decode(prompt_ids[0], skip_special_tokens=False)

            error_entry = {
                "index": idx,
                "input_ids": data_item["input_ids"].tolist(),
                "labels": data_item["labels"].tolist(),
                "score": round(score, 4),
                "prompt": readable_prompt,
                "ground_truth": gt_text,
                "prediction": response_text
            }
            bad_cases.append(error_entry)

            # Optional: Print it live if you want to see progress
            # print(f"Found error at idx {idx} | Score: {score:.2f}")

    # --- 4. Save Results ---
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(bad_cases, f, indent=2, ensure_ascii=False)

    print(f"\nScanning complete. Found {len(bad_cases)} bad cases.")
    print(f"Saved details to {output_file}")

def get_gen_results(train_dataset, indices, model, tokenizer,):
    results = {}
    model.eval()
    for idx in indices:
        sample = train_dataset[idx]
        results[idx] = generate_response(model, tokenizer, sample["input_ids"], model.device)
    return results

if __name__ == "__main__":
    # inference_samples(num_samples=10)

    find_error_samples(output_file="wrong_predictions_test.json", similarity_threshold=0.6)
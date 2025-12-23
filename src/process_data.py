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
from tree_sitter import Language, Parser
import tree_sitter_go as tsgo # pip install tree-sitter tree-sitter-go
from tree_sitter import QueryCursor

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HOME"]="/mnt/nvme0n1/hf_hub"
os.environ["TRANSFORMERS_CACHE"]="/mnt/nvme0n1/hf_hub"
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BatchDict = Dict[str, torch.Tensor]


def list_of_dicts_to_dict_of_lists(data_list):
    return {
        "system": [d["system"] for d in data_list],
        "input": [d["input"] for d in data_list],
        "output": [d["output"] for d in data_list]
    }

class CustomCollator:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features: List[Dict[str, Any]]) -> BatchDict:
        indices = []
        for f in features:
            if "sample_index" in f:
                indices.append(f.pop("sample_index"))

        batch = self.base_collator(features)

        if indices:
            batch["sample_index"] = torch.tensor(indices, dtype=torch.long)

        return batch

def extract_code_content(text):
    text = text.replace("This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. ", "")
    pattern = r"And here is the function you are asked to complete\s*([\s\S]*) Ensure that only missing codes marked as <MID> are returned"

    match = re.search(pattern, text)

    if match:
        extracted_part = match.group(1)
        return extracted_part.strip()
    else:
        return text

def strip_text_output(text):
    reformated_text = text.lstrip().replace("\t", "").strip()
    return reformated_text

def process_func_chatml(examples, tokenizer, max_len=2048):
    """
    将 input/output 转换为 ChatML 格式：
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {input}<|im_end|>
        <|im_start|>assistant
        {output}<|im_end|>
    并仅对 assistant 的回复部分计算 Loss。
    """
    systems  = examples["system"]
    inputs  = examples["input"]
    outputs = examples["output"]

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id  = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens  = tokenizer.encode("\n", add_special_tokens=False)

    def _build_turn(role, content, is_train=False):
        role_ids    = [im_start_id] + tokenizer.encode(role, add_special_tokens=False) + nl_tokens
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        footer_ids  = [im_end_id] + nl_tokens
        full_ids = role_ids + content_ids + footer_ids

        if is_train:
            # -100 表示忽略计算 Loss
            labels = [-100] * len(role_ids) + content_ids + footer_ids
        else:
            labels = [-100] * len(full_ids)
        return full_ids, labels

    new_input_ids = []
    new_labels = []
    # 注意：不再需要手动生成 attention_mask，DataCollator 会自动处理

    for sys, inp, outp in zip(systems, inputs, outputs):
        input_ids, labels = [], []

        # System
        sys_ids, sys_labels = _build_turn("system", sys, is_train=False)
        input_ids += sys_ids
        labels += sys_labels

        # User
        user_ids, user_labels = _build_turn("user", inp, is_train=False)
        input_ids += user_ids
        labels += user_labels

        # Assistant
        asst_ids, asst_labels = _build_turn("assistant", outp, is_train=True)
        input_ids += asst_ids
        labels += asst_labels

        # Truncation (仅截断，不 Padding)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

        new_input_ids.append(input_ids)
        new_labels.append(labels)

    return {
        "input_ids": new_input_ids,
        "labels": new_labels
    }


def get_ignored_token_ids(tokenizer):
    go_keywords = {
        "if", "else", "return", "func", "package", "import",
        "select", "break",
        "case", "continue",
        "for",
    }

    # 更加通用的符号匹配：匹配纯标点、纯空白符或它们的组合
    go_symbols_regex = re.compile(r'^[!\"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~\s]+$')
    ignored_ids = set()
    vocab = tokenizer.get_vocab()
    # 用于统计的计数器
    count_symbol_match = 0
    count_keyword_match = 0

    for token_str, token_id in vocab.items():
        # --- 解码关键步骤 ---
        # 使用 tokenizer 官方方法解码，确保处理了字节序列和特殊映射
        try:
            decoded_token = tokenizer.convert_tokens_to_string([token_str])
        except Exception as e:
            # 极少数情况下解码可能会失败，通常是特殊的控制字符
            # logger.warning(f"Failed to decode token ID {token_id} ({token_str}): {e}")
            decoded_token = token_str

        if not decoded_token:  # 跳过空字符串
            continue

        clean_content = decoded_token.strip()

        # --- 策略执行 ---

        # 策略 A：纯符号/空白符过滤 (优先执行，效率更高)
        # 如果整个 token 都是符号，直接屏蔽
        if go_symbols_regex.match(decoded_token):
            ignored_ids.add(token_id)
            count_symbol_match += 1
            continue

        # 策略 B：关键字匹配
        # 检查 token 中是否包含作为独立单词的关键字
        is_keyword_token = False
        for kw in go_keywords:
            # 使用 \b 确保匹配的是完整的单词边界
            # re.escape 用于处理关键字中可能包含的特殊字符（如 <|im_end|> 中的 |）
            if re.search(rf'\b{re.escape(kw)}\b', decoded_token):
                ignored_ids.add(token_id)
                is_keyword_token = True
                break

        if is_keyword_token:
            count_keyword_match += 1
            continue

    # 转换为 tensor
    ignored_ids.add(tokenizer.encode(tokenizer.eos_token)[0])
    ignored_tensor = torch.tensor(list(ignored_ids), dtype=torch.long)

    return ignored_tensor


if __name__ == '__main__':
    # JSONL_PATH = "/home/ruofan/git_space/Empirical-Influence-Function/small_data.jsonl"  # 请确保文件存在
    #
    # with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         if line.strip():
    #             obj = json.loads(line)
    #             if len(obj['messages']) > 2 and len(obj['messages'][2]['content']) > 0:
    #                 inp = extract_code_content(obj['messages'][1]['content'])
    #                 output = obj['messages'][2]['content']
    #                 print()

    # 测试用例

    # 初始化 Parser
    GO_LANGUAGE = Language(tsgo.language())
    parser = Parser(GO_LANGUAGE)
    example_code = """
    
        return res + 1
    """
    print(version_go_variables(example_code))
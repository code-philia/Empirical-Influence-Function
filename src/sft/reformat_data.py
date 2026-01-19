
import json
from ..process_data import extract_code_content, strip_text_output
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
import json
import re
import random  # 新增：用于随机打乱
from transformers import AutoTokenizer

MODEL_ID = "/mnt/nvme0n1/ruofan/git_space/Empirical-Influence-Function/src/sft/scripts/checkpoint-full"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

tokens = tokenizer.tokenize("\\t\\tb.Warning = &value\\n\\t\\treturn b")

def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for entry in data:
            json_record = json.dumps(entry, ensure_ascii=False)
            f.write(json_record + "\n")
    print(f"Saved {len(data)} samples to {filename}")

def clean_comment(outp: str) -> str:
    # 正则解释：
    # //       匹配双斜杠
    # [^\n\t]* 匹配任意数量的字符，只要不是 \n 或 \t
    return re.sub(r"//[^\n\t]*", "", outp)


# train_texts = []
# with open(JSONL_PATH, 'r', encoding='utf-8') as f:
#     for line in f:
#         if line.strip():
#             obj = json.loads(line)
#             if len(obj['messages']) > 2 and len(obj['messages'][2]['content']) > 0:
#                 inp  = extract_code_content(obj['messages'][1]['content'])
#                 outp = strip_text_output(obj['messages'][2]['content'])
#                 if "//" in outp:
#                     outp = clean_comment(outp)
#                 train_texts.append(
#                     {
#                         "messages": [
#                             {"role": "system",
#                              "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Your task is to complete the missing part of a Go function marked with `<MID>`."},
#                             {"role": "user", "content": inp},
#                             {"role": "assistant", "content": outp}
#                         ],
#                         "format": "chatml"
#                     }
#                 )
#
# with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
#     for entry in train_texts:
#         json_record = json.dumps(entry, ensure_ascii=False)
#         f.write(json_record + "\n")




JSONL_PATH = "big_data_L_2_5.jsonl"
TRAIN_OUTPUT_PATH = "sft_train.jsonl"  # 定义训练集文件名
TEST_OUTPUT_PATH = "sft_test.jsonl"  # 定义测试集文件名

ct = 0
train_texts = []

# --- 1. 读取并处理数据 ---
with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            # 简单的校验逻辑
            if len(obj['messages']) > 2 and len(obj['messages'][2]['content']) > 0:
                outp = obj['messages'][2]['content']
                if "//" in outp:
                    # 清理注释
                    outp_after = clean_comment(outp)
                    obj['messages'][2]['content'] = outp_after
                    ct += 1

            train_texts.append(obj)

print(f"Total processed samples: {len(train_texts)}")
print(f"Cleaned comments in {ct} samples.")

# --- 2. 打乱并划分数据集 ---
random.seed(42)  # 固定随机种子，保证每次划分一致
random.shuffle(train_texts)

split_ratio = 0.9  # 90% 用于训练，10% 用于测试
split_idx = int(len(train_texts) * split_ratio)

train_data = train_texts[:split_idx]
test_data = train_texts[split_idx:]


save_jsonl(train_data, TRAIN_OUTPUT_PATH)
save_jsonl(test_data, TEST_OUTPUT_PATH)


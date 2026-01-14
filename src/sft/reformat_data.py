
import json
from ..process_data import extract_code_content, strip_text_output
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

MODEL_ID = "/mnt/nvme0n1/ruofan/git_space/Empirical-Influence-Function/src/sft/scripts/checkpoint-full"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

tokens = tokenizer.tokenize("\\t\\tb.Warning = &value\\n\\t\\treturn b")

def clean_comment(outp: str) -> str:
    # 正则解释：
    # //       匹配双斜杠
    # [^\n\t]* 匹配任意数量的字符，只要不是 \n 或 \t
    return re.sub(r"//[^\n\t]*", "", outp)

JSONL_PATH  = "big_data_L_2_5.jsonl"  # 请确保文件存在
OUTPUT_PATH = "sft.jsonl"
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


ct = 0
train_texts = []
with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            if len(obj['messages']) > 2 and len(obj['messages'][2]['content']) > 0:
                sys = obj['messages'][0]['content']
                inp = obj['messages'][1]['content']
                outp = obj['messages'][2]['content']
                if "//" in outp:
                    outp_after = clean_comment(outp)
                    obj['messages'][2]['content'] = outp_after
                    ct += 1
            train_texts.append(obj)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for entry in train_texts:
        json_record = json.dumps(entry, ensure_ascii=False)
        f.write(json_record + "\n")


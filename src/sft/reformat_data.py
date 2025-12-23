
import json
from ..process_data import extract_code_content, strip_text_output

JSONL_PATH  = "small_data.jsonl"  # 请确保文件存在
OUTPUT_PATH = "sft.jsonl"
train_texts = []
with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            if len(obj['messages']) > 2 and len(obj['messages'][2]['content']) > 0:
                inp = extract_code_content(obj['messages'][1]['content'])
                outp = strip_text_output(obj['messages'][2]['content'])
                train_texts.append(
                    {
                        "messages": [
                            {"role": "system",
                             "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Your task is to complete the missing part of a Go function marked with `<MID>`."},
                            {"role": "user", "content": inp},
                            {"role": "assistant", "content": outp}
                        ],
                        "format": "chatml"
                    }
                )

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for entry in train_texts:
        json_record = json.dumps(entry, ensure_ascii=False)
        f.write(json_record + "\n")
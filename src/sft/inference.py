import torch
import random
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import training_datasets


def inference_samples(num_samples=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载模型与分词器 (路径建议设为变量)
    model_path = "./src/sft/checkpoint-full"
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16  # 推荐使用 bf16 节省显存且匹配训练
    ).eval()

    # 2. 加载数据集
    dataset = training_datasets.SupervisedDataset(
        tokenizer=tokenizer,
        data_path="./sft-processed.jsonl",
        args=argparse.Namespace(**{
            "model_max_length": 1280,
            "truncate_source": False
        })
    )

    # 3. 随机采样索引
    total_data = len(dataset)
    sample_indices = random.sample(range(total_data), min(num_samples, total_data))
    print(f"Total dataset size: {total_data}. Sampling {len(sample_indices)} cases...\n")

    assistant_start_token = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    assistant_start_tensor = torch.tensor(assistant_start_token)

    # 4. 循环生成
    for idx in sample_indices:
        data_item = dataset[idx]
        input_ids_full = data_item["input_ids"]

        # 定位 <|im_start|>assistant 的位置以截断 Prompt
        # 这种逻辑确保我们只给模型看 Prompt 部分
        break_idx = -1
        for i in range(len(input_ids_full) - len(assistant_start_tensor) + 1):
            if torch.equal(input_ids_full[i:i + len(assistant_start_tensor)], assistant_start_tensor):
                break_idx = i + len(assistant_start_tensor)
                break

        if break_idx == -1:
            print(f"Warning: Assistant start token not found in sample {idx}")
            continue

        prompt_ids = input_ids_full[:break_idx].unsqueeze(0).to(model.device)

        # 解码 Prompt 用于打印查看 (剔除特殊 token 让控制台更整洁)
        readable_prompt = tokenizer.decode(prompt_ids[0], skip_special_tokens=False)

        with torch.no_grad():
            generated_ids = model.generate(
                prompt_ids,
                max_new_tokens=512,  # 采样时可以先设小一点观察效果
                eos_token_id=tokenizer.eos_token_id
            )

        # 截断掉输入的 Prompt 部分，只留生成的回答
        response_ids = generated_ids[0][prompt_ids.shape[-1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # 5. 结构化打印 (ISTJ 风格：清晰、有分隔符)
        print("-" * 50)
        print(f"【Sample Index】: {idx}")
        print(f"【Input Prompt】:\n{readable_prompt}")
        print(f"\n【Model Response】:\n{response_text}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    inference_samples(num_samples=3)  # 这里修改你想查看的数量
# Finetune Qwen-Coder on Customized Data

## Step 1: Tokenize the data

```bash
python -m src.sft.binarize_data --input_path sft.jsonl --output_path sft-processed.jsonl
```

A ```sft-processed.jsonl``` file will be created where all training samples are tokenized.

## Step 2: Run SFT Training with LoRA

```bash
cd src/sft/scripts/
chmod +x ./sft_qwencoder_with_lora.sh
./sft_qwencoder_with_lora.sh
```

The adapter will be saved under ```src/sft/scripts/checkpoints```.

## Step 3: Merge LoRA Adapter into a Full Model Checkpoint

```bash
python -m src.sft.merge_adapter --base_model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --train_adapters_path ./src/sft/scripts/checkpoints --output_path ./src/sft/scripts/checkpoint-full
```

## Step 4: Test the Trained Model on Some Samples

```bash
python -m src.sft.inference
```
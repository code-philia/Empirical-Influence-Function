# Finetune Qwen-Coder on Customized Data

## Step 1: Preprocess the Data

Run 
```bash
python -m src.sft.reformat_data
```

Each sample will be parsed into the **ChatML** format, the extra explanation parts are removed.
```json
{
    "messages": [
        {
          "role": "system", 
          "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Your task is to complete the missing part of a Go function marked with `<MID>`."
        },
        {
          "role": "user", 
          "content": "func (c *integerArrayDescendingCursor) Stats() tsdb.CursorStats\nfunc (c *integerArrayDescendingCursor) Stats() tsdb.CursorStats { <MID> }"
        },
        {
          "role": "assistant", 
          "content": "mountedVolume = append(\nmountedVolume,\ngetMountedVolume(&podObj, &volumeObj))"
        }
    ],
    "format": "chatml"
}
```

After this step, all samples will be saved in a JSONL file ```sft.jsonl```, all samples share the same system instruction:

```json lines
{"messages": [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Your task is to complete the missing part of a Go function marked with `<MID>`."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "format": "chatml"}
{"messages": [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Your task is to complete the missing part of a Go function marked with `<MID>`."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "format": "chatml"}
{"messages": [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Your task is to complete the missing part of a Go function marked with `<MID>`."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "format": "chatml"}
```

## Step 2: Tokenize the data

```bash
python -m src.sft.binarize_data --input_path sft.jsonl --output_path sft-processed.jsonl
```

A ```sft-processed.jsonl``` file will be created where all training samples are tokenized.

## Step 3: Run SFT Training with LoRA

```bash
cd src/sft/scripts/
chmod +x ./sft_qwencoder_with_lora.sh
./sft_qwencoder_with_lora.sh
```

The adapter will be saved under ```src/sft/scripts/checkpoints```.

## Step 4: Merge LoRA Adapter into a Full Model Checkpoint

```bash
python -m src.sft.merge_adapter --base_model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --train_adapters_path ./src/sft/scripts/checkpoints --output_path ./src/sft/scripts/checkpoint-full
```

## Step 5: Test the Trained Model on Some Samples

```bash
python -m src.sft.inference
```
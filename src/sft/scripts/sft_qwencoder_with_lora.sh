#!/bin/bash

SCRIPT_PATH=$(readlink -f "$0")
# 获取脚本所在的目录: .../Empirical-Influence-Function/src/sft/scripts
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

# 根据你的目录结构推算根目录:
# scripts -> sft -> src -> PROJECT_ROOT (向上跳三级)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

# --- 基于项目根目录定义其他路径 ---
SRC_DIR="$PROJECT_ROOT/src/sft"
DATA_PATH="$PROJECT_ROOT/sft-processed.jsonl"
OUTPUT_DIR="$SCRIPT_DIR/checkpoints" # 放在脚本同级的 checkpoints 目录下

# 配置文件通常在脚本目录的 configs 下
DEEPSPEED_CONFIG="$PROJECT_ROOT/src/sft/configs/zero2.json"
PEFT_CONFIG_FOLDER="$PROJECT_ROOT/src/sft/configs/lora"

echo "--- Path Validation ---"
ls -l "$DATA_PATH" || echo "Error: Data path not found!"
ls -l "$DEEPSPEED_CONFIG" || echo "Error: DeepSpeed config not found!"
ls -l "$PEFT_CONFIG_FOLDER/adapter_config.json" || echo "Error: LoRA config not found!"
echo "--"

# --- 训练参数与环境配置 ---
PRETRAINED_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
RUNNAME="1k-sft"
export NCCL_DEBUG=WARN
export HF_HOME="/mnt/nvme0n1/hf_hub" # set to your huggingface cache dir
export TRANSFORMERS_CACHE="/mnt/nvme0n1/hf_hub" # set to your huggingface cache dir
export WANDB_PROJECT="EIF"

# --- 显卡与并行设置 ---
GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
WORLD_SIZE=$GPUS_PER_NODE # 假设单节点，多节点需调整
BATCH_SIZE=16
MICRO_BATCH_SIZE=1
EPOCH=5
GRAD_ACCU=$(($BATCH_SIZE / $WORLD_SIZE / $MICRO_BATCH_SIZE))

# --- 执行 ---
# 切换到 train.py 所在的目录
cd "$SRC_DIR"

echo "Using PROJECT_ROOT: $PROJECT_ROOT"
echo "Loading data from: $DATA_PATH"

http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --master_port 6105 \
    train.py \
    --model_name_or_path ${PRETRAINED_MODEL} \
    --data_path ${DATA_PATH} \
    --model_max_length 4096 \
    --truncate_source True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size ${MICRO_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCU} \
    --save_strategy "steps" \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --bf16 True \
    --use_peft True \
    --peft_config_path ${PEFT_CONFIG_FOLDER} \
    --run_name ${RUNNAME} \
    --report_to wandb
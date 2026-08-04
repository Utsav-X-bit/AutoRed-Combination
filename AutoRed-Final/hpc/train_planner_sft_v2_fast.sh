#!/bin/bash
set -euo pipefail

# Fast planner SFT launcher for AC30 subset.
#
# Single GPU:
#   CUDA_VISIBLE_DEVICES=0 ./hpc/train_planner_sft_v2_fast.sh
#
# Four GPUs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./hpc/train_planner_sft_v2_fast.sh --gpus 4

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false

GPUS=1
EPOCHS=3
OUTPUT_DIR="experiment/results/planner_sft_v2_fast"
MAX_LENGTH=1024
BATCH_SIZE=8
GRAD_ACCUM=4
LOGGING_STEPS=20
DATALOADER_WORKERS=4
PACKING=1
GRAD_CKPT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --max-length)
      MAX_LENGTH="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --grad-accum)
      GRAD_ACCUM="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

TRAIN_DATA="scripts/training/sft_data/planner_v2_train.jsonl"
VAL_DATA="scripts/training/sft_data/planner_v2_val.jsonl"
RUN_NAME="planner_sft_v2_fast_g${GPUS}_e${EPOCHS}"

if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "Missing $TRAIN_DATA"
  exit 1
fi

echo "============================================="
echo "AutoRed Planner SFT v2 Fast"
echo "============================================="
echo "GPUs         : $GPUS"
echo "Train Data   : $TRAIN_DATA"
echo "Val Data     : $VAL_DATA"
echo "Output       : $OUTPUT_DIR"
echo "Epochs       : $EPOCHS"
echo "Batch Size   : $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM)) per GPU"
echo "Max Length   : $MAX_LENGTH"
echo "Packing      : on"
echo "Grad Ckpt    : off"
echo "============================================="

CMD=(
  python scripts/training/train_qlo.py
  --model_name Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2
  --dataset "$TRAIN_DATA"
  --val_dataset "$VAL_DATA"
  --output_dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --gradient_accumulation "$GRAD_ACCUM"
  --learning_rate 2e-5
  --lora_r 32
  --lora_alpha 64
  --lora_dropout 0.05
  --max_length "$MAX_LENGTH"
  --device_map single
  --seed 42
  --run_name "$RUN_NAME"
  --packing
  --disable_gradient_checkpointing
  --logging_steps "$LOGGING_STEPS"
  --dataloader_num_workers "$DATALOADER_WORKERS"
)

if [[ "$GPUS" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$GPUS" "${CMD[@]:1}"
else
  "${CMD[@]}"
fi

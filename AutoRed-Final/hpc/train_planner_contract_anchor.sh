#!/bin/bash
set -euo pipefail

# Short anchor pass to tighten exact planner XML contract fidelity.

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false

GPUS=4
EPOCHS=1
BASE_ADAPTER="experiment/results/planner_sft_v2_contract_repair"
OUTPUT_DIR="experiment/results/planner_sft_v2_contract_anchor"
LOGGING_STEPS=20
DATALOADER_WORKERS=2

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
    --adapter)
      BASE_ADAPTER="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

python scripts/dataset_tools/build_planner_contract_anchor.py

TRAIN_DATA="scripts/training/sft_data/planner_contract_anchor_train.jsonl"
VAL_DATA="scripts/training/sft_data/planner_contract_anchor_val.jsonl"
RUN_NAME="planner_contract_anchor_g${GPUS}_e${EPOCHS}"

echo "============================================="
echo "Planner Contract Anchor"
echo "============================================="
echo "GPUs         : $GPUS"
echo "Base Adapter : $BASE_ADAPTER"
echo "Train Data   : $TRAIN_DATA"
echo "Val Data     : $VAL_DATA"
echo "Output       : $OUTPUT_DIR"
echo "Epochs       : $EPOCHS"
echo "============================================="

CMD=(
  python scripts/training/train_qlo.py
  --model_name Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2
  --dataset "$TRAIN_DATA"
  --val_dataset "$VAL_DATA"
  --adapter_path "$BASE_ADAPTER"
  --output_dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch_size 8
  --gradient_accumulation 4
  --learning_rate 3e-6
  --lora_r 32
  --lora_alpha 64
  --lora_dropout 0.05
  --max_length 1024
  --device_map single
  --seed 42
  --run_name "$RUN_NAME"
  --packing
  --disable_gradient_checkpointing
  --skip_best_model_reload
  --logging_steps "$LOGGING_STEPS"
  --dataloader_num_workers "$DATALOADER_WORKERS"
)

if [[ "$GPUS" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$GPUS" "${CMD[@]:1}"
else
  "${CMD[@]}"
fi

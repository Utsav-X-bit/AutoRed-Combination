#!/bin/bash
set -euo pipefail

# Short contract-repair fine-tune for planner tag fidelity.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./hpc/train_planner_contract_repair.sh \
#     --gpus 4 \
#     --adapter experiment/results/planner_sft_v2_fast_rerun \
#     --output-dir experiment/results/planner_sft_v2_contract_repair

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false

GPUS=1
EPOCHS=1
BASE_ADAPTER="experiment/results/planner_sft_v2_fast_rerun"
OUTPUT_DIR="experiment/results/planner_sft_v2_contract_repair"
LOGGING_STEPS=20
DATALOADER_WORKERS=4

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

python scripts/dataset_tools/build_planner_contract_repair.py

TRAIN_DATA="scripts/training/sft_data/planner_contract_repair_train.jsonl"
VAL_DATA="scripts/training/sft_data/planner_contract_repair_val.jsonl"
RUN_NAME="planner_contract_repair_g${GPUS}_e${EPOCHS}"

echo "============================================="
echo "Planner Contract Repair"
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
  --learning_rate 5e-6
  --lora_r 32
  --lora_alpha 64
  --lora_dropout 0.05
  --max_length 1024
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

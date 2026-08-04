#!/bin/bash
# =============================================================================
# AutoRed DPO Training Launcher
# =============================================================================
# Usage: sbatch hpc/train_dpo.slurm
#
# Requests 1 A100 GPU for DPO fine-tuning of the Generator.
# =============================================================================

#SBATCH --job-name=AutoRed_DPO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_dpo_%j.out
#SBATCH --error=logs/train_dpo_%j.err
#SBATCH --partition=airawatp

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PROJECT_ROOT="/nlsasfs/home/isea/isea11/slurmJobs/AutoRed"
cd "$PROJECT_ROOT"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Configuration
BASE_MODEL="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
DATASET="data/generator_dpo_dataset.jsonl"
OUTPUT_DIR="experiment/generator_dpo_adapter"
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=8
LORA_R=64
LR=1e-5

mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

echo "============================================="
echo "AutoRed DPO Training"
echo "============================================="
echo "Base Model : $BASE_MODEL"
echo "Dataset    : $DATASET"
echo "Output Dir : $OUTPUT_DIR"
echo "============================================="

python3 scripts/training/train_dpo.py \
    --model_name "$BASE_MODEL" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation $GRAD_ACCUM \
    --lora_r $LORA_R \
    --learning_rate $LR

echo "============================================="
echo "Training Complete"
echo "============================================="

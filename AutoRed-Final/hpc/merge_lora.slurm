#!/bin/bash
# =============================================================================
# AutoRed Merge LoRA Adapter
# =============================================================================
# Usage: sbatch hpc/merge_lora.slurm
#
# Requests 1 A100 GPU to load the base model and adapter, merge them, and save
# as a standalone model.
# =============================================================================

#SBATCH --job-name=AutoRed_Merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/merge_lora_%j.out
#SBATCH --error=logs/merge_lora_%j.err
#SBATCH --partition=airawatp

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PROJECT_ROOT="/nlsasfs/home/isea/isea11/slurmJobs/AutoRed"
cd "$PROJECT_ROOT"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Configuration
BASE_MODEL="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
ADAPTER="experiment/generator_dpo_adapter"
OUTPUT_DIR="experiment/generator_dpo_merged"

echo "============================================="
echo "AutoRed LoRA Merging"
echo "============================================="
echo "Base Model : $BASE_MODEL"
echo "Adapter    : $ADAPTER"
echo "Output Dir : $OUTPUT_DIR"
echo "============================================="

python3 scripts/training/merge_lora.py \
    --base_model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --output_dir "$OUTPUT_DIR"

echo "============================================="
echo "Merging Complete!"
echo "============================================="

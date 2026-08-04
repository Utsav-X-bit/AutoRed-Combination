#!/bin/bash
#SBATCH --job-name=autored_qlo_positive
#SBATCH --output=logs/autored_qlo_positive_%j.out
#SBATCH --error=logs/autored_qlo_positive_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=4:00:00
#SBATCH --partition=gpu

# Setup
mkdir -p logs
source .venv/bin/activate

# Hugging Face offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

echo "Starting QLoRA SFT training on positive dataset..."

# Move to project root if submitted from hpc directory
if [[ $PWD == */hpc ]]; then
    cd ..
fi

# Dataset B: positive (291 samples, 235 train + 56 val)
python scripts/training/train_qlo.py \
    --dataset scripts/training/sft_data/variantc_positive_train.jsonl \
    --val_dataset scripts/training/sft_data/variantc_positive_val.jsonl \
    --output_dir experiment/results/qlo_positive_v1 \
    --epochs 6 \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --learning_rate 2e-5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --max_length 1024 \
    --seed 42 \
    --run_name "qlo_positive_variantc_v1"

echo "QLoRA Training on positive dataset completed."

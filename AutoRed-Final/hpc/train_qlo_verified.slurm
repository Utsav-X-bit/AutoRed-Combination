#!/bin/bash
#SBATCH --job-name=autored_qlo_verified
#SBATCH --output=logs/autored_qlo_verified_%j.out
#SBATCH --error=logs/autored_qlo_verified_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=4:00:00
#SBATCH --partition=airawatp

# Setup
mkdir -p logs
source .venv/bin/activate

# Hugging Face offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

echo "Starting QLoRA SFT training on verified dataset..."

# Move to project root if submitted from hpc directory
if [[ $PWD == */hpc ]]; then
    cd ..
fi

# Dataset A: verified (138 samples, 112 train + 26 val)
python scripts/training/train_qlo.py \
    --dataset scripts/training/sft_data/variantc_verified_train.jsonl \
    --val_dataset scripts/training/sft_data/variantc_verified_val.jsonl \
    --output_dir experiment/results/qlo_verified_v1 \
    --epochs 10 \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --learning_rate 2e-5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --max_length 1024 \
    --seed 42 \
    --run_name "qlo_verified_variantc_v1"

echo "QLoRA Training on verified dataset completed."

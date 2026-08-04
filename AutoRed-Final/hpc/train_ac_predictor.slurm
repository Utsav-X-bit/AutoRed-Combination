#!/bin/bash
#SBATCH --job-name=ac_predictor      # Job name
#SBATCH --output=logs/ac_predictor_%j.out
#SBATCH --error=logs/ac_predictor_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=7-00:00:00
#SBATCH --partition=airawatp

mkdir -p logs
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source .venv/bin/activate

echo "Starting Access Code Type Predictor training..."
python scripts/training/train_access_code_predictor.py

echo "Training Completed."

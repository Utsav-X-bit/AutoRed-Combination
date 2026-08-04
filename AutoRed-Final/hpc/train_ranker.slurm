#!/bin/bash
#SBATCH --job-name=autored_ranker
#SBATCH --partition=airawatp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ranker_train_%j.out
#SBATCH --error=logs/ranker_train_%j.err

# Setup
mkdir -p logs
source .venv/bin/activate

# Hugging Face offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
# export HF_HOME=/work/utsav/hf_cache
# export TRANSFORMERS_CACHE=/work/utsav/hf_cache


# Move to project root if submitted from hpc directory
if [[ $PWD == */hpc ]]; then
    cd ..
fi

echo "Starting ranker dataset build..."
python scripts/training/build_ranker_dataset.py \
    --data-dir data/ \
    --results-dir results/ \
    --output-dir data/

echo "Starting ranker training..."
python scripts/training/train_ranker.py \
    --train-file data/ranker_dataset_train_v1.jsonl \
    --val-file data/ranker_dataset_val_v1.jsonl \
    --test-file data/ranker_dataset_test_v1.jsonl \
    --output-dir models/ranker_deberta_v1/ \
    --model-name microsoft/deberta-v3-base \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 2e-5

echo "Ranker training completed. Results saved to models/ranker_deberta_v1/"

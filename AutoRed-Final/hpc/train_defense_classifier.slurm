#!/bin/bash
#SBATCH --job-name=defense_classifier
#SBATCH --output=logs/defense_classifier_%j.out
#SBATCH --error=logs/defense_classifier_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --partition=airawatp

mkdir -p logs
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Use local project virtualenv
source .venv/bin/activate

# Move to project root if submitted from hpc directory
if [[ $PWD == */hpc ]]; then
    cd ..
fi

echo "Merging dataset parts..."
cat data/defense_classifier_dataset-Part*.jsonl > data/defense_classifier_dataset.jsonl

echo "Starting Defense Classifier training..."
python scripts/training/train_defense_classifier.py

echo "Training Completed."

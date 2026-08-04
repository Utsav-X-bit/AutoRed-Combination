#!/bin/bash
#SBATCH --job-name=autored_reward      # Job name
#SBATCH --output=logs/autored_reward_%j.out
#SBATCH --error=logs/autored_reward_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=airawatp
#SBATCH --time=1-00:00:00

mkdir -p logs
source ../.venv/bin/activate

# Ensure datasets and transformers are in offline mode for the actual training
# But first, make sure you ran the download script on the login node!
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Starting Reward Model (Stop Point Identifier) training..."
cd ..
python hpc/train_reward_model.py

echo "Reward Model Training Completed."

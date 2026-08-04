#!/bin/bash
#SBATCH --job-name=autored_predictor       # Job name
#SBATCH --output=logs/autored_predictor_%j.out  # Standard output log
#SBATCH --error=logs/autored_predictor_%j.err   # Standard error log
#SBATCH --nodes=1                          # Run all processes on a single node
#SBATCH --ntasks=1                         # Run a single task
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH --partition=airawatp               # Partition (Queue) name
#SBATCH --time=01:00:00                    # Time limit hrs:min:sec

# Create log directory if it doesn't exist
mkdir -p logs

# Activate your virtual environment
source .venv/bin/activate

echo "Starting Strategy Predictor Training..."

# Run the predictor training script
python3 scripts/training/train_strategy_predictor.py

echo "Training Completed."

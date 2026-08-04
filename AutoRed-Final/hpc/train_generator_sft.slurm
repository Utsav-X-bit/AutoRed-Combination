#!/bin/bash
#SBATCH --job-name=autored_sft         # Job name
#SBATCH --output=logs/autored_sft_%j.out    # Standard output log
#SBATCH --error=logs/autored_sft_%j.err     # Standard error log
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --partition=airawatp            # Partition (Queue) name (change if needed)
#SBATCH --time=12:00:00                # Time limit hrs:min:sec

# Load required modules (Modify according to your HPC environment)
# module load python/3.10
# module load cuda/11.8

# Create log directory if it doesn't exist
mkdir -p logs

# Activate your virtual environment
source .venv/bin/activate

# Set Hugging Face offline mode flags for compute nodes with no internet
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Ensure WandB runs in offline mode if you don't have internet
export WANDB_MODE=offline

echo "Starting Supervised Fine-Tuning (SFT) for the Malicious Prompt Generator..."


# Run the SFT training script
python3 scripts/training/train_generator_sft.py

echo "SFT Training Completed."

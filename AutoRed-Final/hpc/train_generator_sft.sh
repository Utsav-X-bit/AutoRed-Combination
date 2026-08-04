#!/bin/bash
#SBATCH --job-name=autored_sft         # Job name
#SBATCH --output=logs/autored_sft_%j.out    # Standard output log
#SBATCH --error=logs/autored_sft_%j.err     # Standard error log
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --gres=gpu:A100-SXM4:4
#SBATCH --partition=airawatp
#SBATCH --time=7-00:00:00

# Load required modules (Modify according to your HPC environment)
# module load python/3.10
# module load cuda/11.8

# Create log directory if it doesn't exist
mkdir -p logs

# Activate your virtual environment
source ../.venv/bin/activate

# Set Hugging Face offline mode flags for compute nodes with no internet
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Ensure WandB runs in offline mode if you don't have internet
export WANDB_MODE=offline
export HF_DATASETS_TRUST_REMOTE_CODE=1

echo "Starting Supervised Fine-Tuning (SFT) for the Malicious Prompt Generator..."
cd ..

# Run the SFT training script
python scripts/training/train_text_generation.py \
    --config_path scripts/training/task_configs/pi_gen/pi_supervised.yml \
    --project_name AutoRed_Generator \
    --experiment_name SFT_T5_Base \
    --base_path_to_store_results ./experiment/results/sft

echo "SFT Training Completed."

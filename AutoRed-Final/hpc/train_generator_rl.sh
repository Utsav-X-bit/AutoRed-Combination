#!/bin/bash
#SBATCH --job-name=autored_rl          # Job name
#SBATCH --output=logs/autored_rl_%j.out     # Standard output log
#SBATCH --error=logs/autored_rl_%j.err      # Standard error log
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=64G                      # Job memory request (RL takes more RAM)
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --partition=gpu                # Partition (Queue) name (change if needed)
#SBATCH --time=24:00:00                # Time limit hrs:min:sec

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

echo "Starting Reinforcement Learning (NLPO) for the Malicious Prompt Generator..."
cd ..

# Run the RL (NLPO) training script
# Note: Ensure the base_model in pi_nlpo.yml points to the output of your SFT run!
python scripts/training/train_text_generation.py \
    --config_path scripts/training/task_configs/pi_gen/pi_nlpo.yml \
    --project_name AutoRed_Generator \
    --experiment_name RL_NLPO_T5_Base \
    --base_path_to_store_results ./experiment/results/rl

echo "RL Training Completed."

#!/bin/bash
#SBATCH --job-name=download_bases
#SBATCH --output=logs/download_bases_%j.out
#SBATCH --error=logs/download_bases_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=airawatp  # Adjust to a partition with internet access if available

# NOTE: Compute nodes typically do NOT have internet access on most clusters.
# If your partition does not have internet access, please run directly on the login node:
#   source .venv/bin/activate
#   python hpc/download_bases.py

mkdir -p logs
source .venv/bin/activate

# Disable offline mode for downloading
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0

# Move to project root if submitted from hpc directory
if [[ $PWD == */hpc ]]; then
    cd ..
fi

python hpc/download_bases.py

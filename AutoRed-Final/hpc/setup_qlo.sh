#!/bin/bash
# Setup script for QLoRA training environment on HPC
# Run this once before submitting training jobs

set -e

echo "Setting up QLoRA training environment..."

# Navigate to project root
cd "$(dirname "$0")/.."

# Create directories
mkdir -p logs
mkdir -p experiment/results

# Install training dependencies
source .venv/bin/activate

echo "Installing training dependencies..."
uv pip install -r requirements_qlo.txt

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
import peft
import bitsandbytes
import trl
import transformers
import datasets

print(f'  torch: {torch.__version__}')
print(f'  peft: {peft.__version__}')
print(f'  bitsandbytes: {bitsandbytes.__version__}')
print(f'  trl: {trl.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  datasets: {datasets.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "Setup complete. You can now submit training jobs:"
echo "  sbatch hpc/train_qlo_verified.slurm"
echo "  sbatch hpc/train_qlo_positive.slurm"

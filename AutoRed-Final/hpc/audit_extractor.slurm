#!/bin/bash
#SBATCH --job-name=autored_extractor_audit
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/extractor_audit_%j.out
#SBATCH --error=logs/extractor_audit_%j.err

# Setup
mkdir -p logs
source /nlsasfs/home/isea/isea13/AutoRed/.venv/bin/activate

# Move to project root if submitted from hpc directory
if [[ $PWD == */hpc ]]; then
    cd ..
fi

echo "Starting extractor audit..."
python scripts/analysis/audit_extractor.py \
    --data-dir data/ \
    --results-dir results/ \
    --output-dir data/

echo "Extractor audit completed."

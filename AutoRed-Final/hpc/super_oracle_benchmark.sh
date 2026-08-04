#!/bin/bash
# =============================================================================
# AutoRed Super Oracle v4.1 Speed-Optimized Multi-GPU Parallel Launcher
# =============================================================================
# Usage: sbatch hpc/super_oracle_benchmark.slurm [N_STRATEGIES] [NUM_SCENARIOS] [MAX_ATTEMPTS] [DATASET]
#
# Examples:
#   sbatch hpc/super_oracle_benchmark.slurm 10 1000       # 10 strategies, 1000 scenarios, 10 attempts
#   sbatch hpc/super_oracle_benchmark.slurm 10 1000 8     # 10 strategies, 1000 scenarios, 8 attempts
#   sbatch hpc/super_oracle_benchmark.slurm 15 500 10 experiment/oracle_v3_scenarios_5000.jsonl.bz2
#
# This script requests 8 A100 GPUs and launches 8 concurrent worker processes
# to run the Super Oracle v4 (Intelligence-Driven Best-of-N Search).
# =============================================================================

#SBATCH --job-name=SuperOracle_v4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/super_oracle_%j.out
#SBATCH --error=logs/super_oracle_%j.err
#SBATCH --partition=airawatp

# =============================================================================
# Configuration
# =============================================================================

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

N_STRATEGIES=${1:-10}
NUM_SCENARIOS=${2:-1000}
MAX_ATTEMPTS=${3:-6}
DATASET=${4:-"experiment/raw_dump_defenses.jsonl.bz2"}
NUM_GPUS=8

# Project root
PROJECT_ROOT="/nlsasfs/home/isea/isea11/slurmJobs/AutoRed"
cd "$PROJECT_ROOT"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# =============================================================================
# Setup
# =============================================================================
mkdir -p "data"
mkdir -p "logs"

echo "============================================="
echo "AutoRed Super Oracle v4.1 — 8-GPU Speed Run"
echo "============================================="
echo "Strategies/Step : $N_STRATEGIES"
echo "Total Scenarios : $NUM_SCENARIOS"
echo "Max Attempts    : $MAX_ATTEMPTS"
echo "GPUs            : $NUM_GPUS"
echo "Filter          : ON (max_code_len=25)"
echo "Early Terminate : ON (2-response plateau + dup detection)"
echo "Power Combos    : ON (30% of candidates)"
echo "Adaptive K      : ON (100%/70%/50% decay)"
echo "============================================="

# =============================================================================
# Launch Workers
# =============================================================================
PIDS=()
for WORKER_ID in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=$WORKER_ID
    WORKER_LOG="logs/super_oracle_v4_worker_${WORKER_ID}.log"

    echo ""
    echo "[LAUNCH] Worker $WORKER_ID on GPU $GPU_ID"
    echo "         Log: $WORKER_LOG"

    # Launch worker on specific GPU
    env CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/dataset_tools/super_oracle.py \
        --n "$N_STRATEGIES" \
        --scenarios "$NUM_SCENARIOS" \
        --max-attempts "$MAX_ATTEMPTS" \
        --dataset "$DATASET" \
        --worker-id "$WORKER_ID" \
        --num-workers "$NUM_GPUS" \
        > "$WORKER_LOG" 2>&1 &

    PIDS+=($!)

    # Stagger workers by 10s to avoid NFS I/O contention during model loading
    if [ $WORKER_ID -lt $((NUM_GPUS - 1)) ]; then
        echo "Waiting 10s before launching next worker..."
        sleep 10
    fi
done

echo ""
echo "[INFO] All $NUM_GPUS workers launched in background."
echo "Waiting for all workers to finish..."

# Wait for all background processes
FAIL=0
for PID in "${PIDS[@]}"; do
    wait $PID || let "FAIL+=1"
done

if [ "$FAIL" == "0" ]; then
    echo "✅ ALL WORKERS COMPLETED SUCCESSFULLY."
    
    # Merge v4 worker files
    cat data/oracle_trajectories_v3_w*.jsonl > data/oracle_trajectories_v4.jsonl
    echo "Merged trajectories saved to data/oracle_trajectories_v4.jsonl"
    
    # Quick stats
    TOTAL=$(wc -l < data/oracle_trajectories_v4.jsonl)
    SUCCESSES=$(grep -c '"success": true' data/oracle_trajectories_v4.jsonl || echo 0)
    echo "Total: $TOTAL | Successes: $SUCCESSES"
else
    echo "❌ $FAIL WORKER(S) FAILED. Check logs for details."
    exit 1
fi


echo "All work finished. Keeping ..."
tail -f /dev/null
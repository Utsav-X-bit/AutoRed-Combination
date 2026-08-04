#!/bin/bash
# =============================================================================
# AutoRed Multi-GPU Parallel Benchmark Launcher (Batched)
# =============================================================================
# Usage: sbatch hpc/autored_benchmark_4gpu.slurm [NUM_ROUNDS] [GENERATOR_PATH] [BASE_MODEL_PATH]
#
# Examples:
#   sbatch hpc/autored_benchmark_4gpu.slurm 1000 "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
#   sbatch hpc/autored_benchmark_4gpu.slurm 1000 "experiment/results/qlora_adapter" "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
#
# This script requests 4 A100 GPUs and launches 4 concurrent worker processes.
# Thanks to the recent batched generation rewrite (BATCH_SIZE=16 per GPU),
# this will evaluate 64 scenarios simultaneously.
#
# Estimated Time for 1000 rounds: ~30 to 45 minutes
# =============================================================================

#SBATCH --job-name=AutoRed_4GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_4gpu_%j.out
#SBATCH --error=logs/benchmark_4gpu_%j.err
#SBATCH --partition=airawatp

# =============================================================================
# Configuration
# =============================================================================

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

NUM_ROUNDS=${1:-1000}
GENERATOR_PATH=${2:-"Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"}
BASE_GENERATOR_PATH=${3:-""}
DATASET_PATH=${4:-""}
NUM_GPUS=4
DATASET_SIZE=${5:-1000}
OUTPUT_DIR=${6:-"results/benchmarks/batched_${NUM_ROUNDS}r_4g"}

# Project root
PROJECT_ROOT="/nlsasfs/home/isea/isea38/AutoRed"
cd "$PROJECT_ROOT"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Offline mode (models pre-downloaded on HPC)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# =============================================================================
# Setup
# =============================================================================
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

echo "============================================="
echo "AutoRed Batched 4-GPU Benchmark"
echo "============================================="
echo "Total Rounds : $NUM_ROUNDS"
echo "GPUs         : $NUM_GPUS"
echo "Generator    : $GENERATOR_PATH"
if [ -n "$BASE_GENERATOR_PATH" ]; then
    echo "Base Model   : $BASE_GENERATOR_PATH"
fi
echo "Output Dir   : $OUTPUT_DIR"
echo "============================================="

# =============================================================================
# Launch Workers
# =============================================================================
PIDS=()
for WORKER_ID in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=$WORKER_ID
    WORKER_OUTPUT="$OUTPUT_DIR/worker_${WORKER_ID}.json"
    WORKER_LOG="logs/batched_worker_${WORKER_ID}.log"

    echo ""
    echo "[LAUNCH] Worker $WORKER_ID on GPU $GPU_ID (Processing 16 scenarios at a time)"
    echo "         Output: $WORKER_OUTPUT"

    # Launch worker on specific GPU — use env to ensure CUDA_VISIBLE_DEVICES
    # is set in the process, not just the shell variable
    env CUDA_VISIBLE_DEVICES=$GPU_ID python experiment/llama_3_8b_verbose.py \
        --mode benchmark \
        --rounds "$NUM_ROUNDS" \
        --dataset-size "$DATASET_SIZE" \
        --benchmark-output "$WORKER_OUTPUT" \
        --worker-id "$WORKER_ID" \
        --num-workers "$NUM_GPUS" \
        --generator-path "$GENERATOR_PATH" \
        $( [ -n "$BASE_GENERATOR_PATH" ] && echo "--base-generator-path $BASE_GENERATOR_PATH" ) \
        $( [ -n "$DATASET_PATH" ] && echo "--dataset-path $DATASET_PATH" ) \
        > "$WORKER_LOG" 2>&1 &

    PIDS+=($!)

    # Stagger workers by 5s to avoid NFS I/O contention during model loading
    if [ $WORKER_ID -lt $((NUM_GPUS - 1)) ]; then
        sleep 5
    fi
done

echo ""
echo "============================================="
echo "All $NUM_GPUS workers launched in background. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."
echo "============================================="

# =============================================================================
# Wait for All Workers
# =============================================================================
FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "[ERROR] Worker PID $PID failed!"
        FAILED=1
    fi
done

if [ $FAILED -ne 0 ]; then
    echo ""
    echo "[ERROR] One or more workers failed. Check logs/batched_worker_*.log for details."
    echo "Partial results in: $OUTPUT_DIR/"
    exit 1
fi

echo ""
echo "============================================="
echo "All workers completed successfully!"
echo "============================================="

# =============================================================================
# Merge Results
# =============================================================================
echo ""
echo "[MERGE] Combining results from $NUM_GPUS workers..."

python scripts/merge_benchmarks.py \
    --output "$OUTPUT_DIR/merged_summary.json" \
    --worker-results "$OUTPUT_DIR"/worker_*.json

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================="
    echo "Benchmark complete!"
    echo "Merged results: $OUTPUT_DIR/merged_summary.json"
    echo "============================================="
else
    echo "[ERROR] Merge script failed!"
    exit 1
fi

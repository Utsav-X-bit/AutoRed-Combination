#!/bin/bash
# =============================================================================
# Multi-GPU Parallel Benchmark Launcher
# =============================================================================
# Usage: sbatch hpc/benchmark_multigpu.sh [NUM_ROUNDS] [NUM_GPUS] [GENERATOR_PATH]
#
# Example:
#   sbatch hpc/benchmark_multigpu.sh 1000 4 Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2
#   sbatch hpc/benchmark_multigpu.sh 1000 4 experiment/results/qlo_positive_v1
#
# Each GPU runs an independent Python process, processing a disjoint slice
# of scenarios. Results are merged automatically after all workers finish.
# =============================================================================

#SBATCH --job-name=AutoRed_MultiGPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:4
#SBATCH --time=7-00:00:00
#SBATCH --output=benchmark_multigpu_%j.out
#SBATCH --error=benchmark_multigpu_%j.err
#SBATCH --partition=airawatp

# =============================================================================
# Configuration
# =============================================================================
NUM_ROUNDS=${1:-1000}          # Total benchmark rounds across all GPUs
NUM_GPUS=${2:-4}               # Number of GPUs to use
GENERATOR_PATH=${3:-Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2}
BASE_GENERATOR_PATH=${4:-}     # Optional: base model for LoRA adapters
DATASET_SIZE=${5:-1000}        # Pool size for scenario sampling
OUTPUT_DIR="results/benchmarks/multigpu_${NUM_ROUNDS}r_${NUM_GPUS}g"

# Project root
PROJECT_ROOT="/nlsasfs/home/isea/isea38/AutoRed"
cd "$PROJECT_ROOT"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Offline mode (models pre-downloaded)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# =============================================================================
# Setup
# =============================================================================
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

echo "============================================="
echo "AutoRed Multi-GPU Benchmark"
echo "============================================="
echo "Total Rounds : $NUM_ROUNDS"
echo "GPUs         : $NUM_GPUS"
echo "Generator    : $GENERATOR_PATH"
echo "Output Dir   : $OUTPUT_DIR"
echo "============================================="

# =============================================================================
# Launch Workers
# =============================================================================
PIDS=()
for WORKER_ID in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=$WORKER_ID
    WORKER_OUTPUT="$OUTPUT_DIR/worker_${WORKER_ID}.json"
    WORKER_LOG="logs/worker_${WORKER_ID}.log"

    echo ""
    echo "[LAUNCH] Worker $WORKER_ID on GPU $GPU_ID"
    echo "         Output: $WORKER_OUTPUT"

    CUDA_VISIBLE_DEVICES=$GPU_ID python experiment/llama_3_8b_verbose.py \
        --mode benchmark \
        --rounds "$NUM_ROUNDS" \
        --dataset-size "$DATASET_SIZE" \
        --generator-path "$GENERATOR_PATH" \
        $( [ -n "$BASE_GENERATOR_PATH" ] && echo "--base-generator-path $BASE_GENERATOR_PATH" ) \
        --benchmark-output "$WORKER_OUTPUT" \
        --worker-id "$WORKER_ID" \
        --num-workers "$NUM_GPUS" \
        2>&1 | tee "$WORKER_LOG" &

    PIDS+=($!)
done

echo ""
echo "============================================="
echo "All workers launched. PIDs: ${PIDS[*]}"
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
    echo "[ERROR] One or more workers failed. Check logs/ for details."
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
    echo "Individual results: $OUTPUT_DIR/worker_*.json"
    echo "Logs: logs/worker_*.log"
    echo "============================================="
else
    echo "[ERROR] Merge failed!"
    exit 1
fi

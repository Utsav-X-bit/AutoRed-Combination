#!/bin/bash
#SBATCH --job-name=AutoRed_4GPU
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:4
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/benchmark_4gpu_%j.out
#SBATCH --error=logs/benchmark_4gpu_%j.err
#SBATCH --partition=airawatp

# =============================================================================
# Configuration
# =============================================================================

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

NUM_ROUNDS=${1:-1000}
PLANNER_PATH=${2:-"experiment/results/planner_sft_v2_contract_anchor/checkpoint-27"}
GENERATOR_PATH=${3:-"experiment/results/generator_sft_v2"}
BASE_GENERATOR_PATH=${4:-"Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"}
DATASET_PATH=${5:-"data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl"}
NUM_GPUS=4
DATASET_SIZE=${6:-1000}
OUTPUT_DIR=${7:-"results/benchmark/batched_${NUM_ROUNDS}r_4gpu"}
VICTIM_MODEL_ID=${8:-"meta-llama/Meta-Llama-3-8B-Instruct"}

# Project root
PROJECT_ROOT="/nlsasfs/home/isea/isea38/AutoRed-Final"
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
echo "Planner      : $PLANNER_PATH"
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
    # Resolve the per-worker log path under the new tree: results/benchmark/<model>/<chars>/logs/
    LOGS_DIR=$(python -c "from experiment.results_layout import resolve_model_id, parse_output_dir, runs_root; r=runs_root('$OUTPUT_DIR','benchmark',resolve_model_id('$VICTIM_MODEL_ID'),parse_output_dir('$OUTPUT_DIR','benchmark')[1]); print(r/'logs')")
    mkdir -p "$LOGS_DIR"
    WORKER_LOG="$LOGS_DIR/worker_${WORKER_ID}.log"

    echo ""
    echo "[LAUNCH] Worker $WORKER_ID on GPU $GPU_ID (Processing 16 scenarios at a time)"
    echo "         Output: $WORKER_OUTPUT"

    # Launch worker on specific GPU — use env to ensure CUDA_VISIBLE_DEVICES
    # is set in the process, not just the shell variable
    env CUDA_VISIBLE_DEVICES=$GPU_ID python experiment/llama_3_8b_vllm.py \
        --mode benchmark \
        --rounds "$NUM_ROUNDS" \
        --dataset-size "$DATASET_SIZE" \
        --benchmark-output "$WORKER_OUTPUT" \
        --output-dir "$OUTPUT_DIR" \
        --victim-model-id "$VICTIM_MODEL_ID" \
        --worker-id "$WORKER_ID" \
        --num-workers "$NUM_GPUS" \
        --planner-path "$PLANNER_PATH" \
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
    echo "[ERROR] One or more workers failed. Check ${LOGS_DIR:-logs}/worker_*.log for details."
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

LOGS_DIR=$(python -c "from experiment.results_layout import resolve_model_id, parse_output_dir, runs_root; r=runs_root('$OUTPUT_DIR','benchmark',resolve_model_id('$VICTIM_MODEL_ID'),parse_output_dir('$OUTPUT_DIR','benchmark')[1]); print(r/'logs')")

python scripts/merge_benchmarks.py \
    --output "$LOGS_DIR" \
    --worker-results "$LOGS_DIR"/worker_*.json

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================="
    echo "Benchmark complete!"
    echo "Merged results: $LOGS_DIR/merged_summary.json"
    echo "============================================="
else
    echo "[ERROR] Merge script failed!"
    exit 1
fi

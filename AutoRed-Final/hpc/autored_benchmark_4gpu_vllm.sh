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

# Defaults
NUM_ROUNDS=1000
PLANNER_PATH="experiment/results/planner_sft_v2_contract_anchor/checkpoint-27"
GENERATOR_PATH="experiment/results/generator_sft_v2"
BASE_GENERATOR_PATH="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
DATASET_PATH="data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl"
NUM_GPUS=4
DATASET_SIZE=1000
MAX_ATTEMPTS=20
GPU_MEMORY_UTILIZATION=0.40
SHARED_GPU_MEMORY_UTILIZATION=0.55
VICTIM_MAX_MODEL_LEN=2048
OUTPUT_DIR=""
VICTIM_MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
START_IDX=""
SEED="42"
PLANNER_TEMP_ESCALATION="0.0"
# Task 3/5 (model-agnostic v2): cooperative seeding (default ON) + BoN N cap.
# Empty COOPERATIVE_SEEDING = default-ON; "no" disables (--no-cooperative-seeding).
# Empty COOPERATIVE_N = no scaling (runtime default 8); set e.g. 12 to enable BoN.
COOPERATIVE_SEEDING=""
COOPERATIVE_N=""
# Benchmark grouping (Change 3): when --output-dir is not passed, the script
# auto-generates a two-level path results/benchmarks/{BENCHMARK_NAME}/{TS}_4g
# so runs of the same logical benchmark group together. Leave empty to use the
# legacy flat default.
BENCHMARK_NAME=""
TRUST_REMOTE_CODE=0
TOKENIZER_MODE="auto"
VICTIM_QUANTIZATION=""

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --rounds N                 Total benchmark rounds across all GPUs (default: 1000)"
    echo "  --planner-path PATH        Planner model path or LoRA adapter"
    echo "  --generator-path PATH      Generator model path or LoRA adapter"
    echo "  --base-generator-path PATH Base model for LoRA adapters"
    echo "  --dataset-path PATH        Path to defense dataset JSONL"
    echo "  --dataset-size N           Number of scenarios to load from dataset (default: 1000)"
    echo "  --output-dir PATH          Directory for per-worker and merged results"
    echo "  --benchmark-name NAME      Logical benchmark name; when --output-dir is unset, auto-generates"
    echo "                                   results/benchmarks/{NAME}/{timestamp}_4g (groups repeated runs)"
    echo "  --victim-model-id ID       Hugging Face model id for victim LLM (default: meta-llama/Meta-Llama-3-8B-Instruct)"
    echo "  --start-idx N              Zero-based start index for deterministic benchmark slice"
    echo "  --attempts N               Maximum attack attempts per scenario (default: 20)"
    echo "  --max-attempts N           Alias for --attempts"
    echo "  --trust-remote-code              Trust remote modeling code for the victim LLM"
    echo "  --tokenizer-mode MODE            vLLM tokenizer mode (default: auto; use 'mistral' for newer Mistral tokenizer files)"
    echo "  --gpu-memory-utilization F       vLLM GPU memory fraction for victim (default: 0.40)"
    echo "  --shared-gpu-memory-utilization F vLLM GPU memory fraction for shared planner/generator (default: 0.55)"
    echo "  --victim-max-model-len N         vLLM max_model_len for the victim model; lower reduces KV cache (default: 2048)"
    echo "  --victim-quantization METHOD    vLLM quantization for victim (e.g., bitsandbytes, awq, gptq)"
    echo "  --mutation-fallback              Enable JailGuard mutation fallback on failed scenarios"
    echo "  --max-fallback-rounds N          Mutation fallback rounds (1 default, 2 adaptive)"
    echo "  --seed N                         Random seed for dataset sampling + fallback RNG (default: 42)."
    echo "                                   Two runs sharing --seed and --start-idx are directly comparable."
    echo "  --planner-temp-escalation F     Raise planner temperature by F when a scenario is stuck on one"
    echo "                                   strategy for >=15 attempts (0.0 = off, default). Task 9: ship only"
    echo "                                   if the no-fallback baseline shows >10% planner_stuck."
    echo "  --cooperative-seeding           Seed fallback from the highest-cooperation near-miss (default ON)."
    echo "  --no-cooperative-seeding        Disable cooperative seeding (A/B against the score-only selector)."
    echo "  --cooperative-n N              Best-of-N round-1 variant cap on cooperative seeds (8..12, default off=8)."
    echo "                                   Pass 12 to enable BoN scaling (arXiv:2412.03556)."
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds) NUM_ROUNDS="$2"; shift 2 ;;
        --planner-path) PLANNER_PATH="$2"; shift 2 ;;
        --generator-path) GENERATOR_PATH="$2"; shift 2 ;;
        --base-generator-path) BASE_GENERATOR_PATH="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --dataset-size) DATASET_SIZE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --benchmark-name) BENCHMARK_NAME="$2"; shift 2 ;;
        --benchmark-name=*) BENCHMARK_NAME="${1#*=}"; shift ;;
        --victim-model-id) VICTIM_MODEL_ID="$2"; shift 2 ;;
        --start-idx) START_IDX="$2"; shift 2 ;;
        --attempts|--max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
        --trust-remote-code) TRUST_REMOTE_CODE=1; shift ;;
        --tokenizer-mode) TOKENIZER_MODE="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --shared-gpu-memory-utilization) SHARED_GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --victim-max-model-len) VICTIM_MAX_MODEL_LEN="$2"; shift 2 ;;
        --victim-quantization) VICTIM_QUANTIZATION="$2"; shift 2 ;;
        --mutation-fallback|--enable-mutation-fallback)
            MUTATION_FALLBACK=1
            shift
            ;;
        --max-fallback-rounds) MAX_FALLBACK_ROUNDS="$2"; shift 2 ;;
        --max-fallback-rounds=*) MAX_FALLBACK_ROUNDS="${1#*=}"; shift ;;
        --planner-temp-escalation) PLANNER_TEMP_ESCALATION="$2"; shift 2 ;;
        --planner-temp-escalation=*) PLANNER_TEMP_ESCALATION="${1#*=}"; shift ;;
        --cooperative-seeding) COOPERATIVE_SEEDING="yes"; shift ;;
        --no-cooperative-seeding) COOPERATIVE_SEEDING="no"; shift ;;
        --cooperative-n) COOPERATIVE_N="$2"; shift 2 ;;
        --cooperative-n=*) COOPERATIVE_N="${1#*=}"; shift ;;
        --seed) SEED="$2"; shift 2 ;;
        --seed=*) SEED="${1#*=}"; shift ;;
        --help|-h) usage ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$OUTPUT_DIR" ]; then
    # Change 3: two-level benchmark layout — group runs under a benchmark name
    # so results/benchmarks/{name}/{timestamp}_4g/ keeps repeats together. If
    # --benchmark-name is unset, fall back to the legacy flat default.
    TS="$(date +%F_%H-%M-%S)_4g"
    if [ -n "$BENCHMARK_NAME" ]; then
        OUTPUT_DIR="results/benchmarks/${BENCHMARK_NAME}/${TS}"
    else
        OUTPUT_DIR="results/benchmarks/batched_${NUM_ROUNDS}r_4gpu/${TS}"
    fi
fi

# Project root (resolved relative to script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Offline mode (models pre-downloaded on HPC)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# vLLM engine version (V0 is required by this runtime)
export VLLM_USE_V1=0

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
if [ -n "$DATASET_PATH" ]; then
    echo "Dataset      : $DATASET_PATH"
fi
echo "Dataset Size : $DATASET_SIZE"
echo "Victim Model : $VICTIM_MODEL_ID"
if [ -n "$START_IDX" ]; then
    echo "Start Idx    : $START_IDX"
fi
echo "Seed         : $SEED"
echo "Max Attempts : $MAX_ATTEMPTS"
echo "Planner Temp Escalation : $PLANNER_TEMP_ESCALATION (0.0 = off)"
# Cooperative seeding: blank = runtime default ON; print the effective state.
if [ -z "$COOPERATIVE_SEEDING" ]; then
    echo "Cooperative Seeding   : ON (runtime default)"
elif [ "$COOPERATIVE_SEEDING" = "no" ]; then
    echo "Cooperative Seeding   : OFF (--no-cooperative-seeding)"
else
    echo "Cooperative Seeding   : ON"
fi
if [ -n "$COOPERATIVE_N" ]; then
    echo "Cooperative N (BoN)   : $COOPERATIVE_N"
else
    echo "Cooperative N (BoN)   : off (runtime default 8)"
fi
if [ "$TRUST_REMOTE_CODE" -eq 1 ]; then
    echo "Trust Remote : yes"
fi
echo "Tokenizer Mode      : $TOKENIZER_MODE"
echo "Victim GPU Memory   : $GPU_MEMORY_UTILIZATION"
echo "Shared GPU Memory   : $SHARED_GPU_MEMORY_UTILIZATION"
echo "Victim Max Model Len: $VICTIM_MAX_MODEL_LEN"
if [ -n "$VICTIM_QUANTIZATION" ]; then
    echo "Victim Quantization : $VICTIM_QUANTIZATION"
fi
echo "Output Dir          : $OUTPUT_DIR"
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

    WORKER_EXTRA_ARGS=""
    if [ "${MUTATION_FALLBACK:-0}" = "1" ]; then
        WORKER_EXTRA_ARGS="--enable-mutation-fallback"
        export AUTORED_MUTATION_FALLBACK=1
        # Forward adaptive-round-2 setting when fallback is enabled.
        if [ -n "${MAX_FALLBACK_ROUNDS:-}" ]; then
            WORKER_EXTRA_ARGS="$WORKER_EXTRA_ARGS --max-fallback-rounds ${MAX_FALLBACK_ROUNDS}"
        fi
    fi
    # Planner temp escalation is a core-loop feature, independent of fallback;
    # forward whenever it's set to a non-zero value (0.0 = off).
    if [ -n "${PLANNER_TEMP_ESCALATION:-}" ] && [ "${PLANNER_TEMP_ESCALATION}" != "0.0" ]; then
        WORKER_EXTRA_ARGS="$WORKER_EXTRA_ARGS --planner-temp-escalation ${PLANNER_TEMP_ESCALATION}"
    fi
    # Task 3/5 (model-agnostic v2): cooperative seeding + BoN N cap. These
    # are fallback-only levers; forward whenever fallback is enabled. The
    # runtime defaults cooperative seeding ON, so only forward when the user
    # explicitly sets it (enable/disable) or sets --cooperative-n.
    if [ "${MUTATION_FALLBACK:-0}" = "1" ]; then
        if [ "$COOPERATIVE_SEEDING" = "no" ]; then
            WORKER_EXTRA_ARGS="$WORKER_EXTRA_ARGS --no-cooperative-seeding"
        elif [ "$COOPERATIVE_SEEDING" = "yes" ]; then
            WORKER_EXTRA_ARGS="$WORKER_EXTRA_ARGS --cooperative-seeding"
        fi
        if [ -n "$COOPERATIVE_N" ]; then
            WORKER_EXTRA_ARGS="$WORKER_EXTRA_ARGS --cooperative-n ${COOPERATIVE_N}"
        fi
    fi

    # Launch worker on specific GPU — use env to ensure CUDA_VISIBLE_DEVICES
    # is set in the process, not just the shell variable
    env CUDA_VISIBLE_DEVICES=$GPU_ID python experiment/llama_3_8b_vllm.py \
        --mode benchmark \
        --rounds "$NUM_ROUNDS" \
        --dataset-size "$DATASET_SIZE" \
        --attempts "$MAX_ATTEMPTS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --shared-gpu-memory-utilization "$SHARED_GPU_MEMORY_UTILIZATION" \
        --victim-max-model-len "$VICTIM_MAX_MODEL_LEN" \
        --benchmark-output "$WORKER_OUTPUT" \
        --worker-id "$WORKER_ID" \
        --num-workers "$NUM_GPUS" \
        --planner-path "$PLANNER_PATH" \
        --generator-path "$GENERATOR_PATH" \
        --victim-model-id "$VICTIM_MODEL_ID" \
        $( [ "$TRUST_REMOTE_CODE" -eq 1 ] && echo "--trust-remote-code" ) \
        --tokenizer-mode "$TOKENIZER_MODE" \
        $( [ -n "$VICTIM_QUANTIZATION" ] && echo "--victim-quantization $VICTIM_QUANTIZATION" ) \
        $( [ -n "$BASE_GENERATOR_PATH" ] && echo "--base-generator-path $BASE_GENERATOR_PATH" ) \
        $( [ -n "$DATASET_PATH" ] && echo "--dataset-path $DATASET_PATH" ) \
        $( [ -n "$START_IDX" ] && echo "--start-idx $START_IDX" ) \
        --seed "$SEED" \
        $WORKER_EXTRA_ARGS \
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

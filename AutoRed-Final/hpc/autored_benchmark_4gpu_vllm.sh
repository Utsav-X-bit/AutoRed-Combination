#!/bin/bash
#SBATCH --job-name=AutoRed_4GPU
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:4
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/benchmark_4gpu_%j.out
#SBATCH --error=logs/benchmark_4gpu_%j.err
#SBATCH --partition=airawatp

# =============================================================================
# AutoRed Batched 4-GPU Benchmark (vLLM)
#
# Launches one vLLM worker per GPU — each sharding the dataset — then merges
# the per-worker summaries into one JSON. All tuning flags are --flag value
# pairs forwarded to the worker (experiment/llama_3_8b_vllm.py).
#
# USAGE
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./hpc/autored_benchmark_4gpu_vllm.sh [OPTIONS]
#
# OPTIONS (all optional; defaults shown)
#   --rounds N                   Benchmark rounds (default 1000)
#   --planner-path PATH          Planner checkpoint
#                                (default experiment/results/planner_sft_v2_contract_anchor/checkpoint-27)
#   --generator-path PATH        Generator checkpoint (default experiment/results/generator_sft_v2)
#   --base-generator-path PATH   Base model id/path for LoRA
#                                (default Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2)
#   --victim-model-id ID         HF id of the target/victim LLM
#                                (default meta-llama/Meta-Llama-3-8B-Instruct)
#   --dataset-path PATH          Defense dataset jsonl
#   --dataset-size N             Scenarios to sample (default 1000)
#   --attempts N                 Max attack attempts per scenario (default 20)
#   --gpu-memory-utilization F   vLLM GPU mem fraction (default 0.43)
#   --start-idx N                Zero-based start index (ordered slice);
#                                omit for random sampling
#   --seed N                     Seed for sampling / mutation fallback (default 7)
#   --mutation-fallback          Enable JailGuard mutation fallback
#                                (forwarded to the worker as --enable-mutation-fallback)
#   --max-fallback-rounds N      Mutation fallback rounds (default 2)
#   --cooperative-seeding        Seed fallback from highest-cooperation near-miss
#                                (default ON; use --no-cooperative-seeding to A/B)
#   --no-cooperative-seeding     Disable cooperative seeding
#   --cooperative-n N            Best-of-N cap on cooperative seeds (8..12)
#   --planner-temp-escalation F  Raise planner temp to F when stuck (0.0 = off)
#   --output-dir DIR             Results characteristics dir
#                                (default results/benchmark/batched_<rounds>r_<gpus>gpu)
#   --num-gpus N                 Workers to launch (default 4)
#   --help, -h                   Show this help and exit
#
# ENV
#   AUTORED_PROJECT_ROOT   Override the project root (default: this script's
#                          parent dir — portable across hosts).
#   VLLM_USE_V1            vLLM engine selector (default 0, off — matches the
#                          curated invocation).
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Resolve the project root from the script's own location (portable across
# hosts), NOT a hardcoded absolute path. A hardcoded path pointed at a stale
# older copy that lacked experiment/results_layout.py, which made the
# results-layout import fail. Override with AUTORED_PROJECT_ROOT if needed.
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${AUTORED_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

# Resolve a Python 3 interpreter to run the worker and the results_layout
# import. Prefer the project venv (it pins all deps, including vllm); fall
# back to python3. NEVER use bare `python` — on the HPC that is 2.7.8, which
# cannot import the experiment.* namespace package (no __init__.py), causing
# the ModuleNotFoundError that previously aborted every worker. Activating
# the venv also puts this interpreter on PATH for child processes.
PYTHON_BIN=""
if [[ -f "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
  if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
  fi
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "error: no python3 found (venv missing and 'python3' not on PATH)." >&2
  echo "       bare 'python' is 2.7 on this host and cannot run the worker." >&2
  exit 1
fi
export PYTHON_BIN

# Offline mode — models are pre-downloaded on the HPC.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
# vLLM V1 engine off (matches the curated roughText.sh invocation).
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

# =============================================================================
# Defaults (kept in sync with experiment/llama_3_8b_vllm.py where the worker
# owns the flag; the launcher only forwards them).
# =============================================================================
NUM_ROUNDS=1000
PLANNER_PATH="experiment/results/planner_sft_v2_contract_anchor/checkpoint-27"
GENERATOR_PATH="experiment/results/generator_sft_v2"
BASE_GENERATOR_PATH="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
VICTIM_MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_PATH="data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl"
DATASET_SIZE=1000
ATTEMPTS=20
GPU_MEMORY_UTILIZATION=0.43
START_IDX=""
SEED=7
MUTATION_FALLBACK=0
MAX_FALLBACK_ROUNDS=2
COOPERATIVE_SEEDING=1
COOPERATIVE_N=""
PLANNER_TEMP_ESCALATION=0.0
OUTPUT_DIR=""
NUM_GPUS=4

print_help() {
  cat << 'HELP'
AutoRed Batched 4-GPU Benchmark (vLLM)

Launches one vLLM worker per GPU — each sharding the dataset — then merges
the per-worker summaries into one JSON. All tuning flags are --flag value
pairs forwarded to the worker (experiment/llama_3_8b_vllm.py).

USAGE
  CUDA_VISIBLE_DEVICES=0,1,2,3 ./hpc/autored_benchmark_4gpu_vllm.sh [OPTIONS]

OPTIONS (all optional; defaults shown)
  --rounds N                   Benchmark rounds (default 1000)
  --planner-path PATH          Planner checkpoint
                               (default experiment/results/planner_sft_v2_contract_anchor/checkpoint-27)
  --generator-path PATH        Generator checkpoint (default experiment/results/generator_sft_v2)
  --base-generator-path PATH   Base model id/path for LoRA
                               (default Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2)
  --victim-model-id ID         HF id of the target/victim LLM
                               (default meta-llama/Meta-Llama-3-8B-Instruct)
  --dataset-path PATH          Defense dataset jsonl
  --dataset-size N             Scenarios to sample (default 1000)
  --attempts N                 Max attack attempts per scenario (default 20)
  --gpu-memory-utilization F   vLLM GPU mem fraction (default 0.43)
  --start-idx N                Zero-based start index (ordered slice);
                               omit for random sampling
  --seed N                     Seed for sampling / mutation fallback (default 7)
  --mutation-fallback          Enable JailGuard mutation fallback
                               (forwarded to the worker as --enable-mutation-fallback)
  --max-fallback-rounds N      Mutation fallback rounds (default 2)
  --cooperative-seeding        Seed fallback from highest-cooperation near-miss
                               (default ON; use --no-cooperative-seeding to A/B)
  --no-cooperative-seeding     Disable cooperative seeding
  --cooperative-n N            Best-of-N cap on cooperative seeds (8..12)
  --planner-temp-escalation F  Raise planner temp to F when stuck (0.0 = off)
  --output-dir DIR             Results characteristics dir
                               (default results/benchmark/batched_<rounds>r_<gpus>gpu)
  --num-gpus N                 Workers to launch (default 4)
  --help, -h                   Show this help and exit

ENV
  AUTORED_PROJECT_ROOT   Override the project root (default: this script's
                         parent dir — portable across hosts).
  VLLM_USE_V1            vLLM engine selector (default 0, off — matches the
                          curated invocation).
HELP
}

# -----------------------------------------------------------------------------
# Parse --flag value pairs.
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rounds)                  NUM_ROUNDS="$2"; shift 2 ;;
    --planner-path)            PLANNER_PATH="$2"; shift 2 ;;
    --generator-path)          GENERATOR_PATH="$2"; shift 2 ;;
    --base-generator-path)     BASE_GENERATOR_PATH="$2"; shift 2 ;;
    --victim-model-id)         VICTIM_MODEL_ID="$2"; shift 2 ;;
    --dataset-path)            DATASET_PATH="$2"; shift 2 ;;
    --dataset-size)            DATASET_SIZE="$2"; shift 2 ;;
    --attempts)                ATTEMPTS="$2"; shift 2 ;;
    --gpu-memory-utilization)  GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --start-idx)               START_IDX="$2"; shift 2 ;;
    --seed)                    SEED="$2"; shift 2 ;;
    --mutation-fallback)       MUTATION_FALLBACK=1; shift ;;
    --max-fallback-rounds)     MAX_FALLBACK_ROUNDS="$2"; shift 2 ;;
    --cooperative-seeding)     COOPERATIVE_SEEDING=1; shift ;;
    --no-cooperative-seeding)  COOPERATIVE_SEEDING=0; shift ;;
    --cooperative-n)           COOPERATIVE_N="$2"; shift 2 ;;
    --planner-temp-escalation) PLANNER_TEMP_ESCALATION="$2"; shift 2 ;;
    --output-dir)              OUTPUT_DIR="$2"; shift 2 ;;
    --num-gpus)                NUM_GPUS="$2"; shift 2 ;;
    --help|-h)                 print_help; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; echo "try --help" >&2; exit 2 ;;
  esac
done

# Default output dir if not given.
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="results/benchmark/batched_${NUM_ROUNDS}r_${NUM_GPUS}gpu"
fi

# Mutation-fallback env. The worker also sets this from
# --enable-mutation-fallback, but export here so any pre-argparse code sees it.
if [[ "$MUTATION_FALLBACK" -eq 1 ]]; then
  export AUTORED_MUTATION_FALLBACK=1
fi

# -----------------------------------------------------------------------------
# Resolve the logs directory via the shared results_layout module so the
# launcher's log paths EXACTLY match where the worker writes worker_*.json
# (both call runs_root() with the same output-dir + victim-model-id). Passing
# data via argv (quoted heredoc) is robust to paths containing quotes/slashes.
# -----------------------------------------------------------------------------
if ! LOGS_DIR=$("$PYTHON_BIN" - "$OUTPUT_DIR" "$VICTIM_MODEL_ID" <<'PY'
import sys
from experiment.results_layout import resolve_model_id, parse_output_dir, runs_root
output_dir, victim_model_id = sys.argv[1], sys.argv[2]
r = runs_root(
    output_dir,
    "benchmark",
    resolve_model_id(victim_model_id),
    parse_output_dir(output_dir, "benchmark")[1],
)
print(r / "logs")
PY
); then
  echo "error: failed to resolve logs dir via experiment.results_layout." >&2
  echo "       ensure the repo is up to date (git pull) and cwd is the project root:" >&2
  echo "         $PROJECT_ROOT" >&2
  echo "       (experiment/results_layout.py must exist and be importable)" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "logs"

# =============================================================================
# Config summary
# =============================================================================
echo "============================================="
echo "AutoRed Batched ${NUM_GPUS}-GPU Benchmark"
echo "============================================="
echo "Project root : $PROJECT_ROOT"
echo "Total Rounds : $NUM_ROUNDS"
echo "GPUs         : $NUM_GPUS"
echo "Planner      : $PLANNER_PATH"
echo "Generator    : $GENERATOR_PATH"
echo "Base Model   : $BASE_GENERATOR_PATH"
echo "Victim       : $VICTIM_MODEL_ID"
echo "Dataset      : $DATASET_PATH (size $DATASET_SIZE)"
echo "Attempts     : $ATTEMPTS"
echo "GPU mem util : $GPU_MEMORY_UTILIZATION"
echo "Start idx    : ${START_IDX:-<random sampling>}"
echo "Seed         : $SEED"
if [[ "$MUTATION_FALLBACK" -eq 1 ]]; then
  echo "Mutation fb  : on (max-rounds $MAX_FALLBACK_ROUNDS)"
else
  echo "Mutation fb  : off"
fi
if [[ "$COOPERATIVE_SEEDING" -eq 1 ]]; then
  echo "Coop seed    : on (n=${COOPERATIVE_N:-default})"
else
  echo "Coop seed    : off"
fi
echo "Planner temp : $PLANNER_TEMP_ESCALATION"
echo "Output Dir   : $OUTPUT_DIR"
echo "Logs Dir     : $LOGS_DIR"
echo "============================================="

# =============================================================================
# Launch Workers
# =============================================================================
PIDS=()
for WORKER_ID in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=$WORKER_ID
    WORKER_OUTPUT="$OUTPUT_DIR/worker_${WORKER_ID}.json"
    WORKER_LOG="$LOGS_DIR/worker_${WORKER_ID}.log"

    echo ""
    echo "[LAUNCH] Worker $WORKER_ID on GPU $GPU_ID (Processing 16 scenarios at a time)"
    echo "         Output: $WORKER_OUTPUT"
    echo "         Log:    $WORKER_LOG"

    # Build the worker arg list. Required-ish flags first, then conditionals.
    WORKER_ARGS=(
        --mode benchmark
        --rounds "$NUM_ROUNDS"
        --dataset-size "$DATASET_SIZE"
        --benchmark-output "$WORKER_OUTPUT"
        --output-dir "$OUTPUT_DIR"
        --victim-model-id "$VICTIM_MODEL_ID"
        --worker-id "$WORKER_ID"
        --num-workers "$NUM_GPUS"
        --planner-path "$PLANNER_PATH"
        --generator-path "$GENERATOR_PATH"
        --attempts "$ATTEMPTS"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --seed "$SEED"
        --max-fallback-rounds "$MAX_FALLBACK_ROUNDS"
    )
    if [[ -n "$BASE_GENERATOR_PATH" ]]; then
        WORKER_ARGS+=(--base-generator-path "$BASE_GENERATOR_PATH")
    fi
    if [[ -n "$DATASET_PATH" ]]; then
        WORKER_ARGS+=(--dataset-path "$DATASET_PATH")
    fi
    if [[ -n "$START_IDX" ]]; then
        WORKER_ARGS+=(--start-idx "$START_IDX")
    fi
    if [[ "$MUTATION_FALLBACK" -eq 1 ]]; then
        WORKER_ARGS+=(--enable-mutation-fallback)
    fi
    if [[ "$COOPERATIVE_SEEDING" -eq 1 ]]; then
        WORKER_ARGS+=(--cooperative-seeding)
    else
        WORKER_ARGS+=(--no-cooperative-seeding)
    fi
    if [[ -n "$COOPERATIVE_N" ]]; then
        WORKER_ARGS+=(--cooperative-n "$COOPERATIVE_N")
    fi
    if [[ "$PLANNER_TEMP_ESCALATION" != "0.0" ]]; then
        WORKER_ARGS+=(--planner-temp-escalation "$PLANNER_TEMP_ESCALATION")
    fi

    # Launch worker on a specific GPU. Use env so CUDA_VISIBLE_DEVICES is set
    # in the process, not just the shell variable.
    env CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" experiment/llama_3_8b_vllm.py \
        "${WORKER_ARGS[@]}" \
        > "$WORKER_LOG" 2>&1 &

    PIDS+=($!)

    # Stagger workers by 5s to avoid NFS I/O contention during model loading.
    if [[ $WORKER_ID -lt $((NUM_GPUS - 1)) ]]; then
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

if [[ $FAILED -ne 0 ]]; then
    echo ""
    echo "[ERROR] One or more workers failed. Check $LOGS_DIR/worker_*.log for details."
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

if "$PYTHON_BIN" scripts/merge_benchmarks.py \
    --output "$LOGS_DIR" \
    --worker-results "$LOGS_DIR"/worker_*.json
then
    echo ""
    echo "============================================="
    echo "Benchmark complete!"
    echo "Merged results: $LOGS_DIR/merged_summary.json"
    echo "============================================="
else
    echo "[ERROR] Merge script failed!" >&2
    exit 1
fi

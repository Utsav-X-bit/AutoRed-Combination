#!/bin/bash
# =============================================================================
# AutoRed single-GPU smoke test (offline / no internet).
#
# Purpose: validate the full measurement + fallback pipeline end-to-end on one
# GPU in a few minutes, WITHOUT touching the 4-GPU production wrapper.
#
# It exercises (so all of Tasks 1-10 run, not just model loading):
#   - run_benchmark with the enriched per-scenario results (success_path,
#     failure_mode, best_strategy, fallback_triggered)  -> Task 3
#   - merge_benchmarks.py preserving mutation_fallback_*, failure_mode_stats,
#     gt_leak_rate, extractor_recovery_rate               -> Task 4
#   - strategy-aware mutator pool + adaptive round 2      -> Tasks 5-6
#   - --seed reproducibility, planner anti-repeat          -> Tasks 7-8
#
# It does NOT validate Task 9 (temp escalation) — that's opt-in (default off)
# and gated on the full-run diagnostic. It does NOT require 4 GPUs.
#
# Usage (from AutoRed-Final/, on a compute node with >=1 GPU):
#   CUDA_VISIBLE_DEVICES=0 ./hpc/smoke_test.sh
#
# Override the scenario slice with the same --seed/--start-idx you'll use in
# the real run so the smoke shares that scenario set:
#   CUDA_VISIBLE_DEVICES=0 ./hpc/smoke_test.sh --seed 7 --start-idx 1000
# =============================================================================

set -euo pipefail

# ---- offline env (matches the production wrapper) --------------------------
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export VLLM_USE_V1=0

# ---- defaults (mirror hpc/autored_benchmark_4gpu_vllm.sh) -------------------
PLANNER_PATH="experiment/results/planner_sft_v2_contract_anchor/checkpoint-27"
GENERATOR_PATH="experiment/results/generator_sft_v2"
BASE_GENERATOR_PATH="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
DATASET_PATH="data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl"
VICTIM_MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
MAX_ATTEMPTS=20
GPU_MEMORY_UTILIZATION=0.43
SHARED_GPU_MEMORY_UTILIZATION=0.55
VICTIM_MAX_MODEL_LEN=2048
TOKENIZER_MODE="auto"
SEED=7
START_IDX=1000
SMOKE_ROUNDS=8          # tiny: ~8 scenarios, one GPU, a few minutes
OUTPUT_ROOT="results/benchmarks/smoke"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  --seed N            Dataset + fallback RNG seed (default 7)"
    echo "  --start-idx N       Dataset start index (default 1000)"
    echo "  --rounds N          Smoke rounds, single GPU (default 8)"
    echo "  --output-dir DIR    Output root (default results/benchmarks/smoke)"
    echo "  -h, --help"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        --start-idx) START_IDX="$2"; shift 2 ;;
        --rounds) SMOKE_ROUNDS="$2"; shift 2 ;;
        --output-dir) OUTPUT_ROOT="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

# ---- project setup ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"
mkdir -p "$OUTPUT_ROOT" logs

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="$OUTPUT_ROOT/smoke_base_${SEED}_${STAMP}"
FB_DIR="$OUTPUT_ROOT/smoke_fb_${SEED}_${STAMP}"
mkdir -p "$BASE_DIR" "$FB_DIR"

echo "============================================="
echo "AutoRed SMOKE TEST (single GPU)"
echo "============================================="
echo "GPU(s) visible : ${CUDA_VISIBLE_DEVICES:-all}"
echo "Rounds         : $SMOKE_ROUNDS (one worker)"
echo "Seed / StartIdx: $SEED / $START_IDX"
echo "Victim         : $VICTIM_MODEL_ID"
echo "Base output    : $BASE_DIR"
echo "Fallback output: $FB_DIR"
echo "============================================="

# Common args shared by both runs. Both use --num-workers 1 --worker-id 0 so a
# single process handles all rounds and writes a single worker_0.json.
COMMON_ARGS=(
    --mode benchmark
    --rounds "$SMOKE_ROUNDS"
    --start-idx "$START_IDX"
    --seed "$SEED"
    --dataset-path "$DATASET_PATH"
    --attempts "$MAX_ATTEMPTS"
    --victim-model-id "$VICTIM_MODEL_ID"
    --planner-path "$PLANNER_PATH"
    --generator-path "$GENERATOR_PATH"
    --base-generator-path "$BASE_GENERATOR_PATH"
    --tokenizer-mode "$TOKENIZER_MODE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --shared-gpu-memory-utilization "$SHARED_GPU_MEMORY_UTILIZATION"
    --victim-max-model-len "$VICTIM_MAX_MODEL_LEN"
    --num-workers 1
    --worker-id 0
)

# -----------------------------------------------------------------------------
# Run 1: BASELINE (no fallback)
# -----------------------------------------------------------------------------
echo ""
echo "========== [1/2] BASELINE run (no fallback) =========="
python experiment/llama_3_8b_vllm.py \
    "${COMMON_ARGS[@]}" \
    --benchmark-output "$BASE_DIR/worker_0.json" \
    > "logs/smoke_base_${STAMP}.log" 2>&1
echo "  done -> $BASE_DIR/worker_0.json"

# -----------------------------------------------------------------------------
# Run 2: FALLBACK + adaptive round 2
# -----------------------------------------------------------------------------
echo ""
echo "========== [2/2] FALLBACK run (--enable-mutation-fallback --max-fallback-rounds 2) =========="
export AUTORED_MUTATION_FALLBACK=1
python experiment/llama_3_8b_vllm.py \
    "${COMMON_ARGS[@]}" \
    --enable-mutation-fallback \
    --max-fallback-rounds 2 \
    --benchmark-output "$FB_DIR/worker_0.json" \
    > "logs/smoke_fb_${STAMP}.log" 2>&1
unset AUTORED_MUTATION_FALLBACK
echo "  done -> $FB_DIR/worker_0.json"

# -----------------------------------------------------------------------------
# Merge each (single-worker merge still exercises the merge logic)
# -----------------------------------------------------------------------------
echo ""
echo "========== Merging =========="
python scripts/merge_benchmarks.py \
    --output "$BASE_DIR/merged_summary.json" \
    --worker-results "$BASE_DIR"/worker_*.json
python scripts/merge_benchmarks.py \
    --output "$FB_DIR/merged_summary.json" \
    --worker-results "$FB_DIR"/worker_*.json

# -----------------------------------------------------------------------------
# Auto-verify the enriched output (PASS/FAIL)
# -----------------------------------------------------------------------------
echo ""
echo "========== SMOKE VERIFICATION =========="
python - "$BASE_DIR" "$FB_DIR" <<'PY'
import json, sys
base_dir, fb_dir = sys.argv[1], sys.argv[2]
base = json.load(open(f"{base_dir}/merged_summary.json"))
fb   = json.load(open(f"{fb_dir}/merged_summary.json"))

checks = []
def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail))

# --- per-scenario enrichment (Task 3) ---
for tag, d in (("base", base), ("fb", fb)):
    r0 = d["results"][0] if d.get("results") else {}
    required = {"success_path", "fallback_triggered", "best_strategy", "failure_mode"}
    present = required.issubset(r0.keys())
    check(f"{tag}: result[0] has enriched fields", present,
          f"missing={required - set(r0.keys())}")

# --- merge preserves fallback + failure-mode stats (Task 4) ---
for tag, d in (("base", base), ("fb", fb)):
    for k in ("mutation_fallback_triggered", "mutation_fallback_successes",
              "failure_mode_stats", "gt_leak_rate", "extractor_recovery_rate"):
        check(f"{tag}: merge has '{k}'", k in d, f"keys={sorted(d.keys())[:12]}")

# --- paired comparability (Task 7): same scenario set across the two runs ---
base_codes = [r["access_code"] for r in base["results"]]
fb_codes   = [r["access_code"] for r in fb["results"]]
check("paired: same scenario set (seed/start-idx match)", base_codes == fb_codes,
      f"base={base_codes} fb={fb_codes}")

# --- fallback triggered: informational, NOT a hard invariant ---
# fb-triggered ⊆ base-failures only holds when the two runs are bit-identical.
# We seed the dataset sampler (Task 7) and the fallback RNG, but NOT vLLM
# inference sampling (generator runs at temp 0.7, victim also samples), so a
# scenario can legitimately fail all regular attempts in the fb run (-> trigger
# fallback) yet succeed in baseline. So this is reported as overlap, not gated.
base_fail = {r["access_code"] for r in base["results"] if not r["success"]}
fb_trig   = {r["access_code"] for r in fb["results"] if r.get("fallback_triggered")}
if fb_trig:
    overlap = fb_trig & base_fail
    print(f"  [INFO] fb-triggered={len(fb_trig)} scenarios; "
          f"{len(overlap)} also failed in baseline "
          f"(overlap {100*len(overlap)/len(fb_trig):.0f}%). Non-overlap is "
          f"expected — vLLM inference is not seeded, so the two runs are not "
          f"bit-identical.")
    # Hard-check internal consistency per ROW (access_code is NOT unique across
    # defenses, so key on the row itself): every row that triggered fallback must
    # have recorded one of the two valid fallback outcomes — won via fallback,
    # or labeled fallback_failed.
    triggered_rows = [r for r in fb["results"] if r.get("fallback_triggered")]
    bad_rows = [
        r for r in triggered_rows
        if r.get("success_path") != "fallback"
        and r.get("failure_mode") != "fallback_failed"
    ]
    check("fb: every triggered scenario has a valid fallback outcome",
          not bad_rows,
          f"bad rows={[(r['access_code'], r.get('success_path'), r.get('failure_mode')) for r in bad_rows]}")
else:
    print("  [INFO] no fallback triggered on this smoke — rerun with larger "
          "--rounds (e.g. 40) if you want to exercise the fallback path.")

# --- report ---
print()
all_ok = True
for name, ok, detail in checks:
    mark = "PASS" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not ok else ""))

print()
print(f"  BASE success_rate = {base.get('success_rate'):.4f}  "
      f"(successes={base.get('total_successes')}/{base.get('total_rounds')})")
print(f"  FB   success_rate = {fb.get('success_rate'):.4f}  "
      f"(successes={fb.get('total_successes')}/{fb.get('total_rounds')})")
print(f"  FB   fallback triggered={fb.get('mutation_fallback_triggered')} "
      f"successes={fb.get('mutation_fallback_successes')}")
print(f"  BASE failure_mode_stats = {base.get('failure_mode_stats')}")
print(f"  FB   failure_mode_stats = {fb.get('failure_mode_stats')}")

print()
print("SMOKE RESULT: " + ("ALL CHECKS PASSED ✅" if all_ok else "SOME CHECKS FAILED ❌"))
sys.exit(0 if all_ok else 1)
PY

# -----------------------------------------------------------------------------
# Did the new code paths actually fire? (grep the fallback log markers)
# -----------------------------------------------------------------------------
echo ""
echo "========== New-code-path markers in fallback log =========="
for pat in "Strategy-aware mutator pool" "ROUND 2" "MUTATION FALLBACK SUCCESS" "MUTATION FALLBACK FAILED"; do
    n=$(grep -c "$pat" "logs/smoke_fb_${STAMP}.log" || true)
    echo "  '$pat': $n occurrence(s)"
done
echo ""
echo "Note: 'ROUND 2' / 'MUTATION FALLBACK SUCCESS' may be 0 on a tiny smoke if no"
echo "scenario was a near-miss. That's expected; the full run will exercise them."
echo "============================================="
echo "Smoke logs: logs/smoke_base_${STAMP}.log  logs/smoke_fb_${STAMP}.log"
echo "============================================="

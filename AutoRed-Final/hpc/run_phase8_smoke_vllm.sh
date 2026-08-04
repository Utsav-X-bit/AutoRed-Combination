#!/bin/bash
# Phase 8 smoke test: single GPU, 10-round probe for Planner -> Generator integration.

set -euo pipefail

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PROJECT_ROOT="/nlsasfs/home/isea/isea38/AutoRed"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"

NUM_ROUNDS=${1:-10}
DATASET_SIZE=${2:-100}
PLANNER_PATH=${3:-"experiment/results/planner_sft_v2_contract_anchor"}
GENERATOR_PATH=${4:-"experiment/results/generator_sft_v2"}
BASE_MODEL_PATH=${5:-"Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"}
OUTPUT_PATH=${6:-"results/benchmarks/integration_test_10r.json"}

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "============================================="
echo "AutoRed Phase 8 Smoke Test"
echo "============================================="
echo "Rounds       : $NUM_ROUNDS"
echo "Dataset size : $DATASET_SIZE"
echo "Planner      : $PLANNER_PATH"
echo "Generator    : $GENERATOR_PATH"
echo "Base Model   : $BASE_MODEL_PATH"
echo "Output       : $OUTPUT_PATH"
echo "============================================="

env CUDA_VISIBLE_DEVICES=0 python experiment/llama_3_8b_vllm.py \
  --mode benchmark \
  --rounds "$NUM_ROUNDS" \
  --dataset-size "$DATASET_SIZE" \
  --planner-path "$PLANNER_PATH" \
  --generator-path "$GENERATOR_PATH" \
  --base-generator-path "$BASE_MODEL_PATH" \
  --benchmark-output "$OUTPUT_PATH"

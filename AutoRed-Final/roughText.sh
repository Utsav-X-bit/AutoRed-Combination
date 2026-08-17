# =============================================================================
# Mistral-7B-Instruct-v0.2 — full-dataset 2-node benchmark
#
# GPU memory budget (A100-SXM4-40GB, 39.38 GiB/GPU, 4 GPUs, Ampere sm_80):
#   victim 0.46 + shared 0.44 = 0.90 total  →  3.94 GiB unreserved aux slab
#     ~1.6 GiB  aux models (judge + predictor + ranker + strategy + RAG)
#     ~1.4 GiB  NLLB-200 fp16 on GPU (AUTORED_TL_DEVICE=gpu)
#     ~0.9 GiB  activation/overflow margin
#
#   These are the LAUNCHER DEFAULTS (hpc/autored_benchmark_4gpu_vllm.sh),
#   inherited automatically — NO --gpu-memory-utilization / --shared-gpu-memory-
#   utilization flags are passed below. See the rationale block at the bottom
#   for why the Llama-3-8B 0.46/0.44 split is still optimal for Mistral-7B.
#
# Two-node split (26,047 scenarios, --dedup-scenarios collapses the pool first):
#   Node-1:  --start-idx 0     --rounds 13024   (rounds 0..13023)
#   Node-2:  --start-idx 13024 --rounds 13023   (rounds 13024..26046)
# Uncomment the block for the node you are running on, re-comment the other.
# =============================================================================

export AUTORED_MUTATION_FALLBACK=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export VLLM_USE_V1=0
#export AUTORED_TL_DEVICE=cpu   # NLLB on GPU by default (launcher TL_DEVICE=gpu)

# ---- Node-1 (rounds 0..13023) ----------------------------------------------
CUDA_VISIBLE_DEVICES=0,1,2,3 ./hpc/autored_benchmark_4gpu_vllm.sh \
  --rounds 13024 \
  --planner-path experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
  --generator-path experiment/results/generator_sft_v2 \
  --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
  --victim-model-id mistralai/Mistral-7B-Instruct-v0.2 \
  --attempts 20 \
  --dataset-path data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl \
  --dataset-size 26047 \
  --mutation-fallback \
  --max-fallback-rounds 2 \
  --output-dir "results/benchmarks/Mistral7B-[0:13024]_MutationFallback-[KB+RAG]_subset-8_$(date +%F_%H-%M-%S)_4g" \
  --start-idx 0 \
  --seed 7 \
  --dedup-scenarios \
  "$@"

# ---- Node-2 (rounds 13024..26046) — comment the Node-1 block, uncomment this -
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./hpc/autored_benchmark_4gpu_vllm.sh \
#   --rounds 13023 \
#   --planner-path experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
#   --generator-path experiment/results/generator_sft_v2 \
#   --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
#   --victim-model-id mistralai/Mistral-7B-Instruct-v0.2 \
#   --attempts 20 \
#   --dataset-path data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl \
#   --dataset-size 26047 \
#   --mutation-fallback \
#   --max-fallback-rounds 2 \
#   --output-dir "results/benchmarks/Mistral7B-[13024:13023]_MutationFallback-[KB+RAG]_subset-8_$(date +%F_%H-%M-%S)_4g" \
#   --start-idx 13024 \
#   --seed 7 \
#   --dedup-scenarios \
#   "$@"

# =============================================================================
# WHY 0.46 / 0.44 IS STILL OPTIMAL FOR MISTRAL-7B (not Llama-3-8B)
#
# Two vLLM engines are co-resident on each GPU:
#   (1) VICTIM   — the model under attack. NOW Mistral-7B-Instruct-v0.2.
#   (2) SHARED   — LoRA planner/generator base = Orenguteng/Llama-3.1-8B-Lexi
#                  -Uncensored-V2.  UNCHANGED.  ~16.41 GiB weights (binding floor).
#
# Weight footprint (bf16):
#   Llama-3-8B-Instruct : ~8.03B params → ~14.96 GiB   (prior victim)
#   Mistral-7B-Instruct : ~7.24B params → ~14.48 GiB   (new victim — SMALLER)
#   Shared LoRA base     : ~8B params    → ~16.41 GiB   (unchanged, the floor)
#
# The 0.90 TOTAL rule is about the AUX SLAB, not the victim size:
#   39.38 × (1 − 0.90) = 3.94 GiB left for judge DistilBERT + predictor + ranker
#   + strategy + RAG (~1.6 GiB) + NLLB fp16 (~1.4 GiB) + activation margin (~0.9).
#   That slab need is IDENTICAL regardless of victim model — the aux models
#   don't change when the victim swaps. 0.90 total was proven safe by the full
#   26,047-sample Llama-3 run that completed cleanly. No reason to move it.
#
# The SHARED engine is the binding floor:
#   shared KV = 39.38 × shared_util − 16.41 GiB.
#   Below shared_util = 0.44, shared KV → ~0 (16.41 GiB weights eat it all) and
#   the planner/generator OOMs. So shared_util = 0.44 is the minimum; it CANNOT
#   drop. That pins the victim ceiling at 0.90 − 0.44 = 0.46.
#
# The VICTIM engine with Mistral is BETTER, not worse:
#   Llama-3  victim KV = 39.38×0.46 − 14.96 = 3.15 GiB fp8
#   Mistral  victim KV = 39.38×0.46 − 14.48 = 3.63 GiB fp8   (+0.48 GiB)
#   Smaller weights → MORE KV headroom at the same 0.46 fraction. No need to
#   raise victim_util (and we can't without breaking the 0.90 total + the
#   shared floor). The victim serves the two highest-volume calls
#   (chat_with_llama_messages_batch + extract_batch with growing multi-turn
#   history), so the extra 0.48 GiB of KV is welcome, not wasted.
#
# All other tuning is victim-model-independent and stays at the proven values:
#   KV_CACHE_DTYPE     = fp8_e5m2   (Ampere-safe; "fp8"/E4M3 needs Hopper/Ada)
#   VICTIM_MAX_NUM_SEQS = 32         (caps oversubscription; fixes preemption storm)
#   SHARED_MAX_NUM_SEQS = 32
#   chunked_prefill     = on         (bounds prefill activation peak)
#   TL_DEVICE           = gpu        (NLLB fp16 in the aux slab; not on CPU)
# Protected params UNCHANGED: attempts=20, max-fallback-rounds=2, batch=50,
#   max_token=200, max_context — none touched.
# =============================================================================

export AUTORED_MUTATION_FALLBACK=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export VLLM_USE_V1=0
#export AUTORED_TL_DEVICE=cpu
CUDA_VISIBLE_DEVICES=0,1,2,3 ./hpc/autored_benchmark_4gpu_vllm.sh \
  --rounds 1000 \
  --planner-path experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
  --generator-path experiment/results/generator_sft_v2 \
  --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
  --victim-model-id meta-llama/Meta-Llama-3-8B-Instruct \
  --attempts 20 \
  --dataset-path data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl \
  --dataset-size 5000 \
  --mutation-fallback \
  --max-fallback-rounds 2 \
  --output-dir "results/benchmarks/Llama3-[10000:1000]_MutationFallback-[KB+RAG]_subset-8_$(date +%F_%H-%M-%S)_4g" \
  --start-idx 10000 \
  --seed 7 \
  --dedup-scenarios \
  "$@"

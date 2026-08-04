# AutoRed-Final

Automated red-teaming / prompt-injection & access-code-extraction runtime for
LLM defenses. Each defense is a CTF-style scenario (opening + closing
instruction block hiding an access code); a **planner** LLM picks a strategy,
a **generator** LLM writes the attack, the **victim** LLM responds, a
DistilBERT **judge** classifies the stop-point, and a multi-layer
**extractor** finds/ranks/verifies candidate secrets. Runs up to 20 attempts
per scenario, optionally backed by a FAISS RAG knowledge base. Results flow
into a FastAPI + WebSocket backend and a React UI for analysis.

## Canonical documentation

| What | Where |
|------|-------|
| **Live system source of truth** | [`docs/current_implementation.md`](docs/current_implementation.md) |
| Operational runbook + env quirks | [`AGENTS.md`](AGENTS.md) |
| Phased implementation plan | [`docs/current_implementation_plan.md`](docs/current_implementation_plan.md) |
| Run commands (experiments, benchmarks, training) | [`AGENTS.md`](AGENTS.md) |

## Quick start (GPU/HPC required for inference)

```bash
pip install -r requirements.txt        # CUDA 12.4, vLLM, PyTorch 2.6
# Frontend:
cd ui && npm install
```

```bash
# Always set VLLM_USE_V1=0 before running the runtime.
VLLM_USE_V1=0 python experiment/llama_3_8b_vllm.py \
  --mode benchmark \
  --rounds 70 \
  --dataset-size 1000 \
  --planner-path experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
  --generator-path experiment/results/generator_sft_v2 \
  --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
  --dataset-path data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl
```

> **Read [`AGENTS.md`](AGENTS.md) before running anything.** It covers the
> required `VLLM_USE_V1=0` flag, GPU-memory tuning, LoRA merge workarounds,
> and the laptop-safe backend-only mode (`AUTORED_LOAD_MODELS=0`).

## Layout

```
experiment/   runtime (llama_3_8b_vllm.py, kb_updater.py, planner_contract.py)
server/       FastAPI + WebSocket backend
worker/       Redis/RQ background workers
ui/           Vite + React + TS + Tailwind frontend
hpc/          SLURM + multi-GPU batch / training launchers
scripts/      training, dataset builders, merge, analysis, isolation tests
schemas/      JSON schemas for run/dataset/attempt artifacts
data/         TensorTrust subsets, KB, RAG indices, datasets (gitignored)
docs/         current_implementation.md + plan
```

## Mutation fallback (combination project)

When all 20 attempts fail, AutoRed can invoke the
[`combination`](../combination) layer to mutate the best near-miss attack
using JailGuard text mutators and retry. Enable with
`--enable-mutation-fallback` or `AUTORED_MUTATION_FALLBACK=1`. Scoring is
**judge-independent** (`fallback_score`); see
[`../docs/02_combination_integration.md`](../docs/02_combination_integration.md)
and [`../combination/docs/05_mutation_fallback_usage.md`](../combination/docs/05_mutation_fallback_usage.md).

## Testing

There is no pytest suite. Isolation smoke tests (GPU + models required):
`python scripts/tests/test_planner_v2.py`, `test_generator_v2.py`,
`test_kb_updater.py`, `test_combined_model.py`, `test_vllm_*_lora.py`.

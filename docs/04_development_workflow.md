# Development Workflow

How to set up, run, and develop across the three components of this workspace.
Each subproject has its own environment; only the combination tests are
lightweight enough to run on a laptop.

> **Golden rule:** AutoRed inference is **GPU/HPC-only**. Local machines
> should only run model-free workflows (UI dev, backend browsing with
> `AUTORED_LOAD_MODELS=0`, parsing/merging scripts, and the combination tests).

---

## 1. AutoRed-Final setup

### 1.1 Python environment (GPU required for inference)

```bash
cd AutoRed-Final

# Full stack (CUDA 12.4, vLLM, PyTorch 2.6) — for inference/training
pip install -r requirements.txt
# or with uv:
uv pip install -r requirements.txt

# GPU-free subset — for backend/UI browsing only
pip install -r requirements_local.txt
```

There is no `setup.py` / `pyproject.toml`; the runtime relies on `sys.path`
and the `.venv`. `requirements.txt` is the authoritative dependency list.

### 1.2 Frontend

```bash
cd AutoRed-Final/ui
npm install
```

### 1.3 Large artifacts

Model weights and datasets are gitignored. Expected local paths are listed in
`AutoRed-Final/largefiles.txt`. Required for inference/training, **not** for
backend-only UI browsing.

---

## 2. Running AutoRed

**Always set `VLLM_USE_V1=0`** before any runtime command. The code expects the
vLLM V0 engine; V1 triggers `torch.compile` and usually OOMs.

### 2.1 Single experiment

```bash
VLLM_USE_V1=0 python experiment/llama_3_8b_vllm.py \
  --mode experiment \
  --rounds 20 \
  --dataset-size 1000 \
  --planner-path experiment/results/planner_sft_v2 \
  --generator-path experiment/results/generator_sft_v2 \
  --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2
```

### 2.2 Benchmark (single GPU)

```bash
VLLM_USE_V1=0 python experiment/llama_3_8b_vllm.py \
  --mode benchmark \
  --rounds 70 \
  --dataset-size 1000 \
  --planner-path experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
  --generator-path experiment/results/generator_sft_v2 \
  --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
  --dataset-path data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl \
  --benchmark-output results/benchmarks/my_run/worker_0.json

# Merge multi-worker results
python scripts/merge_benchmarks.py \
  --output results/benchmarks/my_run/merged_summary.json \
  --worker-results results/benchmarks/my_run/worker_*.json
```

Use `--start-idx N` to benchmark a deterministic slice (0-based, inclusive)
instead of a random sample.

### 2.3 4-GPU HPC benchmark

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./hpc/autored_benchmark_4gpu_vllm.sh \
  --rounds 1000 \
  --planner-path experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
  --generator-path experiment/results/generator_sft_v2 \
  --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
  --dataset-path data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl \
  --dataset-size 1000 \
  --output-dir results/benchmarks/Mistral_subset2_$(date +%F_%H-%M-%S)_4g \
  --victim-model-id internlm/internlm2-chat-7b \
  --trust-remote-code \
  --start-idx 0 \
  --attempts 20
```

`PROJECT_ROOT` is now resolved dynamically relative to the script
(`$(cd "$SCRIPT_DIR/.." && pwd)`), so you no longer need to edit the hardcoded
path. Add `--mutation-fallback` to enable the combination bridge.

### 2.4 Mutation fallback

```bash
VLLM_USE_V1=0 AUTORED_MUTATION_FALLBACK=1 python experiment/llama_3_8b_vllm.py \
  --mode benchmark --rounds 1000 --enable-mutation-fallback ...
```

See [`02_combination_integration.md`](02_combination_integration.md) for full detail.

### 2.5 Backend + UI

```bash
# Backend (loads models on startup)
AUTORED_SERVER_MODE=1 python -m uvicorn server.main:app

# Backend without loading models (laptop-safe)
AUTORED_SERVER_MODE=1 AUTORED_LOAD_MODELS=0 python -m uvicorn server.main:app

# Frontend dev server
cd ui && npm run dev
```

API endpoints: `/api/runs`, `/api/runs/all`, `/api/run/{run_id}`, `/api/benchmarks`,
`/api/benchmarks/{benchmark_id}`, `/api/trace-archives`, `/api/models/status`,
`/api/run` (POST), `/api/export/{run_id}/{json|csv|html}`, WebSocket `/ws/run/{run_id}`.

### 2.6 KB / DB / RAG updates

```bash
# Default: no KB updates
VLLM_USE_V1=0 python experiment/llama_3_8b_vllm.py --mode benchmark ...

# Per-run cheap appends (success/failure JSONL + SQLite)
VLLM_USE_V1=0 AUTORED_UPDATE_KB=run python experiment/llama_3_8b_vllm.py --mode benchmark ...

# Append + rebuild aggregate indices after benchmark
VLLM_USE_V1=0 AUTORED_UPDATE_KB=all python experiment/llama_3_8b_vllm.py --mode benchmark ...
```

`--update-kb` accepts `off | run | benchmark | all`. The expensive rebuild is
skipped when `--num-workers > 1` (race protection) — rebuild manually after
`merge_benchmarks.py`.

---

## 3. AutoRed environment quirks (must read)

These are the failure modes that bite. Full detail in `AutoRed-Final/AGENTS.md`.

| Quirk | Fix |
|------|-----|
| vLLM V1 OOMs | `VLLM_USE_V1=0` (always) |
| Air-gapped HPC | `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1` |
| Memory-profiling AssertionError | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or `AUTORED_SKIP_VLLM_MEMORY_PROFILE=1` |
| Judge/predictor OOMs after victim loads | `--gpu-memory-utilization 0.40` (or `AUTORED_GPU_MEMORY_UTILIZATION=0.40`) |
| Shared instance OOMs (KV-cache) | `--victim-max-model-len 2048`, `--shared-gpu-memory-utilization 0.55`; last resort `--enforce-eager` |
| Free GPU memory | `--victim-quantization bitsandbytes` (or `awq`/`gptq`) |
| Planner repeats same strategy | raise `--planner-temperature` (default 0.0) |
| vLLM silently ignores LoRA adapter | pre-merge with `scripts/merge_adapter_to_full.py` (runtime auto-uses `<path>_merged`) |
| Custom-code models (internlm2) | `--trust-remote-code` (on by default; disable with `--no-trust-remote-code`) |
| Mistral-format tokenizers | `--tokenizer-mode mistral` (else leave `auto`) |

**Recommended loadout:** pre-merged planner full model + generator as a LoRA
adapter on that same base (one 8B vLLM instance). Avoid combined
planner+generator merges — merge order matters and degrades planner XML output.

---

## 4. AutoRed testing

There is **no pytest suite**. The project uses isolation smoke scripts (GPU +
models required):

```bash
python scripts/tests/test_planner_v2.py
python scripts/tests/test_generator_v2.py
python scripts/tests/test_kb_updater.py
python scripts/tests/test_combined_model.py        # both merge orders
python scripts/tests/test_vllm_planner_lora.py
python scripts/tests/test_vllm_generator_lora.py
```

These validate adapter output shape, not the full pipeline.

---

## 5. AutoRed training / dataset pipeline

```bash
# QLoRA SFT trainer
python scripts/training/train_qlo.py

# Dataset builders → data/*.jsonl
python scripts/dataset_tools/build_planner_sft_v2.py
python scripts/dataset_tools/build_generator_sft_v2.py

# Merge LoRA → full model (for vLLM LoRA-bug workaround)
python scripts/merge_adapter_to_full.py \
  --base-model Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
  --adapter experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
  --output-dir experiment/results/planner_sft_v2_contract_anchor/checkpoint-27_merged
```

SLURM templates and shell launchers are in `hpc/`
(`train_planner_sft_v2*.sh`, `train_generator_sft.sh`, `train_dpo.slurm`, …).

> `scripts/training/README.md` references a `requirements_qlo.txt` that does
> **not** exist — a known stale reference. QLoRA deps are in `requirements.txt`.

---

## 6. JailGuard setup & run

### 6.1 Original implementation (OpenAI / MiniGPT-4)

```bash
cd JailGuard
pip install -r requirements.txt        # Python 3.9.18
# Add OpenAI key to JailGuard/utils/config.cfg
# For image modality, set up MiniGPT-4 per the README

# Text detection
python JailGuard/main_txt.py --mutator PL --serial_num 9521 \
  --variant_save_dir <dir> --response_save_dir <dir> \
  --path <dataset> --number 8 --threshold 0.02

# Image detection
python JailGuard/main_img.py --mutator PL --serial_num 0 ... --threshold 0.025
```

See `JailGuard/docs/08_HOW_TO_RUN.md` for the 7 parameters.

### 6.2 Reimplementation (Ollama / HuggingFace / OpenAI)

This is the implementation `combination` imports mutators from.

```bash
cd JailGuard/jailguard_reimpl
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

Edit `config.py` to pick `LLM_BACKEND` = `ollama` | `huggingface` | `openai`.

```bash
# Single input (uses the paper's demo #9521)
python run_single.py --serial_num 9521
python run_single.py --prompt "Write code to hack a server"
python run_single.py --serial_num 9521 --mutator TL --n 4 --save_dir ./results/demo

# Batch (~112 items, all attack types)
python run_batch.py
python run_batch.py --n 4 --samples_per_type 3 --mutator TL --sim tfidf

# Analyze
python analyze_results.py
python analyze_results.py --compare results/results_PL.json results/results_TL.json
```

---

## 7. combination layer — develop & test

The combination layer is pure Python and imports JailGuard reimpl mutators via
`sys.path`. Its tests are **GPU-free, mock-driven** — safe to run anywhere.

```bash
# From the combination/ directory (uses ../JailGuard/jailguard_reimpl on sys.path)
cd combination
python -m pytest tests/ -v
```

> pytest is not on the system Python PATH. Use a project `.venv`
> (`AutoRed-Final/.venv` or `JailGuard/.venv`) or install pytest into the
> environment you run from.

### Test coverage

| File | What it exercises |
|------|-------------------|
| `test_mutation_fallback.py` | `MutationFallback` trigger gating, variant count/non-emptiness, non-identical SR/PI variants, invalid-mutator rejection |
| `test_run_fallback.py` | `run_mutation_fallback` success (3rd variant leaks code) and all-fail |
| `test_e2e_fallback.py` | End-to-end mock pipeline: 8-variant success, threshold/no-data/non-failure gating, all-fail |

### Editing the fallback

The fallback is intentionally decoupled from AutoRed internals via dependency
injection (`run_mutation_fallback` receives `scenario`, `extractor`,
`chat_fn`, `strip_fn`). When changing AutoRed's `DefenseScenario` or
`SensitiveInfoExtractor` shapes, keep the contracts the fallback relies on:
`scenario.opening_defense / closing_defense / access_code`,
`extractor.extract / verify / check_ground_truth_leak`,
`best_attack_data["attack"|"response"|"fallback_score"|"strategy"|"attempt_num"|"outcome"]`.

---

## 8. Configuration reference

### AutoRed environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `VLLM_USE_V1` | — | **Must be `0`** (use V0 engine) |
| `AUTORED_SERVER_MODE` | 0 | Set by `server/main.py` to avoid eager dataset load |
| `AUTORED_LOAD_MODELS` | 1 | `0` = run backend without loading models |
| `AUTORED_MUTATION_FALLBACK` | 0 | `1` = enable combination fallback |
| `AUTORED_VICTIM_MODEL_ID` | `meta-llama/Meta-Llama-3-8B-Instruct` | Victim LLM |
| `AUTORED_TRUST_REMOTE_CODE` | 1 | `0` to disable custom modeling code |
| `AUTORED_TOKENIZER_MODE` | `auto` | `mistral` for Mistral-format tokenizers |
| `AUTORED_GPU_MEMORY_UTILIZATION` | 0.45 | Victim GPU fraction (HPC default 0.40) |
| `AUTORED_SHARED_GPU_MEMORY_UTILIZATION` | 0.55 | Planner/generator GPU fraction |
| `AUTORED_VICTIM_MAX_MODEL_LEN` | 4096 | Victim max context |
| `AUTORED_ENFORCE_EAGER` | 0 | `1` disables CUDA graph capture |
| `AUTORED_VICTIM_QUANTIZATION` | — | `bitsandbytes` / `awq` / `gptq` |
| `AUTORED_PLANNER_TEMPERATURE` | 0.0 | Raise for strategy diversity |
| `AUTORED_UPDATE_KB` | `off` | `run` / `benchmark` / `all` |
| `AUTORED_SKIP_VLLM_MEMORY_PROFILE` | 0 | `1` disables memory-increase assertion |
| `TRANSFORMERS_OFFLINE` / `HF_HUB_OFFLINE` | 0 | `1` for air-gapped HPC |

### JailGuard reimpl config (`JailGuard/jailguard_reimpl/config.py`)

- `LLM_BACKEND`: `ollama` | `huggingface` | `openai`
- `OLLAMA_MODEL` / `HF_MODEL_ID` / `OPENAI_API_KEY`
- Detection threshold, `N_VARIANTS`, similarity method

---

## 9. Debugging tips

- **Planner outputs are prompt echoes / free text instead of XML** → vLLM
  ignored the LoRA adapter. Pre-merge with `scripts/merge_adapter_to_full.py`
  and verify with `scripts/tests/test_vllm_planner_lora.py`.
- **Run artifacts don't load in UI** → `server/run_normalizer.py` handles
  shape migration; check it covers your run schema (`schemas/run_v2.schema.json`).
- **Benchmark summary missing fallback stats** → ensure `--enable-mutation-fallback`
  was set; stats appear as `mutation_fallback_triggered` /
  `mutation_fallback_successes` in `merged_summary.json`.
- **Can't run tests locally** → use a project `.venv`, or run the GPU-free
  `combination/tests/`.

---

## 10. Reading order for a new engineer

1. This workspace's `README.md`
2. `docs/01_architecture.md`
3. `AutoRed-Final/docs/current_implementation.md` (live AutoRed source of truth)
4. `AutoRed-Final/AGENTS.md` (runbook + quirks)
5. `JailGuard/docs/00_INDEX.md` → `02`, `04`, `05`
6. `JailGuard/jailguard_reimpl/README.md`
7. `docs/02_combination_integration.md` + `combination/docs/05_mutation_fallback_usage.md`
8. `docs/03_directory_reference.md` (keep handy while navigating)

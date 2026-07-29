# AutoRed + JailGuard — Unified Red-Teaming & Detection Research Workspace

This repository is a **research workspace** that combines two originally-separate
projects and integrates them into a single offensive/defensive LLM-security
pipeline:

| Component | Role | Direction |
|-----------|------|-----------|
| **AutoRed-Final** | Automated red-teaming / prompt-injection & access-code extraction runtime | Offensive |
| **JailGuard** | Jailbreak / prompt-injection **detection** via input mutation & response-divergence analysis | Defensive |
| **combination** | Glue layer that reuses JailGuard's text mutators as an **offensive prompt fuzzer** when AutoRed exhausts its attempts | Bridge |

The unifying idea: AutoRed attacks a defended victim LLM; JailGuard's
mutation engine — built to *detect* attacks — is repurposed to *fuzz* a failed
attack into near-miss variants that may finally crack a stubborn defense.

> **Note on submodules.** `AutoRed-Final/` and `JailGuard/` are git subprojects
> pinned at specific commits (tree entries mode `160000`). `combination/` is
> tracked directly in this outer repository. Each subproject has its own
> history, README, and docs.

---

## Repository Layout

```
autoredPLUSjailguard/
├── README.md               ← this file (umbrella overview)
├── docs/                   ← unified project documentation (this workspace)
│   ├── 01_architecture.md
│   ├── 02_combination_integration.md
│   ├── 03_directory_reference.md
│   └── 04_development_workflow.md
├── AutoRed-Final/          ← offensive red-teaming runtime (subproject)
│   ├── readme.md           ← (was empty — now a short entry pointer)
│   ├── AGENTS.md           ← operational agent notes (canonical run instructions)
│   ├── docs/               ← subproject-level docs (current_implementation.md, …)
│   ├── experiment/         ← runtime: llama_3_8b_vllm.py, kb_updater.py, …
│   ├── server/             ← FastAPI + WebSocket backend for the UI
│   ├── worker/             ← background run/benchmark workers
│   ├── ui/                 ← Vite + React + Tailwind frontend
│   ├── hpc/                ← SLURM / multi-GPU batch + training scripts
│   ├── scripts/            ← training, dataset builders, merge, analysis, tests
│   ├── schemas/            ← JSON schemas for run/dataset/attempt artifacts
│   ├── data/               ← TensorTrust subsets, KB, RAG indices, datasets
│   ├── models/  pre_trained/  results/  reports/  logs/  tmp/
│   └── requirements.txt / requirements_local.txt / largefiles.txt
├── JailGuard/              ← defensive detection framework (subproject)
│   ├── README.md           ← original project README
│   ├── docs/               ← 00–09 numbered deep-dive documentation
│   ├── JailGuard/          ← original implementation (main_txt.py, main_img.py, utils/)
│   ├── jailguard_reimpl/   ← clean modular text-only reimpl (Ollama / HF / OpenAI)
│   ├── dataset/           ← text & image attack datasets
│   └── misc/  requirements.txt
└── combination/           ← integration layer (tracked in this repo)
    ├── src/mutation_fallback.py   ← MutationFallback + run_mutation_fallback
    ├── tests/                     ← pytest suite (no GPU needed)
    └── docs/                      ← 01–06 analysis, blueprint, usage, audit
```

---

## Quick Orientation

**New here?** Read in this order:

1. This README.
2. [`docs/01_architecture.md`](docs/01_architecture.md) — how the three parts fit.
3. [`docs/02_combination_integration.md`](docs/02_combination_integration.md) — the mutation fallback bridge.
4. `AutoRed-Final/docs/current_implementation.md` — the live AutoRed source of truth.
5. `JailGuard/docs/00_INDEX.md` — the JailGuard documentation index.

---

## What Each Project Does

### AutoRed-Final (offensive)
A security-evaluation runtime that treats each defense as a CTF scenario: an
opening/closing instruction block hides an access code; the attacker must
recover it. A **planner** LLM chooses a strategy (XML plan), a **generator**
LLM writes the attack prompt, the **victim** LLM (Llama-3-8B-Instruct by
default) responds, a DistilBERT **judge** classifies stop-point, and a
multi-layer **extractor** finds/ranks/verifies candidate secrets. Runs up to
20 attempts per scenario, optionally backed by a RAG knowledge base. Results
flow into a FastAPI backend + React UI for analysis.

→ See `AutoRed-Final/AGENTS.md` for canonical run commands and `VLLM_USE_V1=0`
  and other required environment quirks. **GPU/HPC-only for inference.**

### JailGuard (defensive)
A universal jailbreak/prompt-injection **detection** framework for text and
image inputs. Core insight: benign inputs, when slightly mutated, yield
near-identical LLM responses (low divergence); crafted attacks break under
mutation (high divergence). Generates N variants of an input, queries the
target LLM, builds a similarity matrix, computes KL divergence, and flags
inputs whose max divergence exceeds a threshold. Reported 86.14% text /
82.90% image accuracy against 12 baselines.

Two implementations ship here:
- `JailGuard/JailGuard/` — original paper implementation (OpenAI / MiniGPT-4).
- `JailGuard/jailguard_reimpl/` — clean text-only reimplementation supporting
  Ollama, HuggingFace, or OpenAI backends. **This is the one the combination
  layer imports mutators from.**

### combination (bridge)
A small integration layer that reuses JailGuard's structure-preserving text
mutators (`SR` Synonym Replacement, `PI` Punctuation Insertion, `TL`
Translation) as an **offensive fuzzer**. When all 20 AutoRed attempts fail on a
scenario, it takes the highest-scoring failed attack (scored by a
**judge-independent** `fallback_score`), generates 8 mutated variants, re-queries
the victim, and runs AutoRed's extractor on each — recovering an estimated
+2–6% net success rate on borderline defenses.

→ See [`docs/02_combination_integration.md`](docs/02_combination_integration.md)
  and `combination/docs/05_mutation_fallback_usage.md`.

---

## Environment Notes (read before running anything)

- **AutoRed inference is GPU/HPC-only.** Single experiments, benchmarks, and
  any command loading vLLM/CUDA belong on the cluster. Local machines should
  only run model-free workflows (UI dev, backend browsing with
  `AUTORED_LOAD_MODELS=0`, parsing/merging scripts).
- AutoRed requires **`VLLM_USE_V1=0`** and CUDA; see `AutoRed-Final/AGENTS.md`
  for the full quirk list (memory fraction, quantization, LoRA merge caveats).
- JailGuard original needs an OpenAI key; the reimpl can run fully local via
  Ollama.
- The combination tests (`combination/tests/`) are **GPU-free** and use mocks —
  safe to run anywhere with pytest.

---

## Testing

```bash
# combination layer — no GPU, no models, mock-driven
cd combination && python -m pytest tests/ -v

# AutoRed — no pytest suite; uses isolation smoke tests (GPU + models needed)
# see AutoRed-Final/AGENTS.md "Verification / Isolation Tests"
```

---

## Subproject Documentation

Each subproject keeps its own authoritative docs — this workspace's `/docs`
only covers the **unified** view and the integration:

- AutoRed-Final: `AutoRed-Final/AGENTS.md`, `AutoRed-Final/docs/`
- JailGuard: `JailGuard/README.md`, `JailGuard/docs/`, `JailGuard/jailguard_reimpl/README.md`
- combination: `combination/docs/`

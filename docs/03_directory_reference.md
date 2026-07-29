# Directory Reference

Full file-level map of the workspace. Paths are relative to the repository
root (`autoredPLUSjailguard/`). Generated/vendored trees (`.venv/`,
`__pycache__`, `node_modules`, `.git`) are omitted.

```
autoredPLUSjailguard/
├── README.md
├── .gitignore
├── docs/                          ← unified workspace documentation
│   ├── 01_architecture.md
│   ├── 02_combination_integration.md
│   ├── 03_directory_reference.md
│   └── 04_development_workflow.md
│
├── AutoRed-Final/                ← offensive red-teaming runtime (subproject)
│   ├── readme.md                  ← short entry pointer (AGENTS.md is canonical)
│   ├── AGENTS.md                  ← operational runbook + env quirks
│   ├── requirements.txt           ← full CUDA/vLLM dependency list
│   ├── requirements_local.txt     ← GPU-free subset (FastAPI/uvicorn only)
│   ├── uv.lock
│   ├── largefiles.txt             ← expected gitignored model/dataset paths
│   ├── docs/
│   │   ├── current_implementation.md      ← LIVE source of truth
│   │   └── current_implementation_plan.md ← phased plan (historical)
│   ├── experiment/                ← runtime core
│   │   ├── llama_3_8b_vllm.py      ← MAIN runtime entry point
│   │   ├── kb_updater.py           ← KB/DB/RAG post-run updater
│   │   ├── planner_contract.py     ← planner XML contract enforcer
│   │   ├── state_builder.py
│   │   ├── strategy_predictor.pth
│   │   ├── access_code_predictor/  ← DistilBERT access-code-type classifier
│   │   ├── champions/
│   │   └── results/                ← trained LoRA adapters (gitignored)
│   │       ├── planner_sft_v2/
│   │       ├── planner_sft_v2_contract_anchor/checkpoint-27/
│   │       └── generator_sft_v2/
│   ├── server/                    ← FastAPI + WebSocket backend
│   │   ├── main.py                ← app, API endpoints, WebSocket
│   │   ├── models_server.py       ← ServerModelsManager (loads models on GPU)
│   │   ├── experiment_server.py   ← per-scenario runner
│   │   ├── run_normalizer.py      ← normalizes old/new run JSON for UI
│   │   ├── file_manager.py        ← discovers runs & benchmarks on disk
│   │   ├── websocket.py
│   │   └── schemas.py
│   ├── worker/                    ← Redis/RQ background worker scaffolding
│   │   ├── rq_app.py
│   │   ├── experiment_runner.py
│   │   └── models_manager.py
│   ├── ui/                        ← Vite + React + TS + Tailwind frontend
│   │   ├── src/ (App.tsx, pages: RunLoader, InvestigationPage, RunComparison, BenchmarkDashboard)
│   │   ├── package.json, vite.config.ts, tailwind.config.js, tsconfig.json
│   │   └── index.html
│   ├── hpc/                       ← SLURM + shell launchers (39 files)
│   │   ├── autored_benchmark_4gpu_vllm.sh  ← main 4-GPU benchmark wrapper
│   │   ├── run_phase8_smoke_vllm.sh        ← integration smoke test
│   │   ├── train_planner_sft_v2*.sh, train_generator_sft.sh, train_dpo.slurm, ...
│   │   ├── download_bases.py, download_models.py, download_hf_assets.py
│   │   └── train_reward_model.py
│   ├── scripts/
│   │   ├── training/train_qlo.py          ← QLoRA SFT trainer
│   │   ├── dataset_tools/                 ← build_planner_sft_v2.py, build_generator_sft_v2.py
│   │   ├── merge_benchmarks.py             ← merge multi-worker benchmark JSON
│   │   ├── merge_adapter_to_full.py       ← merge LoRA → full model
│   │   ├── analyze_vllm_benchmark.py
│   │   ├── analysis/compare_benchmarks.py
│   │   ├── tests/                         ← isolation smoke tests (GPU)
│   │   │   ├── test_planner_v2.py, test_generator_v2.py
│   │   │   ├── test_combined_model.py, test_kb_updater.py
│   │   │   └── test_vllm_planner_lora.py, test_vllm_generator_lora.py
│   │   ├── pi/, GEMINI.md
│   ├── schemas/                   ← JSON schemas
│   │   ├── run_v2.schema.json, attempt_v2.schema.json
│   │   ├── dataset_entry_v2.schema.json, extractor_candidate_v2.schema.json
│   │   ├── defense_taxonomy_v2.json, strategy_taxonomy_v2.json
│   ├── data/                      ← datasets, KB, RAG (mostly gitignored)
│   │   ├── TensorTrust_subsets/   ← subset_1..9 + manifest.json
│   │   ├── strategy_knowledge_base.json, oracle_rules.json
│   │   ├── rag/success_defenses.index + success_metadata.json
│   │   ├── autored_kb.db, test_kb.db
│   │   └── planner/generator SFT corpora, trajectories, reports
│   ├── models/                    ← small saved classifiers (gitignored)
│   ├── pre_trained/, AR_pre_trained/  ← judge / reward models (gitignored)
│   ├── results/                   ← run traces: results/YYYY-MM-DD/<victim>/HH-MM-SS_µs/
│   │   └── benchmarks/<id>/merged_summary.json
│   ├── results_old/, reports/, logs/, tmp/
│   └── server/ worker/ (see above)
│
├── JailGuard/                     ← defensive detection framework (subproject)
│   ├── README.md                  ← original project README
│   ├── requirements.txt
│   ├── docs/                      ← 00–09 numbered deep-dive
│   │   ├── 00_INDEX.md
│   │   ├── 01_RESEARCH_PAPER_OVERVIEW.md
│   │   ├── 02_SYSTEM_ARCHITECTURE.md
│   │   ├── 03_CODEBASE_STRUCTURE.md
│   │   ├── 04_MUTATORS_EXPLAINED.md
│   │   ├── 05_DETECTION_ALGORITHM.md
│   │   ├── 06_DATASET_EXPLAINED.md
│   │   ├── 07_BASELINES_AND_COMPARISON.md
│   │   ├── 08_HOW_TO_RUN.md
│   │   └── 09_KEY_INSIGHTS_AND_FINDINGS.md
│   ├── JailGuard/                 ← original implementation
│   │   ├── main_txt.py            ← text detection CLI
│   │   ├── main_img.py            ← image detection CLI
│   │   ├── utils/                 ← augmentations.py, baseline_utils.py, similarity.py, ...
│   │   └── demo_case/
│   ├── jailguard_reimpl/          ← clean text-only reimpl (combination imports this)
│   │   ├── README.md
│   │   ├── config.py              ← LLM backend + threshold settings
│   │   ├── mutators.py            ← 9 text mutators + Policy; apply_mutator, AVAILABLE_MUTATORS
│   │   ├── llm_interface.py       ← Ollama / HuggingFace / OpenAI unified interface
│   │   ├── divergence.py          ← similarity matrix + KL divergence
│   │   ├── detector.py            ← 3-step pipeline + DetectionResult
│   │   ├── run_single.py          ← CLI: test one input
│   │   ├── run_batch.py            ← CLI: stratified batch evaluation
│   │   ├── analyze_results.py     ← CLI: metrics + plots
│   │   ├── requirements.txt
│   │   └── results/
│   ├── dataset/                   ← text + image attack datasets
│   │   ├── text/, image/ (dataset.pkl, dataset-key.pkl)
│   │   └── readme.md
│   └── misc/                      ← figures
│
└── combination/                   ← integration layer (tracked in this repo)
    ├── src/
    │   ├── __init__.py
    │   └── mutation_fallback.py    ← MutationFallback, run_mutation_fallback
    ├── tests/
    │   ├── test_mutation_fallback.py  ← unit: gating, variants, validation
    │   ├── test_run_fallback.py       ← run_mutation_fallback success/fail
    │   └── test_e2e_fallback.py        ← end-to-end mock pipeline
    └── docs/
        ├── 01_autored_analysis.md
        ├── 02_jailguard_analysis.md
        ├── 03_combination_blueprint.md       ← design vision (proposed)
        ├── 04_mutation_attack_hypothesis.md   ← feasibility study
        ├── 05_mutation_fallback_usage.md      ← usage guide (implemented)
        ├── 06_best_attack_audit.md            ← judge-independent scoring critique
        └── superpowers/plans/2026-07-25-mutation-fallback-pipeline.md
```

---

## Key Entry Points (quick reference)

| Task | Command / file |
|------|----------------|
| Run an AutoRed experiment | `VLLM_USE_V1=0 python experiment/llama_3_8b_vllm.py --mode experiment ...` (from `AutoRed-Final/`) |
| Run an AutoRed benchmark | same with `--mode benchmark ...` |
| Enable mutation fallback | add `--enable-mutation-fallback` or `AUTORED_MUTATION_FALLBACK=1` |
| 4-GPU HPC benchmark | `AutoRed-Final/hpc/autored_benchmark_4gpu_vllm.sh ...` |
| Merge worker results | `python scripts/merge_benchmarks.py ...` |
| Start backend + UI | `AUTORED_SERVER_MODE=1 python -m uvicorn server.main:app` then `cd ui && npm run dev` |
| Backend, no models (laptop) | `AUTORED_LOAD_MODELS=0 python -m uvicorn server.main:app` |
| JailGuard single text detect | `python run_single.py --serial_num 9521` (from `JailGuard/jailguard_reimpl/`) |
| JailGuard batch detect | `python run_batch.py` (from `JailGuard/jailguard_reimpl/`) |
| combination tests | `cd combination && python -m pytest tests/ -v` |

> **pytest note:** pytest is not on the system Python PATH. Use the
> `AutoRed-Final/.venv` or `JailGuard/.venv` interpreter, or install it into
> whichever environment you run the combination tests from.

---

## Important / Large Artifacts (gitignored)

AutoRed model weights and datasets are not in git. Expected local paths are
listed in `AutoRed-Final/largefiles.txt`; notable ones:

- `models/defense_classifier/`, `models/ranker_deberta_v1/`
- `pre_trained/pi_reward_model/` (judge)
- `experiment/access_code_predictor/`
- `experiment/results/planner_sft_v2*/`, `experiment/results/generator_sft_v2/`
- `data/*--largeFile.jsonl`, `data/TensorTrust_subsets/*.jsonl`
- `data/rag/success_defenses.index` + `success_metadata.json`

The `.gitignore` at the repo root ignores `__pycache__`, `.venv/`,
`.superpowers/`, and binary model files (`*.pt`, `*.pth`, `*.safetensors`,
`*.index`, `*.db`, …).

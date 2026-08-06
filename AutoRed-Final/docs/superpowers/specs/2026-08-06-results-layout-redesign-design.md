# AutoRed-Final Results Layout Redesign

**Date:** 2026-08-06
**Status:** Approved (design sections 1–5) — pending spec review
**Scope:** AutoRed-Final/ — new on-disk results hierarchy for benchmark and single-run modes, plus one-time migration of existing `results/`.

## 1. Objective

Replace the current ad-hoc results hierarchy (`results/<run_id>.json`, `results/YYYY-MM-DD/HH-MM-SS/`, `results/benchmarks/<label>/worker_*.json + merged_summary.json`) with a single, predictable layout keyed by **mode → model → characteristics → {logs, runs/{success, failed}}**. Move the existing `results/` bulk to `results_old/` (deduplicated) and start fresh.

## 2. Current state (what exists today)

Two result-writing paths:

1. **HPC benchmark path** — `hpc/autored_benchmark_4gpu_vllm.sh` launches N workers of `experiment/llama_3_8b_vllm.py --mode benchmark`. Each worker writes `worker_{id}.json` (per-worker aggregate) into `--benchmark-output` (the `OUTPUT_DIR`), then `scripts/merge_benchmarks.py` writes `merged_summary.json` into the same dir. Per-round run JSONs are written to `results/YYYY-MM-DD/HH-MM-SS_<microsec>/` (hardcoded at `llama_3_8b_vllm.py:4030`).
2. **Single/server path** — `worker/experiment_runner.py:291` and the `--mode single` path in `llama_3_8b_vllm.py` write flat `results/<run_id>.json`.

Per-run JSON already records the winning stage in its `result` field:
```
result: { ground_truth_success, generator_success, extractor_success,
           verified_success, success_reason, extracted_value, total_attempts }
```
A run is successful when any of ground-truth leak / extractor match / verifier-accepted is true — which is already what `attempts < MAX_INTERACTIONS` reflects (the loop breaks on success). The mutation-fallback path (commits 9d8bdaa8, bb06f6f9) records fallback success through the same `result`/`success_reason` fields.

`results/` currently holds ~13,200 files (~1.4 GB) and is **git-tracked** (the branch shows thousands of deleted run files). An empty `results_old/` already exists with 4 leftover Mistral benchmark dirs.

## 3. Decisions (locked in during brainstorming)

| # | Decision |
|---|----------|
| D1 | **Migration**: leave existing `results/` bulk alone — rename in place to `results_old/`, dedup by content hash, do not aggressively delete. User deletes later manually. |
| D2 | Top level of `results/` holds exactly `benchmark/` and `single/` — nothing else. |
| D3 | **Model id** is auto-injected from where the model is loaded. Source of truth: a new `--victim-model-id` flag; fallback to parsing the HF cache dir (`models--org--name` → `org/name`); fallback to the resolved load path basename. User never types the model id in `--output-dir`. |
| D4 | **Characteristics** = everything in `--output-dir` after the `benchmark/` or `single/` segment, used verbatim. Internal `/` → `_` so characteristics is one directory (not nested). |
| D5 | `success/` and `failed/` hold **flat** run files; the winning stage stays in the run JSON's `result` field. No per-stage subfolders. |
| D6 | **Single mode** mirrors the benchmark layout (full new tree), not flat. |
| D7 | `logs/` = worker `.log` files + aggregate summaries (`worker_N.json`, `merged_summary.json`). `runs/` = per-round individual run JSONs split into `success/` and `failed/`. |
| D8 | **Routing approach**: route each per-round run JSON to `success/` or `failed/` directly inside the benchmark loop (success is known at build time) — no staging dir, no second pass. |
| D9 | **Run filename**: `run_<scenario_id>_w<worker>_<round>.json` (benchmark), `run_<scenario_id>_single.json` (single). The `<round>` tiebreaker guarantees uniqueness even under with-replacement sampling / duplicate defense_ids. |

## 4. Target directory layout

```
results/
├── benchmark/
│   └── <model_id>/                         # slugified victim model (org/name -> org--name)
│       └── <characteristics>/               # verbatim tail of --output-dir after benchmark/
│           ├── logs/
│           │   ├── worker_0.json            # per-worker aggregate (existing shape, unchanged)
│           │   ├── worker_1.json
│           │   ├── worker_0.log             # per-worker stdout (moved from top-level logs/)
│           │   ├── worker_1.log
│           │   └── merged_summary.json       # final merged aggregate
│           └── runs/
│               ├── success/
│               │   └── run_<scenario_id>_w<worker>_<round>.json
│               └── failed/
│                   └── run_<scenario_id>_w<worker>_<round>.json
└── single/
    └── <model_id>/
        └── <characteristics>/               # defaults to single_YYYY-MM-DD_HH-MM-SS if no --output-dir
            ├── logs/
            │   ├── run.log                   # single-run stdout
            │   └── verbose_trace.json        # the verbose terminal trace, now colocated
            └── runs/
                └── success/   (or failed/)
                    └── run_<scenario_id>_single.json
```

### Naming rules

- **`<model_id>` slug** (`slugify_model_id`):
  - HF id `org/name` or bare `name` → `org--name` / `name` (single directory segment).
  - Local checkpoint path → `<basename>_<6char-hash-of-full-path>` (collision-safe).
  - Slug strips path separators; any other char unsafe for a filesystem dir name is percent/replaced conservatively.
- **`<characteristics>`**: verbatim tail of `--output-dir` after the mode segment. Any `/` within it is replaced with `_` so it is a single directory. May be user-defined (e.g. `Llama3-1000-2000_Mutation-3_subset-8_seed-7_2026-07-31_23-16-07_4g`).
- **`<scenario_id>`**: the defense_id (integer in the current dataset, filename-safe). Slugified defensively in case future datasets use non-integer ids.
- **`<worker>`**: 0-based worker id; single-worker and single-mode runs use `w0` / `_single`.
- **`<round>`**: global 1-based round index within the worker — the uniqueness tiebreaker for repeated scenarios.

## 5. Success / failed classification

- **Benchmark**: after `_build_benchmark_run_json`, route by the existing `success = attempts < MAX_INTERACTIONS` flag (set at `llama_3_8b_vllm.py:3831`). The winning stage is **not** used for routing — it remains in the run JSON `result` field (`ground_truth_success` / `extractor_success` / `verified_success` / `success_reason`).
- **Single**: after `verbose_test_llama`, classify by the same rule — `tries < MAX_INTERACTIONS` → `success/`, else `failed/`.
- Folder rule equals the data's existing success definition (`real_success = success_exact or success_extractor or verified_success`), so on-disk placement and JSON contents agree.

### Artifact → destination

| Artifact | Destination |
|---|---|
| Per-round run JSON (success) | `runs/success/run_<scenario_id>_w<worker>_<round>.json` |
| Per-round run JSON (failed) | `runs/failed/run_<scenario_id>_w<worker>_<round>.json` |
| Per-worker aggregate `worker_N.json` | `logs/worker_N.json` |
| Worker stdout log | `logs/worker_N.log` |
| `merged_summary.json` | `logs/merged_summary.json` |
| Single run JSON | `runs/{success\|failed}/run_<scenario_id>_single.json` |
| Single verbose trace + stdout | `logs/` |

## 6. Code changes

### New file: `experiment/results_layout.py`
A single small, unit-tested module owning the layout rules:
```python
def slugify_model_id(model_id: str) -> str
def resolve_model_id(victim_model_id: str | None, load_path: str | None) -> str
def parse_output_dir(output_dir: str | None, mode: str) -> tuple[str, str]   # (mode, characteristics)
def runs_root(output_dir, mode, model_id, characteristics) -> Path            # the characteristics root
def run_filename(scenario_id, worker_id, round) -> str                       # benchmark
def single_run_filename(scenario_id) -> str                                  # single
```

### Edits
1. **`experiment/llama_3_8b_vllm.py`**
   - Add CLI flags: `--victim-model-id`, `--output-dir` (the mode/characteristics root). `--benchmark-output` is kept as a **deprecated alias** for `--output-dir` only when it already follows the new shape (`results/{benchmark|single}/...`); if a caller passes an old-style flat `--benchmark-output <dir>`, the code treats that `<dir>` basename as the characteristics and routes under the resolved `benchmark/<model_id>/` (i.e. it is reinterpreted, not used verbatim as the characteristics root). A deprecation warning is printed.
   - In `run_benchmark`: resolve the layout dir once; pass `runs_root` into the loop; write each `run_json` directly to `runs/success/` or `runs/failed/` by `success`; write `worker_N.json` to `logs/`.
   - Replace the hardcoded `results/YYYY-MM-DD/HH-MM-SS/` per-round save (`llama_3_8b_vllm.py:4030`) with the new `runs/{success,failed}/` path.
   - Single mode (`--mode single`): write to `results/single/<model_id>/<chars>/runs/{success,failed}/run_<sid>_single.json`; move the verbose trace + stdout into `logs/`.
2. **`worker/experiment_runner.py`** (server/UI path) — route through the same `results_layout` helpers so single runs from the server land in the new tree. Default characteristics `single_<timestamp>` when not specified.
3. **`hpc/autored_benchmark_4gpu_vllm.sh`** — pass `--output-dir results/benchmark/<model_id>/<chars>` and `--victim-model-id`; redirect worker stdout to `<chars>/logs/worker_N.log` instead of top-level `logs/`. (Other HPC benchmark wrappers updated analogously where they share the pattern.)
4. **`scripts/merge_benchmarks.py`** — read `worker_N.json` from the new `logs/` path; write `merged_summary.json` to `logs/`. No per-round moving (Approach A routed them at write time).

### Untouched
- The run JSON schema (`schemas/run_v2.schema.json`) and the `worker_N.json` / `merged_summary.json` shapes stay identical — only **where** they are written changes. Recursive `results/**/*.json` globs keep matching.

## 7. Migration of existing results

Per D1 — rename in place, dedup, do not delete aggressively.

1. `git mv results results_old` for git-tracked paths (the branch already stages most as deletions, so this is cheap); plain `mv` for the untracked bulk. Fold the existing empty `results_old/` (4 Mistral dirs) into the rename.
2. **Dedup**: content-hash (SHA-256) all `.json` files under `results_old/`; remove byte-identical duplicates, keeping one copy per hash (prefer the lexicographically-first path).
3. Create a fresh empty `results/` with `benchmark/` and `single/`.

Implemented as a one-time, idempotent, `--dry-run`-able script: `scripts/migrate_results_layout.py` with `--migrate` and `--dedup` subcommands.

## 8. Downstream reader impact

In-repo scripts that hardcode the old layout must be updated so they don't silently break:

| File:line | Current | Change |
|---|---|---|
| `server/file_manager.py:108,120` | lists `results/YYYY-MM-DD/*`, `results/benchmarks` | walk the new tree |
| `scripts/analyze_vllm_benchmark.py:191` | default `results/benchmarks/batched_100r_4g/merged_summary.json` | default `.../logs/merged_summary.json` under new tree |
| `scripts/dataset_tools/analyse_and_fill_sheet.py:78` | recursive `results/**/*.json` | add `results_old/` exclude |
| `scripts/dataset_tools/build_planner_confusion.py:122` | `--results-dir` default `results/` | default unchanged but note new sub-structure |
| `scripts/dataset_tools/build_primitive_matrix.py:178` | `--results-dir` default `results/` | same |
| `scripts/dataset_tools/analyze_oracle_5000r.py:271` | `analyze("results/2026-06-28", ...)` | repoint to `results_old/...` for legacy data |

**Backward-compat caveat (flagged, not auto-fixed):** external user scripts that hardcode `results/benchmarks/...` will now find nothing (data moved to `results_old/`). The spec's migration notes will state this explicitly.

## 9. Ambiguities & edge cases

1. **Model id with slashes** (`org/name`) → slugify to `org--name` (single dir). Resolved.
2. **Characteristics with slashes** (`results/benchmark/a/b`) → user said "everything after `benchmark/` is characteristics"; kept as a single directory by replacing `/` → `_` (`a_b`). ⚠️ Judgment call — confirmed acceptable during brainstorming.
3. **Same scenario repeats within a worker** (with-replacement sampling at `llama_3_8b_vllm.py:3719`, plus duplicate defense_ids in the dataset) → `<round>` tiebreaker. Resolved.
4. **`results_old/` already exists** → fold into the rename. Resolved.
5. **Recursive globs now scan `results_old/`** → exclude pattern. Resolved.
6. **Run JSON schema** → unchanged; only paths change. Resolved.
7. **"fallback" as a stage** — the user's list said `[gt, extraction, verifier, fallback]`, but fallback is a retry *path*, not a success signal like the other three. In the data, fallback success is captured by the same `result.success` / `success_reason` fields. The design does **not** create a `fallback/` folder — it is recorded in metadata like the other stages. ⚠️ Judgment call — confirmed acceptable during brainstorming.
8. **Single mode with no `--output-dir`** → default `single_<timestamp>` characteristics. Resolved.

## 10. Testing

- **Unit — `results_layout.py`**: `slugify_model_id` (HF id, local path, weird chars); `parse_output_dir` (full-path, bare-characteristics, no-arg single modes); `run_filename` uniqueness under repeated scenario_ids.
- **Integration — benchmark**: tiny `--rounds 4 --num-workers 2` run; assert `runs/success/` + `runs/failed/` split, `logs/worker_N.json` + `merged_summary.json` present, filenames match `run_<sid>_w<w>_<r>.json`, no collisions.
- **Integration — single**: one run → correct `runs/{success\|failed}/` + `logs/` placement.
- **Migration**: `--dry-run` of the dedup script on `results_old/`; assert byte-identical files collapse and the kept copy is readable JSON.

## 11. Out of scope

- No change to the run/worker/merged-summary JSON schemas.
- No re-sampling or benchmark-semantics change (with-replacement behavior preserved).
- No change to model loading or the experiment loop logic — only where results are written.
- External user scripts hardcoding old paths — flagged, not auto-migrated.

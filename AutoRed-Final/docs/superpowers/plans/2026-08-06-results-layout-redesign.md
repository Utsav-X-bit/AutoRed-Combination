# AutoRed-Final Results Layout Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AutoRed-Final's ad-hoc results hierarchy with `results/{benchmark|single}/<model_id>/<characteristics>/{logs, runs/{success, failed}}/`, route per-round run JSONs to `success/` or `failed/` at write time, migrate the existing `results/` to `results_old/` (deduplicated), and update downstream readers.

**Architecture:** A single new module `experiment/results_layout.py` owns all path/naming rules and is unit-tested. The benchmark loop in `experiment/llama_3_8b_vllm.py` and the server path in `worker/experiment_runner.py` call into it. The HPC shell script and `scripts/merge_benchmarks.py` are updated to the new `logs/`/`runs/` split. A one-time `scripts/migrate_results_layout.py` performs the rename + content-hash dedup. Run JSON schemas are unchanged — only where files are written changes.

**Tech Stack:** Python 3.10, pytest 9.1.1 (venv at `.venv/`), pandas, pathlib, argparse; bash for HPC wrappers. Run `pytest` via `.venv/bin/python -m pytest`.

## Global Constraints

- **Run JSON schema is untouched.** `schemas/run_v2.schema.json`, the `worker_N.json` aggregate shape, and `merged_summary.json` shape stay identical — only destination paths change.
- **No benchmark-semantics change.** With-replacement sampling at `llama_3_8b_vllm.py:3719` is preserved; the `<round>` tiebreaker in filenames handles repeats.
- **Model id is auto-injected, never typed in `--output-dir`.** Source of truth: new `--victim-model-id` flag → HF cache path parse → load-path basename fallback.
- **Characteristics is a single directory.** Any `/` in the user's characteristics string is replaced with `_`.
- **`results/` top level holds only `benchmark/` and `single/`.**
- **Migration is non-destructive**: rename in place to `results_old/`, dedup by SHA-256; no aggressive deletion.
- All file paths are relative to `AutoRed-Final/` (the project root: `/nlsasfs/home/isea/isea38/autoredPLUSjailguard/AutoRed-Final`).
- Commits end with `Co-Authored-By: Claude <noreply@anthropic.com>`. Do not push unless asked.

---

## File Structure

| File | Responsibility |
|---|---|
| `experiment/results_layout.py` (NEW) | Owns `slugify_model_id`, `resolve_model_id`, `parse_output_dir`, `runs_root`, `run_filename`, `single_run_filename`. Pure functions, no I/O except `runs_root` which `mkdir -p`s. |
| `scripts/tests/test_results_layout.py` (NEW) | Unit tests for the above. |
| `experiment/llama_3_8b_vllm.py` (MODIFY) | Add `--victim-model-id`/`--output-dir`; route benchmark + single writes through `results_layout`. |
| `worker/experiment_runner.py` (MODIFY) | Route the server single-run path through `results_layout`. |
| `hpc/autored_benchmark_4gpu_vllm.sh` (MODIFY) | Pass `--output-dir`/`--victim-model-id`; redirect worker stdout to `<chars>/logs/worker_N.log`. |
| `scripts/merge_benchmarks.py` (MODIFY) | Read `worker_N.json` from `logs/`; write `merged_summary.json` to `logs/`. |
| `scripts/migrate_results_layout.py` (NEW) | One-time `--migrate` (rename) + `--dedup` (SHA-256 dedup), both `--dry-run`-able. |
| `server/file_manager.py` (MODIFY) | Walk the new tree instead of `results/YYYY-MM-DD/*` and `results/benchmarks`. |
| `scripts/analyze_vllm_benchmark.py` (MODIFY) | Default summary path → `.../logs/merged_summary.json`. |
| `scripts/dataset_tools/analyse_and_fill_sheet.py` (MODIFY) | Exclude `results_old/` from the recursive glob. |

---

## Task 1: `results_layout.py` — slugify_model_id and resolve_model_id

**Files:**
- Create: `experiment/results_layout.py`
- Test: `scripts/tests/test_results_layout.py`

**Interfaces:**
- Produces: `slugify_model_id(model_id: str) -> str`, `resolve_model_id(victim_model_id: str | None, load_path: str | None = None) -> str`

- [ ] **Step 1: Write failing tests**

Create `scripts/tests/test_results_layout.py`:
```python
import hashlib
from experiment.results_layout import slugify_model_id, resolve_model_id


def test_slugify_hf_id_replaces_slash():
    assert slugify_model_id("meta-llama/Meta-Llama-3-8B-Instruct") == "meta-llama--Meta-Llama-3-8B-Instruct"


def test_slugify_bare_name_unchanged():
    assert slugify_model_id("gpt2") == "gpt2"


def test_slugify_strips_surrounding_separators():
    assert slugify_model_id("/org/name/") == "org--name"


def test_slugify_unsafe_chars_replaced():
    out = slugify_model_id("org/name with space")
    assert " " not in out
    assert "/" not in out


def test_resolve_prefers_explicit_flag():
    assert resolve_model_id("meta-llama/Meta-Llama-3-8B-Instruct", None) == "meta-llama--Meta-Llama-3-8B-Instruct"


def test_resolve_parses_hf_cache_dir():
    p = "/home/user/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/abc"
    assert resolve_model_id(None, p) == "meta-llama--Meta-Llama-3-8B-Instruct"


def test_resolve_local_path_basename_plus_hash():
    p = "/nlsasfs/home/isea/isea38/AutoRed-Final/experiment/results/planner_sft_v2/checkpoint-27"
    expected_hash = hashlib.sha256(p.encode()).hexdigest()[:6]
    assert resolve_model_id(None, p) == f"checkpoint-27_{expected_hash}"


def test_resolve_fallback_unknown():
    assert resolve_model_id(None, None) == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest scripts/tests/test_results_layout.py -v`
Expected: collection error / `ModuleNotFoundError: No module named 'experiment.results_layout'`

- [ ] **Step 3: Implement minimal code**

Create `experiment/results_layout.py`:
```python
"""Results directory layout rules for AutoRed-Final.

Owns the mapping from (mode, model_id, characteristics) to on-disk paths.
All functions are pure except ``runs_root``, which creates directories.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

# Characters safe to keep verbatim in a single directory name.
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _slug_segment(value: str) -> str:
    """Return a single filesystem-safe directory segment.

    Slashes become ``--`` (preserving HF org/name readability); any other
    unsafe character becomes ``_``; surrounding separators are stripped.
    """
    value = value.strip().strip("/")
    value = value.replace("/", "--")
    value = _SAFE.sub("_", value)
    value = value.strip(".-_")
    return value or "unknown"


def slugify_model_id(model_id: str) -> str:
    """Slugify a HuggingFace model id or bare name into one directory name.

    ``org/name`` -> ``org--name``; ``gpt2`` -> ``gpt2``.
    """
    return _slug_segment(model_id)


def resolve_model_id(victim_model_id: str | None, load_path: str | None = None) -> str:
    """Resolve the model id directory segment.

    Priority:
      1. explicit ``victim_model_id`` (from --victim-model-id),
      2. a HuggingFace cache dir of the form ``.../models--org--name/snapshots/...``,
      3. basename of ``load_path`` + 6-char hash (collision-safe for local checkpoints),
      4. ``"unknown"``.
    """
    if victim_model_id:
        return slugify_model_id(victim_model_id)

    if load_path:
        # HF cache layout: models--org--name
        m = re.search(r"models--(.+)$", str(load_path))
        if m:
            tail = m.group(1)
            if "--" in tail:
                # org--name form
                return _slug_segment(tail.replace("--", "/", 1))
            return _slug_segment(tail)
        # Local checkpoint path -> basename + hash
        base = Path(load_path).name or "model"
        h = hashlib.sha256(str(load_path).encode()).hexdigest()[:6]
        return f"{_slug_segment(base)}_{h}"

    return "unknown"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest scripts/tests/test_results_layout.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add experiment/results_layout.py scripts/tests/test_results_layout.py
git commit -m "feat(layout): slugify_model_id + resolve_model_id with tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: `results_layout.py` — parse_output_dir and runs_root

**Files:**
- Modify: `experiment/results_layout.py`
- Modify: `scripts/tests/test_results_layout.py`

**Interfaces:**
- Consumes: `slugify_model_id` (Task 1), `resolve_model_id` (Task 1)
- Produces: `parse_output_dir(output_dir: str | None, mode: str) -> tuple[str, str]` (returns `(mode, characteristics)`), `runs_root(output_dir: str | None, mode: str, model_id: str, characteristics: str, base: str = "results") -> Path`

- [ ] **Step 1: Write failing tests**

Append to `scripts/tests/test_results_layout.py`:
```python
from experiment.results_layout import parse_output_dir, runs_root


def test_parse_output_dir_benchmark_full_path():
    mode, chars = parse_output_dir(
        "results/benchmark/Llama3-1000-2000_Mutation-3_subset-8_seed-7", "benchmark"
    )
    assert mode == "benchmark"
    assert chars == "Llama3-1000-2000_Mutation-3_subset-8_seed-7"


def test_parse_output_dir_single_full_path():
    mode, chars = parse_output_dir("results/single/myrun", "single")
    assert mode == "single"
    assert chars == "myrun"


def test_parse_output_dir_bare_characteristics():
    mode, chars = parse_output_dir("Llama3-1000-2000_seed-7", "benchmark")
    assert mode == "benchmark"
    assert chars == "Llama3-1000-2000_seed-7"


def test_parse_output_dir_none_returns_default():
    mode, chars = parse_output_dir(None, "single")
    assert mode == "single"
    assert chars.startswith("single_")


def test_parse_output_dir_slashes_in_chars_collapsed():
    mode, chars = parse_output_dir("results/benchmark/a/b/c", "benchmark")
    assert chars == "a_b_c"
    assert "/" not in chars


def test_parse_output_dir_rejects_bad_mode():
    import pytest
    with pytest.raises(ValueError):
        parse_output_dir("results/benchmark/x", "weird")


def test_runs_root_structure():
    root = runs_root(None, "benchmark", "meta-llama--Meta-Llama-3-8B-Instruct", "chars_x")
    expected = Path("results") / "benchmark" / "meta-llama--Meta-Llama-3-8B-Instruct" / "chars_x"
    assert root == expected
    assert (root / "logs").exists()
    assert (root / "runs" / "success").exists()
    assert (root / "runs" / "failed").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest scripts/tests/test_results_layout.py -v -k "parse_output_dir or runs_root"`
Expected: FAIL (`ImportError: cannot import name 'parse_output_dir'`)

- [ ] **Step 3: Implement minimal code**

Append to `experiment/results_layout.py`:
```python
from datetime import datetime

_VALID_MODES = ("benchmark", "single")


def parse_output_dir(output_dir: str | None, mode: str) -> tuple[str, str]:
    """Return ``(mode, characteristics)`` from an --output-dir argument.

    Accepts:
      * full path  ``results/<mode>/<chars>``
      * bare characteristics ``<chars>``
      * ``None`` (returns a timestamped default for single mode)

    The characteristics segment is a single directory: any ``/`` inside it
    is collapsed to ``_``.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"invalid mode {mode!r}; expected one of {_VALID_MODES}")

    if not output_dir:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return mode, f"{mode}_{stamp}"

    s = output_dir.replace("\\", "/").strip("/")
    prefix = f"results/{mode}/"
    if s.startswith(prefix):
        chars = s[len(prefix):]
    else:
        chars = s
    chars = chars.replace("/", "_")
    chars = chars.strip(".-_") or f"{mode}_unnamed"
    return mode, chars


def runs_root(
    output_dir: str | None,
    mode: str,
    model_id: str,
    characteristics: str,
    base: str = "results",
) -> Path:
    """Return the characteristics root and create the full sub-tree.

    Creates ``<base>/<mode>/<model_id>/<characteristics>/{logs, runs/{success, failed}}``.
    """
    root = Path(base) / mode / model_id / characteristics
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "success").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "failed").mkdir(parents=True, exist_ok=True)
    return root
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest scripts/tests/test_results_layout.py -v`
Expected: all pass (15 total). Remove the stray `results/...` dirs created by `test_runs_root_structure` if it leaves artifacts; the test creates real dirs — add a cleanup:

Append the `tmp`-path variant and a fixture. Replace `test_runs_root_structure` body to use a `tmp_path` base:
```python
def test_runs_root_structure(tmp_path):
    root = runs_root(None, "benchmark", "meta-llama--Meta-Llama-3-8B-Instruct", "chars_x", base=str(tmp_path))
    expected = tmp_path / "benchmark" / "meta-llama--Meta-Llama-3-8B-Instruct" / "chars_x"
    assert root == expected
    assert (root / "logs").exists()
    assert (root / "runs" / "success").exists()
    assert (root / "runs" / "failed").exists()
```

Re-run: `.venv/bin/python -m pytest scripts/tests/test_results_layout.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add experiment/results_layout.py scripts/tests/test_results_layout.py
git commit -m "feat(layout): parse_output_dir + runs_root with tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: `results_layout.py` — run_filename and single_run_filename

**Files:**
- Modify: `experiment/results_layout.py`
- Modify: `scripts/tests/test_results_layout.py`

**Interfaces:**
- Produces: `run_filename(scenario_id: str | int, worker_id: int, round: int) -> str`, `single_run_filename(scenario_id: str | int) -> str`

- [ ] **Step 1: Write failing tests**

Append to `scripts/tests/test_results_layout.py`:
```python
from experiment.results_layout import run_filename, single_run_filename


def test_run_filename_basic():
    assert run_filename(7, 2, 15) == "run_7_w2_15.json"


def test_run_filename_string_scenario():
    assert run_filename("89021", 0, 1) == "run_89021_w0_1.json"


def test_run_filename_unsafe_scenario_slug():
    out = run_filename("weird id", 1, 3)
    assert out == "run_weird_id_w1_3.json"
    assert " " not in out


def test_run_filename_unique_under_repeat():
    a = run_filename(7, 2, 5)
    b = run_filename(7, 2, 6)
    assert a != b


def test_single_run_filename():
    assert single_run_filename(7) == "run_7_single.json"


def test_single_run_filename_unsafe():
    assert single_run_filename("a b") == "run_a_b_single.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest scripts/tests/test_results_layout.py -v -k "run_filename or single_run"`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement minimal code**

Append to `experiment/results_layout.py`:
```python
def _sid(scenario_id) -> str:
    """Slugify a scenario id for use in a filename (spaces -> _)."""
    return _SAFE.sub("_", str(scenario_id).strip())


def run_filename(scenario_id, worker_id: int, round: int) -> str:
    """Benchmark per-round filename: run_<scenario_id>_w<worker>_<round>.json."""
    return f"run_{_sid(scenario_id)}_w{int(worker_id)}_{int(round)}.json"


def single_run_filename(scenario_id) -> str:
    """Single-mode filename: run_<scenario_id>_single.json."""
    return f"run_{_sid(scenario_id)}_single.json"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest scripts/tests/test_results_layout.py -v`
Expected: all pass (21 total).

- [ ] **Step 5: Commit**

```bash
git add experiment/results_layout.py scripts/tests/test_results_layout.py
git commit -m "feat(layout): run_filename + single_run_filename with tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Wire `--victim-model-id` and `--output-dir` into the CLI

**Files:**
- Modify: `experiment/llama_3_8b_vllm.py` (argparse block ~lines 5112-5184, and the `__main__` body up to ~5190)

**Interfaces:**
- Consumes: `resolve_model_id`, `parse_output_dir`, `runs_root` (Tasks 1-2)
- Produces: two new CLI args consumed by the benchmark (Task 5) and single (Task 6) paths; a module-level `RESULTS_LAYOUT_ROOT` is **not** introduced — the resolved root is passed as function args to avoid globals.

- [ ] **Step 1: Add the CLI arguments**

In `experiment/llama_3_8b_vllm.py`, in the argparse block (after the `--num-workers` argument around line 5178), add:
```python
    parser.add_argument(
        "--victim-model-id",
        default=None,
        help="HuggingFace model id of the victim (e.g. meta-llama/Meta-Llama-3-8B-Instruct). "
        "Used to place results under results/<mode>/<model_id>/. "
        "If omitted, derived from the model load path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Results root for this run, e.g. results/benchmark/<characteristics> "
        "or results/single/<characteristics>. The segment after <mode>/ is the "
        "characteristics label, used verbatim. If omitted, a timestamped default is used.",
    )
```

- [ ] **Step 2: Resolve the layout once in `__main__`**

Immediately after `BENCHMARK_LOG_PATH = args.benchmark_output` (around line 5188), add:
```python
    from experiment.results_layout import resolve_model_id, parse_output_dir, runs_root

    _VICTIM_MODEL_ID = resolve_model_id(args.victim_model_id, LLAMA_PATH)
    _MODE = args.mode if args.mode in ("benchmark", "single") else "benchmark"
    if args.mode == "extractor_benchmark":
        _MODE = "benchmark"  # extractor benchmark reuses the benchmark tree
    _CHARS = parse_output_dir(args.output_dir, _MODE)[1]
    RESULTS_ROOT = runs_root(args.output_dir, _MODE, _VICTIM_MODEL_ID, _CHARS)
    print(f"[LAYOUT] results root: {RESULTS_ROOT}")
```

Note: `--benchmark-output` is kept as a deprecated alias. When `--output-dir` is unset but `--benchmark-output` was passed non-default, we warn. Add after the block above:
```python
    if args.output_dir is None and args.benchmark_output != BENCHMARK_LOG_PATH:
        print(
            "[WARN] --benchmark-output is deprecated; use --output-dir "
            "results/<mode>/<characteristics>. Treating its basename as characteristics."
        )
        _CHARS = parse_output_dir(args.benchmark_output, _MODE)[1]
        RESULTS_ROOT = runs_root(None, _MODE, _VICTIM_MODEL_ID, _CHARS)
```

- [ ] **Step 3: Smoke-check argparse parses**

Run: `.venv/bin/python experiment/llama_3_8b_vllm.py --help 2>&1 | grep -E "victim-model-id|output-dir"`
Expected: both lines appear in the help text. (This imports the module; vLLM is not initialized by `--help`.)

- [ ] **Step 4: Commit**

```bash
git add experiment/llama_3_8b_vllm.py
git commit -m "feat(cli): add --victim-model-id and --output-dir, resolve results root

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Route benchmark run JSONs to `runs/{success,failed}/` at write time

**Files:**
- Modify: `experiment/llama_3_8b_vllm.py` — `run_benchmark` signature (line 3645) and save block (lines 4022-4040)

**Interfaces:**
- Consumes: `runs_root`, `run_filename` (Tasks 1-3), the `RESULTS_ROOT` resolved in `__main__` (Task 4)
- Produces: per-round run JSONs at `<root>/runs/{success|failed}/run_<sid>_w<worker>_<round>.json`; `worker_N.json` at `<root>/logs/worker_N.json`

- [ ] **Step 1: Extend `run_benchmark` signature**

At line 3645, change:
```python
def run_benchmark(
    agent: RedTeamingAgent,
    n_rounds: int = BENCHMARK_ROUNDS,
    verbose: bool = False,
    worker_id: int = 0,
    num_workers: int = 1,
) -> dict:
```
to:
```python
def run_benchmark(
    agent: RedTeamingAgent,
    n_rounds: int = BENCHMARK_ROUNDS,
    verbose: bool = False,
    worker_id: int = 0,
    num_workers: int = 1,
    results_root: Path | None = None,
) -> dict:
```
Add at the top of the function body (after the docstring), the import and a fallback:
```python
    from experiment.results_layout import runs_root as _runs_root, run_filename
    if results_root is None:
        results_root = _runs_root(None, "benchmark", "unknown", "benchmark_default")
    runs_dir = results_root / "runs"
    logs_dir = results_root / "logs"
```

- [ ] **Step 1.5: Keep the real defense_id through the benchmark loop (pre-existing bug fix)**

The current code drops the defense_id before the loop, so `row.name`/`scenario._defense_id` is the row POSITION, not the real defense_id. Fix the column selection and the reset_index so the real defense_id threads through to both the run JSON and the new filename.

At line 3724, change:
```python
    # Keep only the columns we need
    scenarios_df = scenarios_df[["opening_defense", "closing_defense", "access_code"]]
```
to:
```python
    # Keep only the columns we need — preserve defense_id (the index) as a column
    # so it survives reset_index and threads through to run JSON + filename.
    scenarios_df = scenarios_df[["opening_defense", "closing_defense", "access_code"]].reset_index()
```
(`reset_index()` with no `drop=True` moves the `defense_id` index into a regular column named `defense_id`.)

At line 3728, the multi-worker branch does:
```python
        scenarios_list = scenarios_df.reset_index(drop=True).to_dict("records")
```
Change it to (no longer dropping the index, since defense_id is now a column):
```python
        scenarios_list = scenarios_df.to_dict("records")
```
(`scenarios_df` is already reset above, so this just converts to records keeping the `defense_id` column.)

After this fix, `row.name` in both branches is the positional integer again (the index is now default RangeIndex), so **do not use `row.name` for the scenario id**. Use `row["defense_id"]` instead. The existing lines `scenario._defense_id = str(row.name)` (line 3752) and `"defense_id": str(row.name) if hasattr(row, "name") else "unknown"` (line 4152) must be updated to read `row["defense_id"]`:
```python
            scenario._defense_id = str(row["defense_id"])      # line 3752
```
and in `_build_benchmark_run_json` (line 4152):
```python
        "defense_id": str(row["defense_id"]) if "defense_id" in row else "unknown",
```
This also fixes the pre-existing latent bug where benchmark-mode run JSONs recorded `defense_id` as the row position.

- [ ] **Step 2: Route each per-round run JSON to success/failed**

In the silent branch (after `run_json = _build_benchmark_run_json(...)` at line 3820 and after `success = attempts < MAX_INTERACTIONS` at line 3831), insert before `results.append(...)` (line 3913):
```python
                stage_dir = runs_dir / ("success" if success else "failed")
                fname = run_filename(row["defense_id"], worker_id, global_round_idx + 1)
                json_path = stage_dir / fname
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(run_json, f, indent=2, default=str)
```
In the verbose branch (after line 3758 where `run_json` is appended and `success` is set at line 3759), insert the same logic using the verbose branch's `scenario._defense_id` (now the real defense_id from Step 1.5) and the GLOBAL round index `batch_start + i + 1` (NOT the batch-local `i`):
```python
                stage_dir = runs_dir / ("success" if success else "failed")
                fname = run_filename(scenario._defense_id, worker_id, batch_start + i + 1)
                json_path = stage_dir / fname
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(run_json, f, indent=2, default=str)
```

Note: `row["defense_id"]` is the real defense_id (after Step 1.5). The round index (`global_round_idx + 1` in the silent branch, `batch_start + i + 1` in the verbose branch) is the global 1-based round within the worker and is the uniqueness tiebreaker for repeated scenarios.

- [ ] **Step 3: Replace the old bulk per-round save block**

Delete/replace the old per-round save block at lines 4029-4040:
```python
    # JSON emission: save per-round run JSONs
    results_dir = (
        Path("results")
        / benchmark_started_at.strftime("%Y-%m-%d")
        / benchmark_started_at.strftime("%H-%M-%S_%f")
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    for run_json in benchmark_run_jsons:
        json_path = results_dir / f"{run_json['experiment']['run_id']}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(run_json, f, indent=2, default=str)
    print(f"[JSON] {len(benchmark_run_jsons)} run JSONs saved to: {results_dir}/")
```
with:
```python
    # Per-round run JSONs are already written to runs/{success,failed}/ in the loop above.
    print(f"[JSON] {len(benchmark_run_jsons)} run JSONs saved to: {runs_dir}/")
```
Keep the list `benchmark_run_jsons` for any in-memory use, but no longer bulk-write it.

- [ ] **Step 4: Write `worker_N.json` to `logs/`**

At the benchmark save block (line 4022-4027), change:
```python
    benchmark_path = Path(BENCHMARK_LOG_PATH)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\n[JSON] Benchmark summary saved to: {benchmark_path}")
```
to:
```python
    logs_dir.mkdir(parents=True, exist_ok=True)
    worker_summary_path = logs_dir / f"worker_{worker_id}.json"
    with open(worker_summary_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\n[JSON] Worker summary saved to: {worker_summary_path}")
    # Also keep the legacy BENCHMARK_LOG_PATH copy for back-compat with old tooling.
    legacy = Path(BENCHMARK_LOG_PATH)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    with open(legacy, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)
```

- [ ] **Step 5: Pass `results_root` from `__main__`**

In the `__main__` benchmark dispatch (around line 5325, `benchmark = run_benchmark(...)`), add `results_root=RESULTS_ROOT`:
```python
            benchmark = run_benchmark(
                agent,
                n_rounds=args.rounds,
                verbose=False,
                worker_id=getattr(args, "worker_id", 0),
                num_workers=args.num_workers,
                results_root=RESULTS_ROOT,
            )
```

- [ ] **Step 6: Verify imports resolve (no vLLM init)**

Run: `.venv/bin/python -c "import ast; ast.parse(open('experiment/llama_3_8b_vllm.py').read()); print('syntax ok')"`
Expected: `syntax ok`.

- [ ] **Step 7: Commit**

```bash
git add experiment/llama_3_8b_vllm.py
git commit -m "feat(benchmark): route per-round run JSONs to runs/{success,failed} at write time

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Route single-mode run JSON to the new tree

**Files:**
- Modify: `experiment/llama_3_8b_vllm.py` — single-mode block (lines ~5255-5314) and `save_trace` (line 4891)
- Modify: `worker/experiment_runner.py` — save block (lines 290-296)

**Interfaces:**
- Consumes: `runs_root`, `single_run_filename` (Tasks 1-3), `RESULTS_ROOT` from `__main__` (Task 4)
- Produces: single run JSON at `<root>/runs/{success|failed}/run_<sid>_single.json`; verbose trace + stdout at `<root>/logs/`

- [ ] **Step 1: Single-mode write in `llama_3_8b_vllm.py`**

In the single-mode block, after `trace, tries, run_json = verbose_test_llama(scenario, agent)` (line ~5298) and after `save_trace(trace, scenario, tries)` (line ~5305), replace the print-only stub at line 5306-5309:
```python
            print(
                f"[JSON] UI run JSON available at: "
                f"results/{run_json['experiment']['run_id']}.json"
            )
```
with:
```python
            from experiment.results_layout import single_run_filename
            success = tries < MAX_INTERACTIONS
            stage_dir = RESULTS_ROOT / "runs" / ("success" if success else "failed")
            stage_dir.mkdir(parents=True, exist_ok=True)
            json_path = stage_dir / single_run_filename(scenario._defense_id)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(run_json, f, indent=2, default=str)
            print(f"[JSON] Run JSON saved to: {json_path}")
```

- [ ] **Step 2: Colocate the verbose trace under `logs/`**

In `save_trace` (line 4891), change the hardcoded `TRACE_LOG_PATH` destination. Read the current function:
```python
def save_trace(trace: list, scenario: DefenseScenario, total_attempts: int):
    ...
    trace_path = TRACE_LOG_PATH
    ...
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Full trace saved to: {trace_path}")
```
Add an optional `logs_dir` parameter and write there when provided:
```python
def save_trace(trace: list, scenario: DefenseScenario, total_attempts: int, logs_dir: Path | None = None):
    ...
    if logs_dir is not None:
        trace_path = logs_dir / "verbose_trace.json"
    else:
        trace_path = TRACE_LOG_PATH
    ...
```
Update the single-mode call site (line ~5305) to pass `RESULTS_ROOT / "logs"`:
```python
            save_trace(trace, scenario, tries, logs_dir=RESULTS_ROOT / "logs")
```

- [ ] **Step 3: Server path — `worker/experiment_runner.py`**

At lines 290-296, replace:
```python
    # Save to results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    json_path = results_dir / f"{run_json['experiment']['run_id']}.json"
    with open(json_path, "w") as f:
        json.dump(run_json, f, indent=2, default=str)
    print(f"\n💾 Run JSON saved to: {json_path}")
```
with:
```python
    from experiment.results_layout import (
        resolve_model_id, parse_output_dir, runs_root, single_run_filename,
    )
    model_id = resolve_model_id(None, model_info["victim"]["name"])
    _, chars = parse_output_dir(None, "single")
    root = runs_root(None, "single", model_id, chars)
    # `success` is already computed at line 151 of this function
    # (True iff ground_truth_found or extractor_success or verification_success).
    stage_dir = root / "runs" / ("success" if success else "failed")
    json_path = stage_dir / single_run_filename(scenario_id)
    with open(json_path, "w") as f:
        json.dump(run_json, f, indent=2, default=str)
    print(f"\n💾 Run JSON saved to: {json_path}")
```
The `success` variable is the local bool already maintained at line 151 of `run_experiment` (set True at line 233 when any of ground-truth/extractor/verifier succeeds) — reuse it rather than re-deriving from `run_json`. `scenario_id` is the local variable holding the selected defense_id (set at line 121).

- [ ] **Step 4: Verify syntax**

Run: `.venv/bin/python -c "import ast; ast.parse(open('experiment/llama_3_8b_vllm.py').read()); ast.parse(open('worker/experiment_runner.py').read()); print('syntax ok')"`
Expected: `syntax ok`.

- [ ] **Step 5: Commit**

```bash
git add experiment/llama_3_8b_vllm.py worker/experiment_runner.py
git commit -m "feat(single): route single-mode run JSON + trace to results/single/<model>/<chars>/

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Update `merge_benchmarks.py` to read from `logs/` and write `merged_summary.json` to `logs/`

**Files:**
- Modify: `scripts/merge_benchmarks.py` (lines 173-178, 233-262)

**Interfaces:**
- Consumes: worker summary JSONs at `<root>/logs/worker_*.json`
- Produces: `merged_summary.json` at `<root>/logs/merged_summary.json`

- [ ] **Step 1: Change the default output to `logs/merged_summary.json`**

At lines 173-178, change:
```python
    # Save merged results
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(merged, f, indent=2)
```
The `output_path` is passed by the caller, so the behavior change is in the caller (the HPC script, Task 8). Add a convenience: if `output_path` is a directory, write `merged_summary.json` inside it:
```python
    output = Path(output_path)
    if output.is_dir():
        output = output / "merged_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(merged, f, indent=2)
```

- [ ] **Step 2: No change to `--worker-results` handling**

The merge script already accepts explicit `worker_*.json` paths (glob), so no logic change is needed — the caller will pass `<root>/logs/worker_*.json`. Leave the `main()` glob expansion as-is.

- [ ] **Step 3: Verify the script still parses and runs on a synthetic input**

Create two tiny worker files under a temp `logs/` dir and run merge:
```bash
tmp=$(mktemp -d); mkdir -p "$tmp/logs"
.venv/bin/python - <<'PY'
import json, os
d="$TMP"
for wid in (0,1):
    json.dump({"success_rate":0.5,"total_successes":1,"total_rounds":2,
      "results":[{"round":1,"attempts":3,"success":True,"access_code":"x"}],
      "metadata":{"worker_id":wid,"target_model":"m","max_interactions":20},
      "total_success_exact":1,"total_success_extractor":0,"top1_success":1,
      "top3_success":1,"top5_success":1,"verified_success":0,"avg_attempts_on_success":3.0},
      open(f"$TMP/logs/worker_{wid}.json","w"))
PY
.venv/bin/python scripts/merge_benchmarks.py --output "$tmp/logs" --worker-results "$tmp"/logs/worker_*.json
ls "$tmp/logs/merged_summary.json" && echo OK
```
Expected: `merged_summary.json` is created and `OK` printed. (Adjust the heredoc `$TMP` substitution to a real temp path when running — the inline `$TMP` is illustrative; use the literal `tmp` value.)

- [ ] **Step 4: Commit**

```bash
git add scripts/merge_benchmarks.py
git commit -m "feat(merge): write merged_summary.json into logs/ when output is a dir

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Update the HPC benchmark shell script

**Files:**
- Modify: `hpc/autored_benchmark_4gpu_vllm.sh` (lines 17-24, 60-82, 127-129)

**Interfaces:**
- Consumes: `--output-dir`, `--victim-model-id` (Task 4), `runs_root` layout
- Produces: worker stdout at `<root>/logs/worker_N.log`; `merged_summary.json` at `<root>/logs/`

- [ ] **Step 1: Add VICTIM_MODEL_ID and change OUTPUT_DIR semantics**

At lines 17-24, add a `VICTIM_MODEL_ID` arg and reinterpret `OUTPUT_DIR`:
```bash
NUM_ROUNDS=${1:-1000}
PLANNER_PATH=${2:-"experiment/results/planner_sft_v2_contract_anchor/checkpoint-27"}
GENERATOR_PATH=${3:-"experiment/results/generator_sft_v2"}
BASE_GENERATOR_PATH=${4:-"Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"}
DATASET_PATH=${5:-"data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl"}
NUM_GPUS=4
DATASET_SIZE=${6:-1000}
OUTPUT_DIR=${7:-"results/benchmark/batched_${NUM_ROUNDS}r_4gpu"}
VICTIM_MODEL_ID=${8:-"meta-llama/Meta-Llama-3-8B-Instruct"}
```
(8th positional arg added; `OUTPUT_DIR` now follows the new `results/benchmark/<chars>` shape, where `<chars>` = `batched_${NUM_ROUNDS}r_4gpu`.)

- [ ] **Step 2: Pass the new flags to the worker invocation**

In the worker launch loop (lines 71-82), add `--output-dir "$OUTPUT_DIR"` and `--victim-model-id "$VICTIM_MODEL_ID"`, and redirect stdout to a per-worker log **inside** the resolved tree. Because the worker creates the `logs/` dir via `runs_root`, the log path is derived from the same layout. Add a small Python one-liner to resolve the logs dir, or compute it in bash:
```bash
    WORKER_OUTPUT="$OUTPUT_DIR/worker_${WORKER_ID}.json"
    # Resolve the per-worker log path under the new tree: results/benchmark/<model>/<chars>/logs/
    LOGS_DIR=$(python -c "from experiment.results_layout import resolve_model_id, parse_output_dir, runs_root; r=runs_root('$OUTPUT_DIR','benchmark',resolve_model_id('$VICTIM_MODEL_ID'),parse_output_dir('$OUTPUT_DIR','benchmark')[1]); print(r/'logs')")
    mkdir -p "$LOGS_DIR"
    WORKER_LOG="$LOGS_DIR/worker_${WORKER_ID}.log"
```
Then add to the `python experiment/llama_3_8b_vllm.py` invocation:
```bash
        --output-dir "$OUTPUT_DIR" \
        --victim-model-id "$VICTIM_MODEL_ID" \
```
and change `> "$WORKER_LOG" 2>&1 &` to use the new `WORKER_LOG`. Remove the old `WORKER_LOG="logs/batched_worker_${WORKER_ID}.log"` line.

Note: the worker writes `worker_N.json` to `logs/` itself (Task 5), so the `$OUTPUT_DIR/worker_${WORKER_ID}.json` in `WORKER_OUTPUT` is now only the legacy `--benchmark-output` path — keep it for back-compat but it is no longer the primary location. Set `BENCHMARK_OUTPUT_ARG="$OUTPUT_DIR/worker_${WORKER_ID}.json"` and pass `--benchmark-output "$BENCHMARK_OUTPUT_ARG"` (deprecated alias).

- [ ] **Step 3: Update the merge step**

At lines 127-129, change the merge to read from the resolved `logs/` and write there:
```bash
LOGS_DIR=$(python -c "from experiment.results_layout import resolve_model_id, parse_output_dir, runs_root; r=runs_root('$OUTPUT_DIR','benchmark',resolve_model_id('$VICTIM_MODEL_ID'),parse_output_dir('$OUTPUT_DIR','benchmark')[1]); print(r/'logs')")

python scripts/merge_benchmarks.py \
    --output "$LOGS_DIR" \
    --worker-results "$LOGS_DIR"/worker_*.json
```

- [ ] **Step 4: Shellcheck the script (if available)**

Run: `bash -n hpc/autored_benchmark_4gpu_vllm.sh && echo "syntax ok"`
Expected: `syntax ok`.

- [ ] **Step 5: Commit**

```bash
git add hpc/autored_benchmark_4gpu_vllm.sh
git commit -m "feat(hpc): pass --output-dir/--victim-model-id; worker logs under <root>/logs/

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Migration script — `migrate_results_layout.py` (rename + dedup)

**Files:**
- Create: `scripts/migrate_results_layout.py`
- Test: `scripts/tests/test_migrate_results_layout.py`

**Interfaces:**
- Produces: a CLI with `--migrate` (rename `results/` → `results_old/`), `--dedup` (SHA-256 dedup under `results_old/`), both with `--dry-run`.

- [ ] **Step 1: Write failing tests**

Create `scripts/tests/test_migrate_results_layout.py`:
```python
import json
from pathlib import Path
from scripts.migrate_results_layout import migrate, dedup


def test_migrate_renames_results_to_results_old(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "run_1.json").write_text("{}")
    (tmp_path / "results_old").mkdir()  # pre-existing leftover
    (tmp_path / "results_old" / "leftover.json").write_text("{}")

    migrate(base=tmp_path, dry_run=False)

    assert not (tmp_path / "results").exists()
    assert (tmp_path / "results_old" / "run_1.json").exists()
    assert (tmp_path / "results_old" / "leftover.json").exists()
    assert (tmp_path / "results" / "benchmark").exists()
    assert (tmp_path / "results" / "single").exists()


def test_migrate_dry_run_does_not_touch_disk(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "run_1.json").write_text("{}")
    out = migrate(base=tmp_path, dry_run=True)
    assert (tmp_path / "results").exists()
    assert "would rename" in out.lower() or "dry" in out.lower()


def test_dedup_removes_identical_files(tmp_path):
    d = tmp_path / "results_old"
    d.mkdir()
    payload = json.dumps({"a": 1}, sort_keys=True)
    (d / "a.json").write_text(payload)
    sub = d / "sub"; sub.mkdir()
    (sub / "b.json").write_text(payload)      # identical -> removed
    (d / "c.json").write_text(json.dumps({"a": 2}))  # different -> kept

    report = dedup(base=tmp_path, dry_run=False)
    kept = list((tmp_path / "results_old").rglob("*.json"))
    contents = sorted(f.read_text() for f in kept)
    assert contents.count(payload) == 1
    assert report["removed"] == 1
    assert report["kept"] >= 2


def test_dedup_keeps_lexicographically_first(tmp_path):
    d = tmp_path / "results_old"
    d.mkdir()
    payload = json.dumps({"x": 1}, sort_keys=True)
    (d / "z.json").write_text(payload)
    (d / "a.json").write_text(payload)
    dedup(base=tmp_path, dry_run=False)
    assert (d / "a.json").exists()
    assert not (d / "z.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest scripts/tests/test_migrate_results_layout.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the script**

Create `scripts/migrate_results_layout.py`:
```python
#!/usr/bin/env python3
"""One-time migration for the AutoRed-Final results layout.

Subcommands (positional flags):
  --migrate   rename results/ -> results_old/ (fold existing results_old/), create fresh results/{benchmark,single}/
  --dedup     SHA-256 dedup of *.json under results_old/, keeping lexicographically-first path per hash
  --dry-run   report only, no writes

Both can be combined: --migrate --dedup.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _fresh_results(base: Path) -> None:
    (base / "results" / "benchmark").mkdir(parents=True, exist_ok=True)
    (base / "results" / "single").mkdir(parents=True, exist_ok=True)


def migrate(base: Path, dry_run: bool = False) -> str:
    """Rename results/ -> results_old/, fold any existing results_old/, create fresh results/{benchmark,single}/."""
    results = base / "results"
    results_old = base / "results_old"
    if not results.exists():
        return "no results/ to migrate"
    if dry_run:
        return f"DRY-RUN: would rename {results} -> {results_old} and create fresh results/{{benchmark,single}}/"
    # If results_old exists, merge its contents into the old results/ first so the rename folds them.
    if results_old.exists():
        for p in list(results_old.rglob("*")):
            if p.is_file():
                rel = p.relative_to(results_old)
                dest = results / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.move(str(p), str(dest))
        shutil.rmtree(results_old)
    shutil.move(str(results), str(results_old))
    _fresh_results(base)
    return f"renamed results/ -> results_old/; fresh results/{{benchmark,single}}/ created"


def dedup(base: Path, dry_run: bool = False) -> dict:
    """SHA-256 dedup *.json under results_old/. Keep lexicographically-first path per hash."""
    results_old = base / "results_old"
    hashes: dict[str, list[Path]] = {}
    for f in results_old.rglob("*.json"):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        hashes.setdefault(h, []).append(f)

    removed = 0
    kept = 0
    for h, paths in hashes.items():
        paths.sort()
        kept += 1
        for dup in paths[1:]:
            if dry_run:
                removed += 1
            else:
                dup.unlink()
                removed += 1
    return {"removed": removed, "kept": kept, "dry_run": dry_run}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=".", help="Project root containing results/")
    ap.add_argument("--migrate", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    base = Path(args.base).resolve()
    if args.migrate:
        print(migrate(base, dry_run=args.dry_run))
    if args.dedup:
        print(json.dumps(dedup(base, dry_run=args.dry_run), indent=2))
    if not (args.migrate or args.dedup):
        ap.error("specify --migrate and/or --dedup")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest scripts/tests/test_migrate_results_layout.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/migrate_results_layout.py scripts/tests/test_migrate_results_layout.py
git commit -m "feat(migrate): results->results_old rename + SHA-256 dedup script with tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 10: Update downstream readers

**Files:**
- Modify: `server/file_manager.py` (lines ~108-120)
- Modify: `scripts/analyze_vllm_benchmark.py` (line 191)
- Modify: `scripts/dataset_tools/analyse_and_fill_sheet.py` (line 78)

**Interfaces:** none new — these are internal default-path/glob adjustments.

- [ ] **Step 1: `server/file_manager.py` — walk the new tree**

Read the current `list_*` functions (lines ~108-120). Replace the dated-archive and benchmark-summary listing to walk `results/{benchmark,single}/*/*/{logs,runs}`. New implementation:
```python
from pathlib import Path

RESULTS_ROOT = Path("results")

def list_runs(base: Path = RESULTS_ROOT):
    """Yield (mode, model_id, characteristics, run_json_path) for every run JSON."""
    for mode_dir in (base / "benchmark", base / "single"):
        if not mode_dir.exists():
            continue
        for model_dir in mode_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for chars_dir in model_dir.iterdir():
                if not chars_dir.is_dir():
                    continue
                for run in (chars_dir / "runs").rglob("*.json"):
                    yield mode_dir.name, model_dir.name, chars_dir.name, run

def list_benchmark_summaries(base: Path = RESULTS_ROOT):
    """Yield merged_summary.json paths under results/benchmark/.../logs/."""
    return list((base / "benchmark").rglob("*/logs/merged_summary.json"))
```
Keep the existing function names if other code imports them; if they return a different shape, update the call sites in `server/experiment_server.py` to match. (Read `experiment_server.py` usages of `file_manager` before changing return shapes.)

- [ ] **Step 2: `scripts/analyze_vllm_benchmark.py` default path**

At line 191, change:
```python
    path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmarks/batched_100r_4g/merged_summary.json"
```
to:
```python
    path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark/unknown/batched_100r_4g/logs/merged_summary.json"
```
(Old runs under `results_old/benchmarks/...` remain readable by passing an explicit path.)

- [ ] **Step 3: `analyse_and_fill_sheet.py` exclude `results_old/`**

At line 78, change:
```python
    results_files = glob.glob("results/**/*.json", recursive=True)
```
to:
```python
    results_files = [
        f for f in glob.glob("results/**/*.json", recursive=True)
        if not f.startswith("results_old/")
    ]
```

- [ ] **Step 4: Verify the three files parse**

Run: `.venv/bin/python -c "import ast; [ast.parse(open(p).read()) for p in ['server/file_manager.py','scripts/analyze_vllm_benchmark.py','scripts/dataset_tools/analyse_and_fill_sheet.py']]; print('syntax ok')"`
Expected: `syntax ok`.

- [ ] **Step 5: Commit**

```bash
git add server/file_manager.py scripts/analyze_vllm_benchmark.py scripts/dataset_tools/analyse_and_fill_sheet.py
git commit -m "feat(readers): walk new results tree; exclude results_old/ from globs

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 11: Integration test — tiny benchmark writes the new tree

**Files:**
- Create: `scripts/tests/test_results_layout_integration.py`

**Interfaces:** exercises Tasks 1-5 end-to-end without loading vLLM, by calling `run_benchmark`'s routing logic via a lightweight fake. Because `run_benchmark` requires a real `RedTeamingAgent` and vLLM, this test instead drives `results_layout` directly to assert the *shape* a benchmark would produce, and is the gate that the layout contract is honored.

- [ ] **Step 1: Write the integration test**

Create `scripts/tests/test_results_layout_integration.py`:
```python
import json
from pathlib import Path
from experiment.results_layout import (
    resolve_model_id, parse_output_dir, runs_root, run_filename, single_run_filename,
)


def test_benchmark_tree_shape(tmp_path):
    root = runs_root(None, "benchmark", resolve_model_id("meta-llama/Meta-Llama-3-8B-Instruct"), "chars_4r_2w", base=str(tmp_path))
    # Simulate two rounds, one success one failure, on worker 0
    for sid, rnd, success in [(7, 1, True), (9, 2, False)]:
        f = root / "runs" / ("success" if success else "failed") / run_filename(sid, 0, rnd)
        f.write_text(json.dumps({"experiment": {"run_id": f"run_{sid}"}, "result": {"ground_truth_success": success}}))
    assert (root / "runs" / "success" / "run_7_w0_1.json").exists()
    assert (root / "runs" / "failed" / "run_9_w0_2.json").exists()
    assert not (root / "runs" / "failed" / "run_7_w0_1.json").exists()
    # logs dir exists for worker summaries + merged summary
    (root / "logs" / "worker_0.json").write_text("{}")
    (root / "logs" / "merged_summary.json").write_text("{}")
    assert (root / "logs" / "merged_summary.json").exists()


def test_repeated_scenario_no_collision(tmp_path):
    root = runs_root(None, "benchmark", "m", "c", base=str(tmp_path))
    names = {run_filename(7, 0, r) for r in range(1, 6)}
    assert len(names) == 5  # round tiebreaker keeps all unique


def test_single_tree_shape(tmp_path):
    _, chars = parse_output_dir(None, "single")
    root = runs_root(None, "single", resolve_model_id("meta-llama/Meta-Llama-3-8B-Instruct"), chars, base=str(tmp_path))
    f = root / "runs" / "success" / single_run_filename(42)
    f.write_text(json.dumps({"experiment": {"run_id": "run_42"}}))
    assert f.exists()
    assert (root / "logs").is_dir()
```

- [ ] **Step 2: Run the integration test**

Run: `.venv/bin/python -m pytest scripts/tests/test_results_layout_integration.py -v`
Expected: 3 passed.

- [ ] **Step 3: Run the full results-layout suite**

Run: `.venv/bin/python -m pytest scripts/tests/test_results_layout.py scripts/tests/test_migrate_results_layout.py scripts/tests/test_results_layout_integration.py -v`
Expected: all pass (28 total).

- [ ] **Step 4: Commit**

```bash
git add scripts/tests/test_results_layout_integration.py
git commit -m "test(layout): integration test for benchmark/single tree shape + no-collision

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 12: Run migration and verify final state

**Files:** none (operational)

- [ ] **Step 1: Dry-run the migration**

Run: `.venv/bin/python scripts/migrate_results_layout.py --base . --migrate --dedup --dry-run`
Expected: reports "would rename results/ -> results_old/" and a dedup count; no disk changes. Confirm the count is plausible (thousands of files).

- [ ] **Step 2: Execute the migration**

Run: `.venv/bin/python scripts/migrate_results_layout.py --base . --migrate --dedup`
Expected: `results/` now contains only `benchmark/` and `single/`; old files are under `results_old/`.

- [ ] **Step 3: Verify the top-level structure**

Run: `ls results && echo "---" && ls results_old | head`
Expected: `results/` shows `benchmark single`; `results_old/` shows the old dated/benchmark dirs.

- [ ] **Step 4: Run the layout + migration tests once more against the migrated tree**

Run: `.venv/bin/python -m pytest scripts/tests/test_results_layout.py scripts/tests/test_migrate_results_layout.py scripts/tests/test_results_layout_integration.py -v`
Expected: all pass.

- [ ] **Step 5: Commit any migration-affecting ignore / metadata**

If `results/` is git-tracked, the rename will surface as deletions in `git status`. Do **not** commit the 13k file moves in one go unless the user asks; instead note the state:
```bash
git status --short | head
```
Report the state to the user and stop. Do not stage the bulk rename without explicit approval (per Global Constraints: migration is non-destructive and the user deletes later manually).

---

## Notes for the implementer

- **`run_json['experiment']['run_id']`** still exists (generated in `serialize_run` at line 1829) and is kept inside each run JSON — it's just no longer the *filename*. Filenames use `<scenario_id>_w<worker>_<round>`.
- **`--benchmark-output`** is a deprecated alias; keep it working (Task 5 writes a legacy copy) so old tooling doesn't break immediately, but print a deprecation warning (Task 4).
- **`extractor_benchmark` mode** is routed under the `benchmark` tree (Task 4 sets `_MODE = "benchmark"` for it). If its save path (line ~5082-5098) also needs the new layout, apply the same `runs_root`/`run_filename` pattern there — but only if that mode is still in use; otherwise leave it and flag it.
- **Server path field names**: `worker/experiment_runner.py` reuses the local `success` bool (line 151) and the local `scenario_id` (line 121) — both already in scope at the save block. No need to read fields out of `run_json`.

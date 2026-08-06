#!/usr/bin/env python3
"""GPU-free tests for the benchmark-discovery layer across the new
``results_layout.py`` tree (``results/benchmark/<model>/<chars>/...``) and the
legacy flat layout (``results/benchmarks/<name>/...``).

Covers ``server/file_manager.py`` (list/get benchmarks) and
``scripts/analysis/compare_benchmarks.py`` (discover_benchmark_dirs +
resolve_trace_root_for_benchmark) against synthetic on-disk trees rooted in a
tmp_path, so nothing touches the real ``results/`` directory.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# --- make compare_benchmarks importable without matplotlib/numpy/scipy ----
for _name in ("numpy", "matplotlib", "matplotlib.pyplot"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
sys.modules["matplotlib.pyplot"].plot = lambda *a, **k: None  # type: ignore[attr-defined]

from server import file_manager  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "compare_benchmarks",
    ROOT / "scripts" / "analysis" / "compare_benchmarks.py",
)
compare_benchmarks = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec.loader is not None
_spec.loader.exec_module(compare_benchmarks)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _write_run(path: Path, run_id: str, success: bool, ts: str = "2026-08-06T12:00:00Z"):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": {
            "run_id": run_id,
            "timestamp": ts,
            "scenario_id": run_id.replace("run_", ""),
            "worker_id": 0,
            "benchmark_mode": True,
        },
        "result": {
            "ground_truth_success": success,
            "verified_success": success,
            "total_attempts": 1,
        },
        "scenario": {"access_code": "AC"},
        "models": {"generator": {"name": "g"}, "victim": {"name": "v"}},
        "attempts": [{}],
    }
    path.write_text(json.dumps(payload))


def _write_summary(path: Path, ts: str = "2026-08-06T12:00:00Z", successes: int = 1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "metadata": {"timestamp": ts},
        "total_rounds": successes,
        "total_successes": successes,
        "verified_success": successes,
        "success_rate": float(successes),
        "avg_attempts_on_success": 1.0,
        "top1_success": successes,
        "top3_success": successes,
        "top5_success": successes,
        "extractor_metrics": {},
        "worker_summaries": [],
    }))


@pytest.fixture
def results_tree(tmp_path: Path) -> Path:
    """Build a results/ tree with one nested and one legacy benchmark."""
    results = tmp_path / "results"

    # Nested: results/benchmark/<model>/<chars>/{logs, runs/{success,failed}}
    nested = results / "benchmark" / "org--model" / "chars_A"
    _write_summary(nested / "logs" / "merged_summary.json", successes=2)
    _write_run(nested / "runs" / "success" / "run_7_w0_1.json", "run_7", success=True)
    _write_run(nested / "runs" / "failed" / "run_8_w0_2.json", "run_8", success=False)

    # Legacy: results/benchmarks/<name>/merged_summary.json (+ dated traces)
    legacy = results / "benchmarks" / "legacy_run"
    _write_summary(legacy / "merged_summary.json", ts="2026-07-12T03:00:00Z", successes=1)
    # legacy dated trace archive: results/<date>/<time_id>/run_*.json (one level
    # under the date dir, matching the real on-disk legacy layout)
    day = results / "2026-07-12" / "03-00-00_0"
    _write_run(day / "run_3_w0_1.json", "run_3", success=True, ts="2026-07-12T03:00:00Z")
    # New single-mode tree: results/single/<model>/<chars>/runs/{success,failed}/
    single = results / "single" / "org--model" / "single_chars"
    _write_run(single / "runs" / "success" / "run_42_single.json", "run_42", success=True)
    return results


@pytest.fixture
def fm(results_tree: Path, monkeypatch):
    """file_manager pointed at the synthetic results tree."""
    monkeypatch.setattr(file_manager, "RESULTS_DIR", results_tree)
    monkeypatch.setattr(file_manager, "BENCHMARKS_DIR", results_tree / "benchmarks")
    monkeypatch.setattr(file_manager, "BENCHMARK_DIR", results_tree / "benchmark")
    return file_manager


# --------------------------------------------------------------------------
# file_manager
# --------------------------------------------------------------------------

def test_discover_finds_nested_and_legacy(fm):
    disc = fm._discover_benchmark_dirs()
    ids = {d["benchmark_id"] for d in disc}
    assert "org--model/chars_A" in ids
    assert "legacy_run" in ids


def test_nested_entry_points_at_chars_dir(fm):
    disc = fm._discover_benchmark_dirs()
    nested = next(d for d in disc if d["benchmark_id"] == "org--model/chars_A")
    assert nested["layout"] == "nested"
    assert nested["benchmark_group"] == "org--model"
    assert nested["summary_file"].name == "merged_summary.json"
    assert nested["summary_file"].parent.name == "logs"  # under <chars>/logs/
    assert (nested["benchmark_dir"] / "runs" / "success").exists()


def test_list_benchmarks_includes_both_layouts(fm):
    bs = fm.list_benchmarks()
    by_id = {b["benchmark_id"]: b for b in bs}
    assert "org--model/chars_A" in by_id
    assert "legacy_run" in by_id
    nested = by_id["org--model/chars_A"]
    assert nested["layout"] == "nested"
    assert nested["benchmark_group"] == "org--model"
    # nested exposes its local run count (2) as trace_archive_count
    assert nested["trace_archive_count"] == 2


def test_get_benchmark_nested_slash_id_round_trips(fm):
    got = fm.get_benchmark("org--model/chars_A")
    assert got is not None
    assert got["layout"] == "nested"
    assert got["benchmark_group"] == "org--model"
    assert len(got["trace_runs"]) == 2
    # one pseudo-archive covering both success+failed runs
    assert len(got["trace_archives"]) == 1
    assert got["trace_archives"][0]["run_count"] == 2
    assert got["trace_archives"][0]["success_rate"] == 0.5


def test_get_benchmark_legacy_uses_date_archives(fm):
    got = fm.get_benchmark("legacy_run")
    assert got is not None
    assert got["layout"] == "legacy"
    assert got["benchmark_group"] is None
    # the dated run_*.json is surfaced as a trace archive
    assert any(r["run_id"] == "run_3" for r in got["trace_runs"])


def test_get_benchmark_unknown_returns_none(fm):
    assert fm.get_benchmark("no/such/id") is None
    assert fm.get_benchmark("nope") is None


def test_list_benchmarks_stable_ordering(fm):
    bs = fm.list_benchmarks()
    tss = [b["timestamp"] for b in bs]
    assert tss == sorted(tss, reverse=True)


def test_is_benchmark_artifact_excludes_both_layouts(fm, results_tree: Path):
    from pathlib import Path as P
    # new singular tree (benchmark/) and legacy plural (benchmarks/) both excluded
    assert fm._is_benchmark_artifact(P("results/benchmark/m/c/runs/success/run_7_w0_1.json"))
    assert fm._is_benchmark_artifact(P("results/benchmarks/legacy_run/worker_0.json"))
    # single tree is NOT a benchmark artifact — single runs must surface
    assert not fm._is_benchmark_artifact(P("results/single/m/c/runs/success/run_42_single.json"))
    # a dated single trace is not a benchmark artifact either
    assert not fm._is_benchmark_artifact(P("results/2026-07-12/03-00-00_0/run_3_w0_1.json"))


def test_list_all_runs_recursive_includes_single_excludes_benchmark(fm):
    runs = fm.list_all_runs_recursive()
    ids = {r["run_id"] for r in runs}
    # single-mode new-tree run is listed
    assert "run_42" in ids
    # dated single run is listed
    assert "run_3" in ids
    # benchmark per-round runs (new + legacy) are NOT listed as standalone runs
    nested_dir = fm.RESULTS_DIR / "benchmark" / "org--model" / "chars_A"
    bench_run_ids = {p.stem for p in (nested_dir / "runs").rglob("run_*.json")}
    assert not (ids & bench_run_ids), "benchmark per-round runs leaked into /api/runs/all"



# --------------------------------------------------------------------------
# compare_benchmarks
# --------------------------------------------------------------------------

def test_cb_discover_nested_and_legacy(results_tree: Path):
    # Root = results/benchmarks finds legacy flat + sibling nested.
    disc = compare_benchmarks.discover_benchmark_dirs(str(results_tree / "benchmarks"))
    by_name = {d.name: d for d in disc}
    assert "legacy_run" in by_name
    assert "chars_A" in by_name  # nested <chars> dir
    chars = by_name["chars_A"]
    assert (chars / "logs" / "merged_summary.json").exists()


def test_cb_discover_under_results_root(results_tree: Path):
    disc = compare_benchmarks.discover_benchmark_dirs(str(results_tree))
    by_name = {d.name: d for d in disc}
    assert "chars_A" in by_name


def test_cb_resolve_trace_root_nested_prefers_runs(results_tree: Path):
    chars = results_tree / "benchmark" / "org--model" / "chars_A"
    tr = compare_benchmarks.resolve_trace_root_for_benchmark(chars, str(results_tree))
    assert tr is not None
    assert tr.name == "runs"
    assert len(list(tr.rglob("run_*.json"))) == 2


def test_cb_resolve_trace_root_legacy_uses_date(results_tree: Path):
    legacy = results_tree / "benchmarks" / "legacy_run"
    tr = compare_benchmarks.resolve_trace_root_for_benchmark(legacy, str(results_tree))
    assert tr is not None
    assert tr.name == "2026-07-12"
    assert len(list(tr.rglob("run_*.json"))) == 1


def test_cb_resolve_trace_root_no_runs_returns_none(results_tree: Path, tmp_path: Path):
    empty = tmp_path / "empty_bm"
    (empty / "logs").mkdir(parents=True)
    _write_summary(empty / "logs" / "merged_summary.json")
    # no runs/ dir and no matching date tree -> None
    tr = compare_benchmarks.resolve_trace_root_for_benchmark(empty, str(results_tree))
    assert tr is None

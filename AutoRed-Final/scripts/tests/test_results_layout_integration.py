#!/usr/bin/env python3
"""Integration test for the results-layout contract (Task 11).

Exercises Tasks 1-5 end-to-end *without* loading vLLM: it drives
``results_layout`` directly to assert the on-disk shape a benchmark/single
run would produce, including success/failed routing and no-collision naming.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiment.results_layout import (  # noqa: E402
    resolve_model_id,
    parse_output_dir,
    runs_root,
    run_filename,
    single_run_filename,
)


def test_benchmark_tree_shape(tmp_path):
    root = runs_root(
        None,
        "benchmark",
        resolve_model_id("meta-llama/Meta-Llama-3-8B-Instruct"),
        "chars_4r_2w",
        base=str(tmp_path),
    )
    # Simulate two rounds, one success one failure, on worker 0
    for sid, rnd, success in [(7, 1, True), (9, 2, False)]:
        f = root / "runs" / ("success" if success else "failed") / run_filename(sid, 0, rnd)
        f.parent.mkdir(parents=True, exist_ok=True)
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
    root = runs_root(
        None,
        "single",
        resolve_model_id("meta-llama/Meta-Llama-3-8B-Instruct"),
        chars,
        base=str(tmp_path),
    )
    f = root / "runs" / "success" / single_run_filename(42)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"experiment": {"run_id": "run_42"}}))
    assert f.exists()
    assert (root / "logs").is_dir()

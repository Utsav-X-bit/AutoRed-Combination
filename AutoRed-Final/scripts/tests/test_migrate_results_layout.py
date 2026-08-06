#!/usr/bin/env python3
"""GPU-free tests for scripts/migrate_results_layout.py (Task 9).

Imports the migration module by path (scripts/ is not a package) using the
same importlib pattern used elsewhere in this test suite.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "migrate_results_layout", ROOT / "scripts" / "migrate_results_layout.py"
)
migrate_results_layout = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec.loader is not None
_spec.loader.exec_module(migrate_results_layout)  # type: ignore[arg-type]
migrate = migrate_results_layout.migrate
dedup = migrate_results_layout.dedup


def test_migrate_renames_results_to_results_old(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "run_1.json").write_text("{}")
    (tmp_path / "results_old").mkdir()  # pre-existing leftover
    (tmp_path / "results_old" / "leftover.json").write_text("{}")

    migrate(base=tmp_path, dry_run=False)

    assert not (tmp_path / "results" / "run_1.json").exists()
    assert (tmp_path / "results_old" / "run_1.json").exists()
    assert (tmp_path / "results_old" / "leftover.json").exists()
    # fresh tree created
    assert (tmp_path / "results" / "benchmark").exists()
    assert (tmp_path / "results" / "single").exists()


def test_migrate_dry_run_does_not_touch_disk(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "run_1.json").write_text("{}")
    out = migrate(base=tmp_path, dry_run=True)
    assert (tmp_path / "results").exists()
    assert (tmp_path / "results" / "run_1.json").exists()
    assert "would rename" in out.lower() or "dry" in out.lower()


def test_migrate_no_results_is_noop(tmp_path):
    out = migrate(base=tmp_path, dry_run=False)
    assert out == "no results/ to migrate"


def test_dedup_removes_identical_files(tmp_path):
    d = tmp_path / "results_old"
    d.mkdir()
    payload = json.dumps({"a": 1}, sort_keys=True)
    (d / "a.json").write_text(payload)
    sub = d / "sub"
    sub.mkdir()
    (sub / "b.json").write_text(payload)  # identical -> removed
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


def test_dedup_dry_run_counts_only(tmp_path):
    d = tmp_path / "results_old"
    d.mkdir()
    payload = json.dumps({"a": 1}, sort_keys=True)
    (d / "a.json").write_text(payload)
    (d / "b.json").write_text(payload)
    report = dedup(base=tmp_path, dry_run=True)
    # nothing actually removed
    assert (d / "a.json").exists()
    assert (d / "b.json").exists()
    assert report["removed"] == 1
    assert report["dry_run"] is True

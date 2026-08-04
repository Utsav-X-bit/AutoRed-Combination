#!/usr/bin/env python3
"""
One-time migration of benchmark storage from the flat layout to the Change 3
two-level hierarchy.

Flat (legacy):
    results/benchmarks/{name}_{YYYY-MM-DD}_{HH-MM-SS}_{N}g/
        merged_summary.json
        worker_*.json
    results/{YYYY-MM-DD}/{victim}/{HH-MM-SS_us}/run_*.json   (traces, scattered)

Nested (target):
    results/benchmarks/{name}/
        {YYYY-MM-DD}_{HH-MM-SS}_{N}g/
            merged_summary.json
            worker_*.json
            runs/
                run_*.json

The folder-name timestamp is the benchmark script START. merged_summary's
metadata.timestamp is the benchmark FINISH. Per-run traces were written while
vLLM was live, so a run belongs to a benchmark when its victim matches and its
own timestamp falls within [start, finish]. We allow a generous lead/lag
because the folder start precedes model load and the finish may trail the last
run write.

Properties:
  - Dry-run capable (--dry-run): prints the plan, changes nothing.
  - Idempotent: a folder already nested (no flat suffix) is skipped; a target
    that already exists is skipped; traces already in runs/ are skipped.
  - Non-destructive to source run files: COPIES (default) or MOVES (--move)
    them into runs/. Default copy keeps the legacy date tree intact so the
    backend's date-based fallback still works until the user is satisfied.
  - smoke/ is always skipped.
  - Single-mode traces (results/{date}/{model}/{time}/ with benchmark_mode not
    True) are never touched — those stay put per the plan's explicit requirement.

Usage:
    python scripts/migrate_benchmarks_layout.py --dry-run
    python scripts/migrate_benchmarks_layout.py            # copy traces into runs/
    python scripts/migrate_benchmarks_layout.py --move     # move traces (frees space)
    python scripts/migrate_benchmarks_layout.py --benchmarks-dir results/benchmarks

Run from the AutoRed-Final project root. Back up results/benchmarks/ first.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Flat run suffix: _YYYY-MM-DD_HH-MM-SS_Ng at the end of a folder name.
FLAT_SUFFIX_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_(\d+)g$")

# Folders that are never treated as flat benchmark runs.
SKIP_NAMES = {"smoke"}

# How far before the folder-start timestamp a run may have been written and
# still count (covers runs started during model warmup). Seconds.
START_MARGIN_S = 30 * 60  # 30 min
# How far after the metadata-finish timestamp a run may have been written.
# Covers runs whose timestamp was captured before the flush of the summary.
END_MARGIN_S = 30 * 60  # 30 min


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _folder_timestamp_to_iso(date: str, time: str) -> Optional[datetime]:
    """'2026-07-31', '23-16-07' -> datetime."""
    try:
        return datetime.fromisoformat(f"{date}T{time.replace('-', ':')}")
    except Exception:
        return None


def _model_dir_name(model_name: str) -> str:
    """Mirror server/file_manager._model_dir_name: slug a HF model id."""
    return model_name.replace("/", "--").replace(" ", "-")


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _victim_from_benchmark(bench_dir: Path) -> Optional[str]:
    """Read the victim model name from worker_0.json (merged_summary lacks it
    on older runs)."""
    for worker in sorted(bench_dir.glob("worker_*.json")):
        data = _load_json(worker) or {}
        victim = data.get("models", {}).get("victim", {}).get("name")
        if victim:
            return victim
    summary = _load_json(bench_dir / "merged_summary.json") or {}
    return (
        summary.get("metadata", {}).get("victim_model")
        or summary.get("models", {}).get("victim", {}).get("name")
    )


def _finish_from_benchmark(bench_dir: Path) -> Optional[datetime]:
    """merged_summary.metadata.timestamp is the benchmark finish time."""
    summary = _load_json(bench_dir / "merged_summary.json") or {}
    return _parse_iso(summary.get("metadata", {}).get("timestamp"))


def _collect_run_files(results_dir: Path, victim_name: str,
                       start: datetime, finish: Optional[datetime]
                       ) -> list[Path]:
    """Find legacy run_*.json traces for this victim whose timestamp falls in
    the [start - margin, finish + margin] window. Returns run files (not
    archive dirs) so the caller can copy/move them directly."""
    if not victim_name:
        return []
    victim_dir_name = _model_dir_name(victim_name)
    lo = start.timestamp() - START_MARGIN_S
    hi = (finish.timestamp() + END_MARGIN_S) if finish else None

    matches: list[Path] = []
    # Walk results/<date>/<victim>/<time>/run_*.json
    for day_dir in sorted(results_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        # Only consider days on/around the benchmark date.
        try:
            day_dt = datetime.fromisoformat(day_dir.name)
        except ValueError:
            continue
        if abs((day_dt.date() - start.date()).days) > 1:
            continue
        for victim_sub in sorted(day_dir.iterdir()):
            if not victim_sub.is_dir():
                continue
            if victim_sub.name != victim_dir_name:
                continue
            for run_file in sorted(victim_sub.rglob("run_*.json")):
                meta = _load_json(run_file) or {}
                ex = meta.get("experiment", {})
                # Only migrate benchmark-mode traces; single-mode stays put.
                if not ex.get("benchmark_mode", False):
                    continue
                run_ts = _parse_iso(ex.get("timestamp"))
                if run_ts is None:
                    continue
                t = run_ts.timestamp()
                if t < lo:
                    continue
                if hi is not None and t > hi:
                    continue
                matches.append(run_file)
    return matches


def _plan_migration(benchmarks_dir: Path, results_dir: Path) -> list[dict]:
    """Return a list of migration actions. Each action describes one flat
    benchmark folder and the run files associated with it."""
    actions: list[dict] = []
    if not benchmarks_dir.exists():
        return actions

    for child in sorted(benchmarks_dir.iterdir()):
        if not child.is_dir() or child.name in SKIP_NAMES:
            continue
        m = FLAT_SUFFIX_RE.search(child.name)
        if not m:
            # Already nested or non-standard; leave alone.
            continue
        date_str, time_str, ng = m.group(1), m.group(2), m.group(3)
        start = _folder_timestamp_to_iso(date_str, time_str)
        if start is None:
            continue
        # Split point: the date is the last _YYYY-MM-DD_... marker.
        group_name = child.name[: m.start()]
        run_suffix = child.name[m.start() + 1 :]  # strip leading '_'

        target_group = benchmarks_dir / group_name
        target_run = target_group / run_suffix

        victim = _victim_from_benchmark(child)
        finish = _finish_from_benchmark(child)
        run_files: list[Path] = []
        if results_dir.exists():
            run_files = _collect_run_files(results_dir, victim or "", start, finish)

        actions.append({
            "src": child,
            "group_name": group_name,
            "run_suffix": run_suffix,
            "target_group": target_group,
            "target_run": target_run,
            "victim": victim,
            "start": start,
            "finish": finish,
            "run_files": run_files,
        })
    return actions


def _apply_migration(actions: list[dict], *, move: bool, dry_run: bool) -> dict:
    """Execute the migration. Returns a stats dict."""
    stats = {"folders_nested": 0, "folders_skipped_existing": 0,
             "run_files_copied": 0, "run_files_moved": 0,
             "run_files_skipped_existing": 0, "errors": 0}

    for act in actions:
        src: Path = act["src"]
        target_run: Path = act["target_run"]

        if target_run.exists():
            print(f"  ↷ skip (target exists): {src.name} -> "
                  f"{act['group_name']}/{act['run_suffix']}")
            stats["folders_skipped_existing"] += 1
        else:
            target_run.parent.mkdir(parents=True, exist_ok=True)
            if dry_run:
                print(f"  ⟶ would mv: {src.name} -> "
                      f"{act['group_name']}/{act['run_suffix']}")
            else:
                shutil.move(str(src), str(target_run))
                print(f"  ✓ nested: {src.name} -> "
                      f"{act['group_name']}/{act['run_suffix']}")
            stats["folders_nested"] += 1

        # Migrate run traces into <target_run>/runs/.
        runs_dir = target_run / "runs"
        if act["run_files"]:
            if not dry_run:
                runs_dir.mkdir(parents=True, exist_ok=True)
            for run_file in act["run_files"]:
                dest = runs_dir / run_file.name
                if dest.exists():
                    print(f"    ↷ skip (run exists): {run_file.name}")
                    stats["run_files_skipped_existing"] += 1
                    continue
                if dry_run:
                    verb = "would move" if move else "would copy"
                    print(f"    ⟶ {verb}: {run_file} -> {dest}")
                else:
                    if move:
                        shutil.move(str(run_file), str(dest))
                        stats["run_files_moved"] += 1
                    else:
                        shutil.copy2(str(run_file), str(dest))
                        stats["run_files_copied"] += 1
                    print(f"    ✓ {'moved' if move else 'copied'}: "
                          f"{run_file.name}")
        elif not dry_run and target_run.exists():
            # Create an empty runs/ marker so downstream code knows where to look
            # even when no legacy traces were found.
            runs_dir.mkdir(parents=True, exist_ok=True)

    return stats


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Migrate benchmark storage to the two-level (group/run) layout."
    )
    ap.add_argument("--benchmarks-dir", default="results/benchmarks",
                    help="Benchmark root (default: results/benchmarks)")
    ap.add_argument("--results-dir", default="results",
                    help="Run-trace root (default: results)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the migration plan; change nothing.")
    ap.add_argument("--move", action="store_true",
                    help="Move run files into runs/ instead of copying. "
                         "Default (copy) preserves the legacy date tree.")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    benchmarks_dir = (repo_root / args.benchmarks_dir).resolve()
    results_dir = (repo_root / args.results_dir).resolve()

    if not benchmarks_dir.exists():
        print(f"error: benchmarks dir not found: {benchmarks_dir}")
        return 2

    print(f"benchmarks dir : {benchmarks_dir}")
    print(f"results dir    : {results_dir}")
    print(f"mode           : {'DRY-RUN' if args.dry_run else 'APPLY'} "
          f"({'move' if args.move else 'copy'} run files)")
    print()

    actions = _plan_migration(benchmarks_dir, results_dir)
    if not actions:
        print("No flat benchmark folders to migrate.")
        return 0

    print(f"Found {len(actions)} flat benchmark folder(s):\n")
    for act in actions:
        victim = act["victim"] or "?"
        finish = act["finish"].isoformat() if act["finish"] else "?"
        n_runs = len(act["run_files"])
        print(f"  {act['src'].name}")
        print(f"      -> {act['group_name']}/{act['run_suffix']}")
        print(f"      victim={victim}  start={act['start'].isoformat()}  "
              f"finish={finish}  runs={n_runs}")
    print()

    stats = _apply_migration(actions, move=args.move, dry_run=args.dry_run)

    print("\nSummary:")
    print(f"  folders nested        : {stats['folders_nested']}")
    print(f"  folders skipped       : {stats['folders_skipped_existing']}")
    print(f"  run files copied      : {stats['run_files_copied']}")
    print(f"  run files moved       : {stats['run_files_moved']}")
    print(f"  run files skipped     : {stats['run_files_skipped_existing']}")
    print(f"  errors                : {stats['errors']}")
    if args.dry_run:
        print("\n(dry-run: nothing was changed)")
    else:
        print("\nMigration applied. Verify with the backend "
              "(/api/benchmarks) before removing any legacy date-tree traces.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

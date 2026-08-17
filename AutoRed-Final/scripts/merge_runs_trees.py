#!/usr/bin/env python3
"""Merge per-attempt runs/ trees from two node benchmarks into one combined tree.

Node-1 runs already use GLOBAL round indices (0..13023) so they are copied
unchanged. Node-2 runs use NODE-LOCAL round indices (0..13022) that collide
with Node-1's, so every Node-2 run is re-keyed:

  - filename round  r      -> r + ROUND_OFFSET   (ROUND_OFFSET = Node-1's total)
  - filename round+1 (r+1) -> (r+1) + ROUND_OFFSET
  - filename worker w0..3  -> w{worker + WORKER_OFFSET}  (WORKER_OFFSET = Node-1's worker count = 4)
  - experiment.scenario_id (string) -> str(int(sid) + ROUND_OFFSET)

The result is a single nested layout that looks exactly like one benchmark
ran the whole dataset, so the UI per-round drill-down (which reads
experiment.scenario_id) works on the combined tree.

Usage:
  python3 merge_runs_trees.py \
      --node1-dir <M1 nested dir> \
      --node2-dir <M2 nested dir> \
      --node1-rounds 13024 \
      --node1-workers 4 \
      --output-dir <combined nested dir>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# run_<round>_w<worker>_<round+1>.json
_FN_RE = re.compile(r"^run_(\d+)_w(\d+)_(\d+)\.json$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node1-dir", required=True, type=Path,
                   help="Node-1 nested results dir (runs already global)")
    p.add_argument("--node2-dir", required=True, type=Path,
                   help="Node-2 nested results dir (runs node-local, need offset)")
    p.add_argument("--node1-rounds", required=True, type=int,
                   help="Number of rounds Node-1 covered (= ROUND_OFFSET)")
    p.add_argument("--node1-workers", required=True, type=int,
                   help="Number of workers Node-1 used (= WORKER_OFFSET)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Combined nested output dir")
    p.add_argument("--node2-skip-workers", default="",
                   help="Comma-separated node-local worker ids to skip when "
                        "copying Node-2 runs (e.g. '0' when Node-2's worker_0 "
                        "failed and its partial runs must be excluded).")
    return p.parse_args()


def _copy_node1(src: Path, out: Path) -> tuple[int, int]:
    """Copy Node-1 runs unchanged. Returns (n_success, n_failed)."""
    ns = nf = 0
    for sub, counter in (("success", "ns"), ("failed", "nf")):
        sdir = src / "runs" / sub
        if not sdir.is_dir():
            print(f"  WARN: {sdir} missing, skipping", file=sys.stderr)
            continue
        ddir = out / "runs" / sub
        ddir.mkdir(parents=True, exist_ok=True)
        i = 0
        for fn in os.listdir(sdir):
            if not fn.endswith(".json"):
                continue
            shutil.copy2(sdir / fn, ddir / fn)
            i += 1
        if counter == "ns":
            ns = i
        else:
            nf = i
        print(f"  Node-1 {sub}: copied {i} runs")
    return ns, nf


def _rekey_node2(src: Path, out: Path, r_off: int, w_off: int,
                 skip_workers: set[int] | None = None) -> tuple[int, int]:
    """Re-key and copy Node-2 runs with round+worker offset. Returns (n_success, n_failed)."""
    skip_workers = skip_workers or set()
    ns = nf = 0
    skipped_w = 0
    for sub, counter in (("success", "ns"), ("failed", "nf")):
        sdir = src / "runs" / sub
        if not sdir.is_dir():
            print(f"  WARN: {sdir} missing, skipping", file=sys.stderr)
            continue
        ddir = out / "runs" / sub
        ddir.mkdir(parents=True, exist_ok=True)
        i = 0
        skip_bad = 0
        for fn in os.listdir(sdir):
            if not fn.endswith(".json"):
                continue
            m = _FN_RE.match(fn)
            if not m:
                skip_bad += 1
                continue
            r, w, r1 = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if w in skip_workers:
                skipped_w += 1
                continue
            nr, nw, nr1 = r + r_off, w + w_off, r1 + r_off
            new_fn = f"run_{nr}_w{nw}_{nr1}.json"
            sp = sdir / fn
            dp = ddir / new_fn
            # load, re-key scenario_id, write
            try:
                with open(sp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception as e:
                skip_bad += 1
                print(f"  WARN: bad JSON {fn}: {e}", file=sys.stderr)
                continue
            exp = obj.get("experiment", {})
            sid = exp.get("scenario_id")
            if sid is not None:
                try:
                    exp["scenario_id"] = str(int(sid) + r_off)
                except (ValueError, TypeError):
                    pass  # non-numeric scenario_id: leave as-is
            with open(dp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
            i += 1
        if counter == "ns":
            ns = i
        else:
            nf = i
        print(f"  Node-2 {sub}: re-keyed {i} runs"
              + (f" (skipped {skip_bad} non-matching)" if skip_bad else "")
              + (f" (skipped {skipped_w} runs from excluded worker(s))" if skipped_w else ""))
    return ns, nf


def _copy_logs_and_summary(src1: Path, src2: Path, out: Path,
                           flat_combined: Path | None) -> None:
    """Copy the merged_summary.json (from flat combined dir if given) into logs/.

    The per-worker log files (.log) and the original worker_*.json are NOT
    merged here (merge_node_benchmarks.py handles the worker-summary merge).
    We only ensure logs/merged_summary.json exists so the UI discovery path
    finds the combined nested dir.
    """
    ldir = out / "logs"
    ldir.mkdir(parents=True, exist_ok=True)
    # prefer the already-merged flat summary (authoritative combined numbers)
    if flat_combined and (flat_combined / "merged_summary.json").is_file():
        shutil.copy2(flat_combined / "merged_summary.json",
                     ldir / "merged_summary.json")
        print(f"  logs/merged_summary.json <- flat combined ({flat_combined.name})")
        # also bring the 8 re-keyed worker jsons for completeness
        for wf in flat_combined.glob("worker_*.json"):
            shutil.copy2(wf, ldir / wf.name)
        print(f"  logs/worker_*.json <- flat combined ({len(list(flat_combined.glob('worker_*.json')))} files)")
    elif (src1 / "logs" / "merged_summary.json").is_file():
        # fallback: just copy Node-1's (incomplete) — not ideal
        shutil.copy2(src1 / "logs" / "merged_summary.json",
                     ldir / "merged_summary.json")
        print(f"  WARN: logs/merged_summary.json <- Node-1 only (provide --flat-combined for full merge)")
    else:
        print("  WARN: no merged_summary.json source found")


def main() -> int:
    a = parse_args()
    r_off = a.node1_rounds
    w_off = a.node1_workers
    a.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Merging runs trees -> {a.output_dir}")
    print(f"  ROUND_OFFSET = {r_off}  WORKER_OFFSET = {w_off}")

    if not (a.node1_dir / "runs").is_dir():
        print(f"ERROR: {a.node1_dir}/runs not found", file=sys.stderr)
        return 1
    if not (a.node2_dir / "runs").is_dir():
        print(f"ERROR: {a.node2_dir}/runs not found", file=sys.stderr)
        return 1

    skip_w = {int(x) for x in a.node2_skip_workers.split(",") if x.strip()}
    s1, f1 = _copy_node1(a.node1_dir, a.output_dir)
    s2, f2 = _rekey_node2(a.node2_dir, a.output_dir, r_off, w_off, skip_w)

    print()
    print(f"  combined success = {s1}+{s2} = {s1 + s2}")
    print(f"  combined failed  = {f1}+{f2} = {f1 + f2}")
    print(f"  combined total   = {s1 + s2 + f1 + f2}")

    # logs / merged_summary
    p = a.__class__  # not used
    flat = None
    # auto-detect flat combined dir sibling
    cand = a.output_dir.name.replace("results_benchmarks_", "")
    flat_cand = a.output_dir.parents[1] / "benchmarks" / cand
    if flat_cand.is_dir():
        flat = flat_cand
    _copy_logs_and_summary(a.node1_dir, a.node2_dir, a.output_dir, flat)

    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

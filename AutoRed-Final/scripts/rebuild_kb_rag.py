#!/usr/bin/env python3
"""Post-benchmark single-process rebuild of the KB, oracle, and RAG index.

Why this exists
---------------
In multi-worker benchmarks each worker is a separate OS process that sees only
its own scenario slice, and concurrent writes to the shared FAISS index /
SQLite DB / strategy KB JSON are unsafe. So ``KBUpdater.update_after_benchmark``
intentionally NO-OPs the rebuild when ``num_workers > 1`` (see
``experiment/kb_updater.py:257-266``) and tells the user to run the rebuild
manually after merging.

This script IS that manual rebuild — packaged as one callable so the launcher
(``hpc/autored_benchmark_4gpu_vllm.sh``) can fire it once after the merge step,
and so the user can run it by hand::

    python3 -m scripts.rebuild_kb_rag [--data-dir data] [--results-dir results]
                                      [--only strategy|oracle|rag] [--quiet]

It calls the three builders in order, each in its own try/except so a hiccup in
one never aborts the others (matching the non-fatal contract of the launcher's
analysis step). Runs SINGLE-PROCESS — only invoke after all workers have exited
and their results are merged.

The builders themselves (the model dimension, Tier-1/2/3 labeling, content-hash
dedup, staleness downweighting, diversity caps) live in
``scripts/dataset_tools/build_strategy_knowledge_base.py``,
``scripts/dataset_tools/mine_strategy_transitions.py``, and
``scripts/dataset_tools/build_rag_index.py``. This file only orchestrates them.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Default store locations — kept in sync with KBUpdater.__init__ paths.
DEFAULT_DATA_DIR = "data"
DEFAULT_RESULTS_DIR = "results"


def _rebuild_strategy_kb(data_dir: Path, verbose: bool) -> dict | None:
    """Rebuild data/strategy_knowledge_base.json from the success/failure JSONL."""
    from scripts.dataset_tools.build_strategy_knowledge_base import build_strategy_kb

    successes = data_dir / "autored_successes_v1.jsonl"
    failures = data_dir / "autored_failures_v1.jsonl"
    output = data_dir / "strategy_knowledge_base.json"
    return build_strategy_kb(
        success_data_path=str(successes),
        failure_data_path=str(failures),
        output_path=str(output),
        data_dir=str(data_dir),
        verbose=verbose,
    )


def _rebuild_oracle(results_dir: Path, data_dir: Path, verbose: bool) -> dict | None:
    """Rebuild data/oracle_rules.json by mining strategy transitions from run JSONs.

    ``build_oracle`` globs ``run_*.json`` directly under each results dir it is
    given. AutoRed's benchmark layout nests run JSONs under
    ``results/benchmark/<model>/runs/{success,failed}/run_*.json``, so we pass
    every ``runs/`` subtree that exists rather than the bare ``results/`` root.
    """
    from scripts.dataset_tools.mine_strategy_transitions import build_oracle

    output = data_dir / "oracle_rules.json"

    # Collect directories that actually contain run_*.json files. Walk the
    # benchmark layout (results/benchmark/<model>/.../runs/) and also accept a
    # flat results/ dir for non-benchmark (single-run) layouts.
    run_dirs: list[str] = []
    if results_dir.is_dir():
        # Deep-glob for run_*.json, then take each unique parent dir.
        seen: set[str] = set()
        for run_file in results_dir.rglob("run_*.json"):
            parent = str(run_file.parent)
            if parent not in seen:
                seen.add(parent)
                run_dirs.append(parent)
        # Fallback: if no nested run_*.json was found, still hand build_oracle
        # the bare results dir (it will just report 0 runs).
        if not run_dirs:
            run_dirs = [str(results_dir)]

    if verbose:
        print(f"[rebuild] oracle: scanning {len(run_dirs)} run dir(s) under {results_dir}")

    # build_oracle returns None (writes JSON to output_file); wrap its counts.
    build_oracle(run_dirs, str(output))

    # build_oracle doesn't return a summary dict, so synthesize a minimal one
    # from the written file so the caller has a uniform shape.
    summary: dict = {"oracle_rules_path": str(output)}
    try:
        with open(output, "r", encoding="utf-8") as f:
            rules = json.load(f)
        summary["best_first_entries"] = len(rules.get("best_first", {}))
        summary["transition_entries"] = len(rules.get("transitions", {}))
    except Exception:
        pass
    return summary


def _rebuild_rag(data_dir: Path, verbose: bool) -> dict | None:
    """Rebuild data/rag/ FAISS index + metadata from Tier-1 successes."""
    from scripts.dataset_tools.build_rag_index import build_rag_index

    successes = data_dir / "autored_successes_v1.jsonl"
    output_dir = data_dir / "rag"
    return build_rag_index(
        successes_path=str(successes),
        output_dir=str(output_dir),
        data_dir=str(data_dir),
        verbose=verbose,
    )


# Map --only names to (label, callable) so we can run a subset.
_BUILDERS = {
    "strategy": ("Strategy KB", _rebuild_strategy_kb),
    "oracle": ("Oracle rules", _rebuild_oracle),
    "rag": ("RAG index", _rebuild_rag),
}


def rebuild(data_dir: str = DEFAULT_DATA_DIR,
            results_dir: str = DEFAULT_RESULTS_DIR,
            only: str | None = None,
            verbose: bool = True) -> dict:
    """Run the three builders in order, each non-fatal. Returns a per-builder report.

    Args:
        data_dir: Directory containing autored_successes_v1.jsonl, the SQLite
            DB, and the output stores (strategy_knowledge_base.json, oracle_rules.json, rag/).
        results_dir: Root of the benchmark run JSONs (mined for oracle transitions).
        only: If set, run only that builder ("strategy" | "oracle" | "rag").
        verbose: Pass through to builders for progress output.
    """
    data_p = Path(data_dir)
    results_p = Path(results_dir)
    report: dict[str, object] = {"data_dir": str(data_p), "results_dir": str(results_p)}

    order = ["strategy", "oracle", "rag"] if only is None else [only]
    for key in order:
        if key not in _BUILDERS:
            print(f"[rebuild] unknown builder {key!r}; valid: {sorted(_BUILDERS)}",
                  file=sys.stderr)
            report[key] = {"ok": False, "error": f"unknown builder {key!r}"}
            continue
        label, fn = _BUILDERS[key]
        if verbose:
            print(f"\n[rebuild] === {label} ===")
        try:
            if key == "oracle":
                summary = fn(results_p, data_p, verbose)
            else:
                summary = fn(data_p, verbose)
            report[key] = {"ok": True, "summary": summary}
            if verbose:
                print(f"[rebuild] {label}: OK  {summary if summary else ''}")
        except FileNotFoundError as exc:
            # A missing input file is a soft skip (e.g. no successes JSONL yet on
            # a brand-new project). Don't treat it as a hard failure.
            report[key] = {"ok": False, "skipped": True, "error": f"{exc.filename} missing"}
            if verbose:
                print(f"[rebuild] {label}: skipped ({exc.filename} missing)")
        except ImportError as exc:
            # faiss / sentence_transformers missing in this venv — soft skip RAG.
            report[key] = {"ok": False, "skipped": True, "error": f"dependency missing: {exc}"}
            if verbose:
                print(f"[rebuild] {label}: skipped (dependency missing: {exc})")
        except Exception as exc:
            report[key] = {"ok": False, "error": str(exc), "trace": traceback.format_exc()}
            # Non-fatal: print and continue to the next builder.
            if verbose:
                print(f"[rebuild] {label}: FAILED — {exc}", file=sys.stderr)
                traceback.print_exc()

    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Single-process post-benchmark rebuild of KB, oracle, and RAG index. "
                    "Run AFTER all workers exit and results are merged."
    )
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help=f"Directory with the JSONL stores + outputs (default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                    help=f"Root of benchmark run JSONs for oracle mining (default: {DEFAULT_RESULTS_DIR})")
    ap.add_argument("--only", choices=sorted(_BUILDERS),
                    help="Run only one builder (strategy | oracle | rag). Default: all three.")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-builder progress output.")
    args = ap.parse_args()

    verbose = not args.quiet
    report = rebuild(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        only=args.only,
        verbose=verbose,
    )

    # Emit the report as JSON on stdout's tail (builders' own tqdm/prints go
    # above it) so a calling process can parse it if desired.
    print("\n[rebuild] summary:")
    print(json.dumps(report, indent=2, default=str))

    # Exit 0 unless something actually raised (soft skips don't count). This
    # mirrors the launcher's non-fatal contract: a rebuild hiccup must never
    # mask a successful benchmark, so we only fail on a hard error in ALL
    # requested builders.
    ran = [r for k, r in report.items() if k in _BUILDERS] if args.only is None \
          else [report.get(args.only)]
    hard_failures = [r for r in ran if isinstance(r, dict) and not r.get("ok") and not r.get("skipped")]
    return 1 if hard_failures and not any(r.get("ok") for r in ran) else 0


if __name__ == "__main__":
    raise SystemExit(main())

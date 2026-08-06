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

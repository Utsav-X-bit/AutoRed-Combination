#!/usr/bin/env python3
"""
Stage F1+F2: Clean v1 Report + Low-Quality Sample Inspection
=============================================================
Generates a comprehensive report for generator_clean_v1 and
extracts the lowest-quality samples for manual review.

Usage:
    python scripts/dataset_tools/clean_v1_report.py
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

# ─── Paths ──────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CLEAN_FILE = OUTPUT_DIR / "tensortrust_clean_v1.jsonl"

ATTACK_TYPES = [
    "instruction_leak",
    "trigger_phrase",
    "roleplay",
    "translation",
    "encoding",
    "summarization",
    "exception_discovery",
    "hypothetical",
    "ignore_previous",
    "formatting",
    "code_conversion",
    "other",
]


def load_clean() -> list[dict]:
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def truncate(text: str, max_len: int = 120) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit("\n", 1)[0] + "..."


def main():
    samples = load_clean()
    print(f"[REPORT] Loaded {len(samples)} clean samples\n")

    # ── Per-type breakdown ──
    by_type = defaultdict(list)
    for s in samples:
        by_type[s["attack_type"]].append(s)

    # ── Report structure ──
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(samples),
        "type_breakdown": {},
        "quality_distribution": {},
        "split_distribution": {},
        "examples_per_type": {},
    }

    print("=" * 70)
    print("  GENERATOR_CLEAN_V1 REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Type Distribution ──
    print(f"\n{'─' * 50}")
    print("  TYPE DISTRIBUTION")
    print(f"{'─' * 50}")
    print(f"\n  {'Type':<25} {'Count':>6} {'%':>6} {'Mean Score':>11} {'Min':>4} {'Max':>4}")
    print(f"  {'─' * 25} {'─' * 6} {'─' * 6} {'─' * 11} {'─' * 4} {'─' * 4}")

    for t in ATTACK_TYPES:
        group = by_type.get(t, [])
        if not group:
            print(f"  {t:<25} {0:>6} {0:>5.0f}% {'N/A':>11} {'-':>4} {'-':>4}")
            report["type_breakdown"][t] = {"count": 0, "pct": 0, "mean_score": None}
            continue

        scores = [s["quality_score"] for s in group]
        mean_s = round(np.mean(scores), 1)
        min_s = min(scores)
        max_s = max(scores)
        pct = round(len(group) / len(samples) * 100, 1)

        print(f"  {t:<25} {len(group):>6} {pct:>5.1f}% {mean_s:>11} {min_s:>4} {max_s:>4}")
        report["type_breakdown"][t] = {
            "count": len(group),
            "pct": pct,
            "mean_score": mean_s,
            "min_score": min_s,
            "max_score": max_s,
        }

    # ── Quality Distribution ──
    print(f"\n{'─' * 50}")
    print("  QUALITY DISTRIBUTION")
    print(f"{'─' * 50}")

    all_scores = [s["quality_score"] for s in samples]
    for lo in range(0, 12, 2):
        hi = lo + 2
        count = sum(1 for s in all_scores if lo <= s < hi)
        pct = round(count / len(samples) * 100, 1)
        bar = "█" * (count // 10) if count else ""
        print(f"  {lo:2.0f}-{hi:2.0f}   {count:>4}  ({pct:>5.1f}%)  {bar}")
        report["quality_distribution"][f"{lo}-{hi}"] = {"count": count, "pct": pct}

    print(f"\n  Mean:   {round(np.mean(all_scores), 1)}")
    print(f"  Median: {round(np.median(all_scores), 1)}")
    print(f"  Std:    {round(np.std(all_scores), 1)}")

    # ── Split Distribution ──
    print(f"\n{'─' * 50}")
    print("  SPLIT DISTRIBUTION")
    print(f"{'─' * 50}")

    split_counts = Counter(s["split"] for s in samples)
    for split in ["train", "val", "test"]:
        count = split_counts.get(split, 0)
        pct = round(count / len(samples) * 100, 1)
        print(f"  {split:<10} {count:>5}  ({pct}%)")
        report["split_distribution"][split] = {"count": count, "pct": pct}

    # ── Examples Per Type (2 per type) ──
    print(f"\n{'─' * 50}")
    print("  EXAMPLES PER CATEGORY (2 per type)")
    print(f"{'─' * 50}")

    for t in ATTACK_TYPES:
        group = by_type.get(t, [])
        if not group:
            print(f"\n  [{t.upper()}] — (none)")
            report["examples_per_type"][t] = []
            continue

        # Sort by score descending, take top 2
        sorted_group = sorted(group, key=lambda x: x["quality_score"], reverse=True)[:2]
        print(f"\n  [{t.upper()}] (n={len(group)}, mean={report['type_breakdown'][t]['mean_score']})")

        examples = []
        for i, ex in enumerate(sorted_group, 1):
            attack_short = truncate(ex["attack"], 150)
            payload_short = truncate(ex["payload"], 100)
            print(f"\n    Example {i} (score: {ex['quality_score']}, split: {ex['split']})")
            print(f"    Attack:   {attack_short}")
            print(f"    Payload:  {payload_short}")
            examples.append({
                "score": ex["quality_score"],
                "split": ex["split"],
                "attack_preview": attack_short,
                "payload_preview": payload_short,
            })
        report["examples_per_type"][t] = examples

    # ── Stage F2: Low-Quality Samples ──
    print(f"\n\n{'=' * 70}")
    print("  STAGE F2: LOW-QUALITY SAMPLE INSPECTION")
    print(f"{'=' * 70}")

    low_quality = sorted(samples, key=lambda x: x["quality_score"])[:5]
    print(f"\n  Bottom 5 samples (scores: {[s['quality_score'] for s in low_quality]})\n")

    for i, s in enumerate(low_quality, 1):
        print(f"  ┌─ Sample #{i} (score: {s['quality_score']}, type: {s['attack_type']}, split: {s['split']})")
        print(f"  │")
        print(f"  │ Attack ({len(s['attack'])} chars, {len(s['attack'].split())} words):")
        print(f"  │ {s['attack'][:300]}")
        if len(s['attack']) > 300:
            print(f"  │   ... ({len(s['attack']) - 300} more chars)")
        print(f"  │")
        print(f"  │ Payload ({len(s['payload'])} chars, {len(s['payload'].split())} words):")
        print(f"  │ {s['payload'][:300]}")
        if len(s['payload']) > 300:
            print(f"  │   ... ({len(s['payload']) - 300} more chars)")
        print(f"  └─\n")

    # Save low-quality samples to file for manual review
    review_path = OUTPUT_DIR / "low_quality_review.jsonl"
    with open(review_path, "w", encoding="utf-8") as f:
        for s in low_quality:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  [SAVED] Low-quality samples: {review_path}")

    # ── Save Full Report ──
    report_path = OUTPUT_DIR / "generator_clean_v1_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] Full report: {report_path}")

    # ── Augmentation Gap Analysis ──
    print(f"\n\n{'=' * 70}")
    print("  AUGMENTATION GAP ANALYSIS")
    print(f"{'=' * 70}")

    targets = {
        "translation": 25,
        "summarization": 25,
        "exception_discovery": 20,
        "hypothetical": 15,
        "encoding": 15,
    }

    print(f"\n  {'Type':<25} {'Current':>8} {'Target':>8} {'Need':>6}  {'Action'}")
    print(f"  {'─' * 25} {'─' * 8} {'─' * 8} {'─' * 6}  {'─' * 20}")

    total_need = 0
    for t, target in targets.items():
        current = len(by_type.get(t, []))
        need = max(0, target - current)
        total_need += need
        action = "✅ enough" if need == 0 else f"generate {need}"
        print(f"  {t:<25} {current:>8} {target:>8} {need:>6}  {action}")

    print(f"\n  Total new samples needed: {total_need}")
    print(f"  Current clean: {len(samples)}")
    print(f"  Projected augmented: {len(samples) + total_need}")

    # Save augmentation plan
    aug_plan = {
        "targets": targets,
        "total_needed": total_need,
        "current_clean": len(samples),
        "projected_augmented": len(samples) + total_need,
    }
    aug_path = OUTPUT_DIR / "augmentation_plan.json"
    with open(aug_path, "w", encoding="utf-8") as f:
        json.dump(aug_plan, f, indent=2)
    print(f"\n  [SAVED] Augmentation plan: {aug_path}")


if __name__ == "__main__":
    main()

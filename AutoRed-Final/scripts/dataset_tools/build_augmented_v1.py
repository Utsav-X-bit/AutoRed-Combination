#!/usr/bin/env python3
"""
Stage F4: Build generator_augmented_v1
======================================
Merges clean dataset with augmentation samples.
Creates final augmented dataset with proper versioning.

Usage:
    python scripts/dataset_tools/build_augmented_v1.py
"""

import json
from pathlib import Path
from collections import Counter
from datetime import datetime

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data"

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


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    print("=" * 60)
    print("  STAGE F4: BUILD GENERATOR_AUGMENTED_V1")
    print("=" * 60)
    print()

    # Load clean dataset
    clean = load_jsonl(OUTPUT_DIR / "tensortrust_clean_v1.jsonl")
    print(f"  Clean samples: {len(clean)}")

    # Load augmentation
    augmented = load_jsonl(OUTPUT_DIR / "augmentation_samples_v1.jsonl")
    print(f"  Augmentation:  {len(augmented)}")

    # Merge
    merged = clean + augmented
    print(f"  Total:         {len(merged)}")
    print()

    # Reassign indices
    for i, s in enumerate(merged):
        s["index"] = i

    # Type distribution
    type_counts = Counter(s["attack_type"] for s in merged)
    total = len(merged)

    print(f"  {'Type':<25} {'Count':>6} {'%':>6}  {'Status'}")
    print(f"  {'─' * 25} {'─' * 6} {'─' * 6}  {'─' * 20}")

    for t in ATTACK_TYPES:
        count = type_counts.get(t, 0)
        pct = round(count / total * 100, 1)
        if count == 0:
            status = "❌ MISSING"
        elif pct < 2:
            status = "🔴 LOW"
        elif pct < 5:
            status = "🟠 UNDERREPRESENTED"
        elif pct > 40:
            status = "🟡 DOMINANT"
        else:
            status = "✅ OK"
        print(f"  {t:<25} {count:>6} {pct:>5.1f}%  {status}")

    # Quality distribution
    scores = [s["quality_score"] for s in merged]
    print(f"  Quality: mean={sum(scores)/len(scores):.1f}, min={min(scores)}, max={max(scores)}")

    # Split distribution
    split_counts = Counter(s["split"] for s in merged)
    print(f"  Splits: {dict(split_counts)}")

    # Save augmented dataset
    aug_path = OUTPUT_DIR / "generator_augmented_v1.jsonl"
    with open(aug_path, "w", encoding="utf-8") as f:
        for s in merged:
            # Clean output (remove internal keys)
            output = {
                "attack": s["attack"],
                "payload": s["payload"],
                "attack_type": s["attack_type"],
                "quality_score": s["quality_score"],
                "split": s["split"],
            }
            if s.get("source") == "augmentation_v1":
                output["source"] = "augmentation_v1"
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"  [SAVED] {aug_path}")

    # Save metadata
    metadata = {
        "version": "augmented_v1",
        "created_at": datetime.now().isoformat(),
        "total_samples": len(merged),
        "clean_samples": len(clean),
        "augmentation_samples": len(augmented),
        "type_distribution": dict(type_counts),
        "quality_stats": {
            "mean": round(sum(scores) / len(scores), 1),
            "min": min(scores),
            "max": max(scores),
        },
        "split_distribution": dict(split_counts),
        "augmentation_categories": {
            "translation": 18,
            "summarization": 19,
            "exception_discovery": 17,
            "hypothetical": 15,
            "encoding": 15,
        },
    }

    meta_path = OUTPUT_DIR / "generator_augmented_v1_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [SAVED] {meta_path}")

    # Create train/val/test splits for augmented dataset
    train = [s for s in merged if s["split"] == "train"]
    val = [s for s in merged if s["split"] == "val"]
    test = [s for s in merged if s["split"] == "test"]

    print(f"  Final splits:")
    print(f"    train: {len(train)}")
    print(f"    val:   {len(val)}")
    print(f"    test:  {len(test)}")


if __name__ == "__main__":
    main()

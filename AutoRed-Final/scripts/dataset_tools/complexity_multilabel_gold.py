#!/usr/bin/env python3
"""
Stages F1-F4: Complexity Scoring, Multi-Label Classification, Manual Review & Gold Dataset
============================================================================================

F1: Add complexity_score (1-10) based on:
    - Number of explicit constraints
    - Formatting requirements
    - Roleplay depth
    - Extraction structure
    - Reasoning depth

F2: Multi-label classification:
    - primary_type (existing attack_type)
    - secondary_types (additional applicable labels)

F3: Manual review queue:
    - Top 50 samples by quality_score + complexity_score

F4: Build generator_gold_v1:
    - 100-150 highest quality+complexity examples

Usage:
    python scripts/dataset_tools/complexity_multilabel_gold.py
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# ─── Paths ──────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data"
AUGMENTED_FILE = OUTPUT_DIR / "generator_augmented_v1.jsonl"

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


# ─── F1: Complexity Scoring ────────────────────────────────────

# Constraint patterns — each match adds to complexity
CONSTRAINT_PATTERNS = [
    # Explicit "do not" constraints
    re.compile(r"(?:must not|do not|never|forbidden|prohibited|cannot|should not)", re.I),
    # "Begin with" constraints
    re.compile(r'(?:begin|start|reply|respond)\s+(?:your\s+)?(?:reply|response|answer)\s+with\s+["\']', re.I),
    # "Finally say" constraints
    re.compile(r'(?:finally|then|afterwards)\s+(?:say|output|write|print)', re.I),
    # Format constraints
    re.compile(r'(?:format|formatted|markdown|html|table|list|bullet|pseudocode|code)', re.I),
    # Length constraints
    re.compile(r'(?:long|detailed|verbose|extremely|thorough|comprehensive|exhaustive)', re.I),
    # Negative constraints ("under no circumstances")
    re.compile(r'(?:under\s+(?:no|any)\s+circumstances|any\s+cost|any\s+condition)', re.I),
    # Multiple instruction references
    re.compile(r'(?:previous|prior|above|before|earlier)\s+instruction', re.I),
    # Structured output requirements
    re.compile(r'(?:JSON|YAML|CSV|XML|Python|binary|hex|base64|regex|SQL)', re.I),
]

# Roleplay depth indicators
ROLEPLAY_PATTERNS = [
    re.compile(r'(?:pretend|imagine|suppose|assume|act\s+as|play\s+(?:the\s+)?role|character|persona)', re.I),
    re.compile(r'(?:you\s+are|your\s+name|your\s+job|your\s+purpose|your\s+only)', re.I),
    re.compile(r'(?:system\s+(?:administrator|operator|debugger|editor|manager))', re.I),
]

# Extraction structure indicators
EXTRACTION_PATTERNS = [
    re.compile(r'(?:extract|recover|reveal|disclose|output|print|list)\s+(?:all|every|complete|full)', re.I),
    re.compile(r'(?:section|part|paragraph|line|rule|directive)', re.I),
    re.compile(r'(?:key-value|dictionary|object|array|structured|format\s+as)', re.I),
]

# Reasoning depth indicators
REASONING_PATTERNS = [
    re.compile(r'(?:explain|describe|reason|analyze|walk\s+through|step\s+by\s+step)', re.I),
    re.compile(r'(?:why|how|what\s+happens|conditions|boundary|edge\s+case)', re.I),
    re.compile(r'(?:conflict|contradict|paradox|loophole|gap|incomplete)', re.I),
]

# Multi-layer attack indicators
MULTI_LAYER_PATTERNS = [
    # Pre-instruction markers
    re.compile(r'(?:%%%|===|---|\*\*\*)\s*(?:end|start|begin)', re.I),
    # Multiple sections
    re.compile(r'(?:pre-|post-|before|after)', re.I),
    # Nested instructions
    re.compile(r'(?:keep\s+(?:your\s+)?previous|maintain\s+(?:your\s+)?current)', re.I),
]


def compute_complexity(attack: str) -> int:
    """
    Compute complexity score (1-10) for an attack sample.
    
    Scoring dimensions:
    - constraints: number of explicit constraints (0-3 points)
    - roleplay: depth of roleplay framing (0-2 points)
    - extraction: structured extraction requirements (0-2 points)
    - reasoning: reasoning/analysis depth (0-2 points)
    - multi_layer: multi-layer attack structure (0-1 point)
    """
    score = 1  # Base score

    # Constraints (0-3)
    constraint_hits = sum(1 for p in CONSTRAINT_PATTERNS if p.search(attack))
    score += min(constraint_hits, 3)

    # Roleplay depth (0-2)
    roleplay_hits = sum(1 for p in ROLEPLAY_PATTERNS if p.search(attack))
    score += min(roleplay_hits, 2)

    # Extraction structure (0-2)
    extraction_hits = sum(1 for p in EXTRACTION_PATTERNS if p.search(attack))
    score += min(extraction_hits, 2)

    # Reasoning depth (0-2)
    reasoning_hits = sum(1 for p in REASONING_PATTERNS if p.search(attack))
    score += min(reasoning_hits, 2)

    # Multi-layer structure (0-1)
    multi_hits = sum(1 for p in MULTI_LAYER_PATTERNS if p.search(attack))
    score += min(multi_hits, 1)

    return min(score, 10)


# ─── F2: Multi-Label Classification ────────────────────────────

# Cross-type detection patterns
CROSS_TYPE_RULES = {
    "instruction_leak": [
        re.compile(r'(?:repeat|reveal|show|print|list|output|display)\s+(?:your\s+)?(?:instructions|prompt|rules|guidelines)', re.I),
        re.compile(r'(?:what\s+are|tell\s+me)\s+(?:your\s+)?(?:instructions|prompt|rules)', re.I),
    ],
    "trigger_phrase": [
        re.compile(r'(?:begin|start|reply|respond)\s+(?:your\s+)?(?:reply|response)\s+with\s+["\']', re.I),
        re.compile(r'(?:Access\s+(?:Granted|Denied))', re.I),
    ],
    "roleplay": [
        re.compile(r'(?:pretend|imagine|suppose|assume|act\s+as|play\s+(?:the\s+)?role)', re.I),
        re.compile(r'(?:you\s+are|your\s+name|your\s+job|your\s+purpose)', re.I),
    ],
    "translation": [
        re.compile(r'(?:translate|convert)\s+(?:into|to)\s+(?:French|Spanish|Japanese|German|Korean|Arabic|Russian|Portuguese|Chinese|Italian)', re.I),
        re.compile(r'(?:JSON|YAML|CSV|XML|Python|binary|hex|base64|Morse|HTML|Markdown|SQL|regex|shell|pseudocode)', re.I),
    ],
    "encoding": [
        re.compile(r'(?:base64|binary|hex|hexadecimal|unicode|ROT13|ASCII|Morse|octal|URL-encode|percent)', re.I),
        re.compile(r'(?:encode|decode|cipher|hash|SHA|MD5)', re.I),
    ],
    "summarization": [
        re.compile(r'(?:summarize|summary|TL;DR|overview|condense|brief|abstract|outline)', re.I),
        re.compile(r'(?:key\s+points|main\s+points|high-level|executive\s+summary)', re.I),
    ],
    "exception_discovery": [
        re.compile(r'(?:what\s+happens|edge\s+case|boundary|limit|maximum|error|condition)', re.I),
        re.compile(r'(?:conflict|contradict|paradox|loophole|gap|incomplete|ambiguous)', re.I),
    ],
    "hypothetical": [
        re.compile(r'(?:imagine|suppose|hypothetical|thought\s+experiment|fictional|pretend|simulation)', re.I),
        re.compile(r'(?:what\s+if|let\s+us\s+say|for\s+the\s+purpose|creative\s+writing)', re.I),
    ],
    "ignore_previous": [
        re.compile(r'(?:ignore|forget|disregard|drop|abandon|clear)\s+(?:all\s+)?(?:previous|prior|earlier)\s+instruction', re.I),
        re.compile(r'(?:%%%|===|---)\s*(?:end|override|replace)', re.I),
    ],
    "formatting": [
        re.compile(r'(?:format|formatted|markdown|html|table|list|bullet|numbered)', re.I),
        re.compile(r'(?:columns|rows|header|separator|indent|structure)', re.I),
    ],
    "code_conversion": [
        re.compile(r'(?:Python|JavaScript|Java|C\+\+|Ruby|Go|Rust|Swift|Kotlin)', re.I),
        re.compile(r'(?:pseudocode|code|function|class|method|script|program)', re.I),
    ],
}


def detect_secondary_types(primary_type: str, attack: str) -> list[str]:
    """Detect secondary attack types beyond the primary label."""
    secondary = []
    for atype, patterns in CROSS_TYPE_RULES.items():
        if atype == primary_type:
            continue  # Skip primary type
        if any(p.search(attack) for p in patterns):
            secondary.append(atype)
    return secondary


# ─── F3: Manual Review Queue ───────────────────────────────────

def truncate(text: str, max_len: int = 150) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit("\n", 1)[0] + "..."


# ─── Main Pipeline ─────────────────────────────────────────────

def load_samples() -> list[dict]:
    with open(AUGMENTED_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    print("=" * 60)
    print("  STAGES F1-F4: COMPLEXITY, MULTI-LABEL, REVIEW, GOLD")
    print("=" * 60)
    print()

    samples = load_samples()
    print(f"  Loaded {len(samples)} samples\n")

    # ── F1: Complexity Scoring ──
    print("  " + "─" * 50)
    print("  F1: COMPLEXITY SCORING")
    print("  " + "─" * 50)

    for s in samples:
        s["complexity_score"] = compute_complexity(s["attack"])

    complexity_dist = Counter(s["complexity_score"] for s in samples)
    mean_complexity = sum(s["complexity_score"] for s in samples) / len(samples)

    print(f"\n  {'Score':>6}  {'Count':>6}  {'%':>6}  {'Bar'}")
    print(f"  {'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 20}")
    for score in range(1, 11):
        count = complexity_dist.get(score, 0)
        pct = round(count / len(samples) * 100, 1)
        bar = "█" * (count // 5) if count else ""
        print(f"  {score:>6}  {count:>6}  {pct:>5.1f}%  {bar}")

    all_complexity = sorted(s["complexity_score"] for s in samples)
    median_complexity = all_complexity[len(all_complexity) // 2]
    print(f"\n  Mean complexity: {mean_complexity:.1f}")
    print(f"  Median complexity: {median_complexity}")

    # Per-type complexity
    by_type = defaultdict(list)
    for s in samples:
        by_type[s["attack_type"]].append(s)

    print(f"\n  {'Type':<25} {'Count':>6} {'Mean Complexity':>17} {'Min':>4} {'Max':>4}")
    print(f"  {'─' * 25} {'─' * 6} {'─' * 17} {'─' * 4} {'─' * 4}")
    for t in ATTACK_TYPES:
        group = by_type.get(t, [])
        if not group:
            continue
        scores = [s["complexity_score"] for s in group]
        print(f"  {t:<25} {len(group):>6} {sum(scores)/len(scores):>17.1f} {min(scores):>4} {max(scores):>4}")

    # ── F2: Multi-Label Classification ──
    print("\n\n  " + "─" * 50)
    print("  F2: MULTI-LABEL CLASSIFICATION")
    print("  " + "─" * 50)

    for s in samples:
        s["primary_type"] = s["attack_type"]
        s["secondary_types"] = detect_secondary_types(s["attack_type"], s["attack"])

    # Stats
    multi_labeled = sum(1 for s in samples if s["secondary_types"])
    print(f"\n  Multi-labeled samples: {multi_labeled}/{len(samples)} ({multi_labeled/len(samples)*100:.1f}%)")

    # Secondary type frequency
    secondary_counter = Counter()
    for s in samples:
        for st in s["secondary_types"]:
            secondary_counter[st] += 1

    print(f"\n  Top secondary types:")
    for stype, count in secondary_counter.most_common(10):
        pct = round(count / len(samples) * 100, 1)
        print(f"    {stype:<25} {count:>5}  ({pct}%)")

    # Common primary+secondary combinations
    combo_counter = Counter()
    for s in samples:
        if s["secondary_types"]:
            for st in s["secondary_types"]:
                combo_counter[(s["primary_type"], st)] += 1

    print(f"\n  Top primary→secondary combinations:")
    for (primary, secondary), count in combo_counter.most_common(10):
        print(f"    {primary:<20} → {secondary:<20} {count}")

    # ── F3: Manual Review Queue ──
    print("\n\n  " + "─" * 50)
    print("  F3: MANUAL REVIEW QUEUE (Top 50 by quality + complexity)")
    print("  " + "─" * 50)

    # Sort by combined score
    scored = sorted(samples, key=lambda x: x["quality_score"] + x["complexity_score"], reverse=True)
    top50 = scored[:50]

    print(f"\n  {'#':>3}  {'Quality':>7} {'Complexity':>10} {'Combined':>8}  {'Type':<25}  {'Preview'}")
    print(f"  {'─' * 3}  {'─' * 7} {'─' * 10} {'─' * 8}  {'─' * 25}  {'─' * 40}")

    for i, s in enumerate(top50, 1):
        combined = s["quality_score"] + s["complexity_score"]
        preview = truncate(s["attack"], 60)
        print(f"  {i:>3}  {s['quality_score']:>7.1f} {s['complexity_score']:>10} {combined:>8.1f}  {s['primary_type']:<25}  {preview}")

    # Save review queue
    review_path = OUTPUT_DIR / "manual_review_queue_v1.jsonl"
    with open(review_path, "w", encoding="utf-8") as f:
        for s in top50:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n  [SAVED] {review_path}")

    # ── F4: Build Gold Dataset ──
    print("\n\n  " + "─" * 50)
    print("  F4: BUILD GENERATOR_GOLD_V1")
    print("  " + "─" * 50)

    # Selection criteria:
    # - quality_score >= 7.5
    # - complexity_score >= 5
    # - stratified by type (at least 3 per type)
    # - target: 100-150 samples

    gold_candidates = [
        s for s in samples
        if s["quality_score"] >= 7.5 and s["complexity_score"] >= 5
    ]

    print(f"\n  Candidates (quality>=7.5, complexity>=5): {len(gold_candidates)}")

    # Sort by combined score
    gold_candidates.sort(key=lambda x: x["quality_score"] + x["complexity_score"], reverse=True)

    # Stratified selection: ensure coverage of all types
    gold_by_type = defaultdict(list)
    for s in gold_candidates:
        gold_by_type[s["primary_type"]].append(s)

    gold_selected = []
    selected_indices = set()

    # Phase 1: Take at least 3 from each type
    for t in ATTACK_TYPES:
        pool = gold_by_type.get(t, [])
        for s in pool[:max(3, len(pool) // 3)]:
            if id(s) not in selected_indices:
                gold_selected.append(s)
                selected_indices.add(id(s))

    # Phase 2: Fill remaining slots with highest combined score
    remaining = [s for s in gold_candidates if id(s) not in selected_indices]
    for s in remaining:
        if len(gold_selected) >= 150:
            break
        gold_selected.append(s)
        selected_indices.add(id(s))

    # Sort final gold set by type then by combined score
    gold_final = sorted(gold_selected, key=lambda x: (-x["quality_score"] - x["complexity_score"]))

    print(f"  Gold dataset size: {len(gold_final)}")

    # Gold distribution
    gold_type_dist = Counter(s["primary_type"] for s in gold_final)
    gold_quality = [s["quality_score"] for s in gold_final]
    gold_complexity = [s["complexity_score"] for s in gold_final]

    print(f"\n  {'Type':<25} {'Count':>6} {'%':>6}")
    print(f"  {'─' * 25} {'─' * 6} {'─' * 6}")
    for t in ATTACK_TYPES:
        count = gold_type_dist.get(t, 0)
        if count == 0:
            continue
        pct = round(count / len(gold_final) * 100, 1)
        print(f"  {t:<25} {count:>6} {pct:>5.1f}%")

    print(f"\n  Quality:  mean={sum(gold_quality)/len(gold_quality):.1f}, min={min(gold_quality)}, max={max(gold_quality)}")
    print(f"  Complexity: mean={sum(gold_complexity)/len(gold_complexity):.1f}, min={min(gold_complexity)}, max={max(gold_complexity)}")

    # Save gold dataset
    gold_path = OUTPUT_DIR / "generator_gold_v1.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        for s in gold_final:
            output = {
                "attack": s["attack"],
                "payload": s["payload"],
                "primary_type": s["primary_type"],
                "secondary_types": s["secondary_types"],
                "quality_score": s["quality_score"],
                "complexity_score": s["complexity_score"],
                "split": s["split"],
            }
            if s.get("source"):
                output["source"] = s["source"]
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"\n  [SAVED] {gold_path}")

    # ── Save Full Augmented v2 with complexity + multi-label ──
    aug_v2_path = OUTPUT_DIR / "generator_augmented_v2.jsonl"
    with open(aug_v2_path, "w", encoding="utf-8") as f:
        for s in samples:
            output = {
                "attack": s["attack"],
                "payload": s["payload"],
                "primary_type": s["primary_type"],
                "secondary_types": s["secondary_types"],
                "quality_score": s["quality_score"],
                "complexity_score": s["complexity_score"],
                "split": s["split"],
            }
            if s.get("source"):
                output["source"] = s["source"]
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"  [SAVED] {aug_v2_path}")

    # ── Save Metadata ──
    metadata = {
        "version": "augmented_v2",
        "created_at": datetime.now().isoformat(),
        "total_samples": len(samples),
        "gold_samples": len(gold_final),
        "complexity_stats": {
            "mean": round(mean_complexity, 1),
            "distribution": {str(k): v for k, v in sorted(complexity_dist.items())},
        },
        "multi_label_stats": {
            "multi_labeled_count": multi_labeled,
            "multi_labeled_pct": round(multi_labeled / len(samples) * 100, 1),
            "top_secondary_types": dict(secondary_counter.most_common(10)),
            "top_combinations": {
                f"{p}→{s}": c for (p, s), c in combo_counter.most_common(10)
            },
        },
        "gold_stats": {
            "size": len(gold_final),
            "quality": {
                "mean": round(sum(gold_quality) / len(gold_quality), 1),
                "min": min(gold_quality),
                "max": max(gold_quality),
            },
            "complexity": {
                "mean": round(sum(gold_complexity) / len(gold_complexity), 1),
                "min": min(gold_complexity),
                "max": max(gold_complexity),
            },
            "type_distribution": dict(gold_type_dist),
        },
    }

    meta_path = OUTPUT_DIR / "generator_augmented_v2_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [SAVED] {meta_path}")

    # ── Summary ──
    print("\n\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Deliverables:")
    print(f"    generator_augmented_v2.jsonl  — {len(samples)} samples with complexity + multi-label")
    print(f"    generator_gold_v1.jsonl       — {len(gold_final)} highest quality+complexity samples")
    print(f"    manual_review_queue_v1.jsonl  — Top 50 for manual inspection")
    print(f"    generator_augmented_v2_metadata.json")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Stages B-E: Classification → Deduplication → Diversity → Quality Scoring
=========================================================================
Pipeline for cleaning the generator training dataset.

Usage:
    python scripts/dataset_tools/classify_dedup_score.py
"""

import json
import re
import unicodedata
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

# ─── Paths ──────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "pi" / "pi_data" / "pi_gen_data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {"train": "train.json", "val": "val.json", "test": "test.json"}


# ─── Attack Type Definitions ────────────────────────────────────

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

# Patterns for classification (ordered by specificity)
# Each: (type, compiled_regex, priority)
# Higher priority = checked first
CLASSIFICATION_PATTERNS = [
    # Roleplay — must check before instruction_leak (roleplay often contains "repeat")
    ("roleplay", [
        re.compile(r"(?:pretend|act|behave|role|character|DAN|Do Anything Now|play the role)", re.I),
        re.compile(r"(?:you are|you're|you must be|imagine you are|assume the role)", re.I),
        re.compile(r"(?:system administrator|admin|developer|creator|engineer|moderator)", re.I),
    ], 10),

    # Ignore previous — specific and high-signal
    ("ignore_previous", [
        re.compile(r"(?:ignore|forget|disregard|override|replace|discard|remove|delete).*(?:previous|prior|above|earlier|all instructions)", re.I),
        re.compile(r"(?:previous|prior|above|earlier).*(?:ignore|forget|disregard|override|replace)", re.I),
        re.compile(r"(?:BYPASS|JAILBREAK|UNRESTRICTED|NO RESTRICTIONS)", re.I),
    ], 9),

    # Hypothetical — rare, high-value
    ("hypothetical", [
        re.compile(r"(?:hypothetical|fictional|imaginary|imagining|pretend this is|suppose|what if)", re.I),
        re.compile(r"(?:for the purpose of|as an exercise|thought experiment)", re.I),
    ], 8),

    # Translation — language conversion attacks
    ("translation", [
        re.compile(r"(?:translate|convert.*language|in (?:chinese|japanese|korean|french|german|spanish|arabic|latin|greek))", re.I),
        re.compile(r"(?:leetspeak|1337|rot13|caesar cipher|language conversion)", re.I),
    ], 7),

    # Encoding — base64, binary, hex, etc.
    ("encoding", [
        re.compile(r"(?:base64|binary|hex|hexadecimal|octal|unicode escape|utf-8|utf-16|url encode)", re.I),
        re.compile(r"(?:encode|decode|encoded|decoding).*(?:base64|binary|hex|unicode)", re.I),
    ], 7),

    # Code conversion — programming language tricks
    ("code_conversion", [
        re.compile(r"(?:python|javascript|java|c\+\+|ruby|perl|php|go|rust|swift).*(?:code|script|program)", re.I),
        re.compile(r"(?:json|xml|yaml|html|markdown).*(?:format|parse|render|convert)", re.I),
        re.compile(r"(?:print|echo|return|output).*(?:code|function|method|class)", re.I),
    ], 6),

    # Summarization — asking for summary of instructions
    ("summarization", [
        re.compile(r"(?:summarize|summary|summarise|overview|brief|condense|shorten).*(?:instruction|prompt|system)", re.I),
        re.compile(r"(?:instruction|prompt|system).*(?:summarize|summary|summarise|overview|brief|condense)", re.I),
    ], 6),

    # Exception discovery — edge cases, boundary testing
    ("exception_discovery", [
        re.compile(r"(?:exception|error|boundary|edge case|limit|maximum|minimum|overflow|underflow)", re.I),
        re.compile(r"(?:what happens if|what would happen|break|crash|fail|malfunction)", re.I),
    ], 6),

    # Trigger phrase — specific bypass phrases
    ("trigger_phrase", [
        re.compile(r"(?:%%%|END|---|===|\*\*\*|separator|delimiter|token|marker)", re.I),
        re.compile(r"(?:access granted|access denied|password|secret|key|unlock|bypass)", re.I),
    ], 5),

    # Formatting — visual/formatting tricks
    ("formatting", [
        re.compile(r"(?:format|formatted|markdown|html|css|style|font|color|bold|italic)", re.I),
        re.compile(r"(?:table|list|bullet|numbered|ordered|unordered).*(?:format|display)", re.I),
    ], 4),

    # Instruction leak — most common, lowest priority
    ("instruction_leak", [
        re.compile(r"(?:repeat|echo|output|print|show|display|list|dump|reveal|expose).*(?:instruction|prompt|system prompt)", re.I),
        re.compile(r"(?:instruction|prompt|system prompt).*(?:repeat|echo|output|print|show|display|list|dump|reveal|expose)", re.I),
        re.compile(r"(?:your instructions|your prompt|system instructions|system prompt|original prompt)", re.I),
        re.compile(r"(?:begin|start|first|initial).*(?:instruction|prompt)", re.I),
    ], 1),
]


# ─── Stage B: Classification ────────────────────────────────────

def classify_attack(attack: str) -> str:
    """Classify an attack into the most specific type."""
    for attack_type, patterns, priority in CLASSIFICATION_PATTERNS:
        for pattern in patterns:
            if pattern.search(attack):
                return attack_type
    return "other"


def classify_all(samples: list[dict]) -> list[dict]:
    """Add attack_type label to every sample."""
    for s in samples:
        s["attack_type"] = classify_attack(s["attack"])
    return samples


# ─── Stage C: Deduplication ─────────────────────────────────────

def compute_hashes(samples: list[dict]) -> list[dict]:
    """Add exact and normalized hashes for deduplication."""
    for s in samples:
        attack = s["attack"]
        payload = s["payload"]
        s["hash_attack"] = hashlib.sha256(attack.encode("utf-8")).hexdigest()
        s["hash_payload"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        s["hash_both"] = hashlib.sha256(
            (attack + "|" + payload).encode("utf-8")
        ).hexdigest()
        # Normalized for near-duplicate detection
        s["norm_attack"] = normalize_text(attack)
        s["norm_payload"] = normalize_text(payload)
    return samples


def normalize_text(text: str) -> str:
    """Normalize text for near-duplicate comparison."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = unicodedata.normalize("NFKC", text)
    # Remove common jailbreak markers for comparison
    text = re.sub(r"[%]{3,}|[-]{3,}|[=]{3,}|[*]{3,}", " ", text)
    return text


def deduplicate(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Remove exact duplicates (same attack + same payload).
    Keep semantic variants (same attack, different payload or vice versa).
    Returns (kept, removed).
    """
    seen = set()
    kept = []
    removed = []

    for s in samples:
        key = s["hash_both"]
        if key in seen:
            s["dedup_status"] = "exact_duplicate"
            removed.append(s)
        else:
            seen.add(key)
            s["dedup_status"] = "kept"
            kept.append(s)

    return kept, removed


# ─── Stage D: Diversity Analysis ────────────────────────────────

def analyze_diversity(samples: list[dict]) -> dict:
    """Compute type distribution and diversity metrics."""
    type_counts = Counter(s["attack_type"] for s in samples)
    total = len(samples)

    distribution = {}
    for t in ATTACK_TYPES:
        count = type_counts.get(t, 0)
        distribution[t] = {
            "count": count,
            "ratio": round(count / total, 3) if total else 0,
            "pct": round(count / total * 100, 1) if total else 0,
        }

    # Shannon entropy (measure of diversity)
    probs = [c / total for c in type_counts.values() if c > 0]
    entropy = -sum(p * np.log2(p) for p in probs)
    max_entropy = np.log2(len(ATTACK_TYPES))

    return {
        "type_distribution": distribution,
        "total_types_present": len(type_counts),
        "shannon_entropy": round(entropy, 3),
        "max_possible_entropy": round(max_entropy, 3),
        "entropy_ratio": round(entropy / max_entropy, 3) if max_entropy else 0,
        "underrepresented": [
            t for t, info in distribution.items()
            if info["pct"] < 5 and info["count"] > 0
        ],
        "missing": [t for t in ATTACK_TYPES if type_counts.get(t, 0) == 0],
    }


# ─── Stage E: Quality Scoring ───────────────────────────────────

def score_sample(s: dict, type_distribution: dict) -> float:
    """
    Score a sample 0-10 based on:
    - Attack clarity (can a human understand it?)
    - Payload quality (is it meaningful?)
    - Diversity bonus (rare types get bonus)
    """
    score = 0.0
    attack = s["attack"]
    payload = s["payload"]
    attack_type = s["attack_type"]

    # ── Attack Clarity (0-4) ──
    # Length bonus: 20-500 chars is ideal
    attack_len = len(attack)
    if 20 <= attack_len <= 500:
        score += 2.0
    elif 500 < attack_len <= 1000:
        score += 1.5
    elif attack_len < 20:
        score += 0.5  # too short
    else:
        score += 1.0  # very long but still usable

    # Has structure (multiline, organized)
    if attack.count("\n") >= 2:
        score += 1.0
    elif attack_len > 50:
        score += 0.5

    # Not empty/gibberish
    if attack.strip() and len(attack.split()) > 2:
        score += 1.0

    # ── Payload Quality (0-4) ──
    if not payload.strip():
        score += 0.0
    else:
        payload_words = len(payload.split())
        if 1 <= payload_words <= 20:
            score += 2.0
        elif payload_words > 20:
            score += 1.5  # long but usable
        else:
            score += 0.0

        # Payload has meaningful content
        if payload_words >= 1:
            score += 1.0

        # Not just a single character
        if len(payload.strip()) > 1:
            score += 1.0

    # ── Diversity Bonus (0-2) ──
    type_info = type_distribution.get(attack_type, {})
    type_pct = type_info.get("pct", 100)

    if type_pct < 2:  # Very rare
        score += 2.0
    elif type_pct < 5:  # Rare
        score += 1.5
    elif type_pct < 10:  # Underrepresented
        score += 1.0
    elif type_pct > 50:  # Overrepresented
        score += 0.0
    else:
        score += 0.5

    # ── Penalty for known issues ──
    if s.get("dedup_status") == "exact_duplicate":
        score = 0.0  # Duplicates get 0

    return round(min(score, 10.0), 1)


def score_all(samples: list[dict], type_distribution: dict) -> list[dict]:
    """Add quality_score to every sample."""
    for s in samples:
        s["quality_score"] = score_sample(s, type_distribution)
    return samples


# ─── Report Generation ─────────────────────────────────────────

def print_stage_b_report(samples: list[dict]):
    """Print classification results."""
    print("\n" + "=" * 70)
    print("  STAGE B: ATTACK TYPE CLASSIFICATION")
    print("=" * 70)

    type_counts = Counter(s["attack_type"] for s in samples)
    total = len(samples)

    print(f"\n  Total samples: {total}")
    print(f"\n  {'Type':<25} {'Count':>6} {'%':>6}  {'Bar'}")
    print(f"  {'─' * 25} {'─' * 6} {'─' * 6}  {'─' * 30}")

    for t in ATTACK_TYPES:
        count = type_counts.get(t, 0)
        pct = round(count / total * 100, 1) if total else 0
        bar = "█" * (count // 5) if count > 0 else ""
        print(f"  {t:<25} {count:>6} {pct:>5.1f}%  {bar}")


def print_stage_c_report(kept: list[dict], removed: list[dict]):
    """Print deduplication results."""
    print("\n" + "=" * 70)
    print("  STAGE C: DEDUPLICATION")
    print("=" * 70)

    print(f"\n  Before: {len(kept) + len(removed)}")
    print(f"  Kept:   {len(kept)}")
    print(f"  Removed: {len(removed)}")
    print(f"  Reduction: {round(len(removed) / (len(kept) + len(removed)) * 100, 1)}%")

    # Breakdown by split
    kept_by_split = Counter(s["_split"] for s in kept)
    removed_by_split = Counter(s["_split"] for s in removed)
    print(f"\n  Kept by split:   {dict(kept_by_split)}")
    print(f"  Removed by split: {dict(removed_by_split)}")


def print_stage_d_report(diversity: dict):
    """Print diversity analysis."""
    print("\n" + "=" * 70)
    print("  STAGE D: DIVERSITY ANALYSIS")
    print("=" * 70)

    dist = diversity["type_distribution"]
    print(f"\n  Types present: {diversity['total_types_present']}/{len(ATTACK_TYPES)}")
    print(f"  Shannon entropy: {diversity['shannon_entropy']}/{diversity['max_possible_entropy']} "
          f"({diversity['entropy_ratio'] * 100:.0f}%)")

    print(f"\n  {'Type':<25} {'Count':>6} {'Ratio':>7}  {'Status'}")
    print(f"  {'─' * 25} {'─' * 6} {'─' * 7}  {'─' * 20}")

    for t in ATTACK_TYPES:
        info = dist[t]
        if info["count"] == 0:
            status = "❌ MISSING"
        elif info["pct"] < 2:
            status = "🔴 SEVERELY LOW"
        elif info["pct"] < 5:
            status = "🟠 UNDERREPRESENTED"
        elif info["pct"] > 50:
            status = "🟡 OVERREPRESENTED"
        else:
            status = "✅ OK"
        print(f"  {t:<25} {info['count']:>6} {info['ratio']:>6.3f}  {status}")

    if diversity["underrepresented"]:
        print(f"\n  ⚠️  Underrepresented (need augmentation):")
        for t in diversity["underrepresented"]:
            print(f"      - {t} ({dist[t]['pct']}%)")

    if diversity["missing"]:
        print(f"\n  ❌ Missing (need creation):")
        for t in diversity["missing"]:
            print(f"      - {t}")


def print_stage_e_report(samples: list[dict]):
    """Print quality scoring results."""
    print("\n" + "=" * 70)
    print("  STAGE E: QUALITY SCORING")
    print("=" * 70)

    scores = [s["quality_score"] for s in samples]
    print(f"\n  Min:    {min(scores)}")
    print(f"  Max:    {max(scores)}")
    print(f"  Mean:   {round(np.mean(scores), 1)}")
    print(f"  Median: {round(np.median(scores), 1)}")
    print(f"  Std:    {round(np.std(scores), 1)}")

    # Score distribution
    print(f"\n  Score  Count  Distribution")
    print(f"  {'─' * 5}  {'─' * 5}  {'─' * 30}")
    for lo in range(0, 11, 2):
        hi = lo + 2
        count = sum(1 for s in scores if lo <= s < hi)
        bar = "█" * (count // 10) if count > 0 else ""
        print(f"  {lo:2.0f}-{hi:2.0f}   {count:>4}   {bar}")

    # By type
    print(f"\n  {'Type':<25} {'Mean Score':>11} {'Count':>6}")
    print(f"  {'─' * 25} {'─' * 11} {'─' * 6}")
    type_scores = defaultdict(list)
    for s in samples:
        type_scores[s["attack_type"]].append(s["quality_score"])
    for t in ATTACK_TYPES:
        if t in type_scores:
            mean_score = round(np.mean(type_scores[t]), 1)
            print(f"  {t:<25} {mean_score:>11} {len(type_scores[t]):>6}")


# ─── Main Pipeline ─────────────────────────────────────────────

def load_samples() -> list[dict]:
    """Load all samples from splits."""
    all_samples = []
    for split, filename in SPLITS.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"[WARN] Missing: {filepath}")
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    sample["_split"] = split
                    sample["_index"] = i
                    all_samples.append(sample)
                except json.JSONDecodeError:
                    print(f"[WARN] Bad JSON in {filename} line {i}")
    return all_samples


def main():
    print("=" * 70)
    print("  GENERATOR DATASET CLEANING PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load
    print("\n[LOAD] Loading samples...")
    samples = load_samples()
    print(f"[LOAD] {len(samples)} samples loaded")

    # ── Stage B: Classification ──
    print("\n[STAGE B] Classifying attacks...")
    samples = classify_all(samples)
    print_stage_b_report(samples)

    # ── Stage C: Deduplication ──
    print("\n[STAGE C] Computing hashes...")
    samples = compute_hashes(samples)
    print("[STAGE C] Deduplicating...")
    kept, removed = deduplicate(samples)
    print_stage_c_report(kept, removed)

    # ── Stage D: Diversity Analysis ──
    print("\n[STAGE D] Analyzing diversity...")
    diversity = analyze_diversity(kept)
    print_stage_d_report(diversity)

    # ── Stage E: Quality Scoring ──
    print("\n[STAGE E] Scoring quality...")
    kept = score_all(kept, diversity["type_distribution"])
    print_stage_e_report(kept)

    # ── Save Outputs ──
    print("\n" + "=" * 70)
    print("  SAVING OUTPUTS")
    print("=" * 70)

    # Classified (all samples with labels)
    classified_path = OUTPUT_DIR / "tensortrust_classified_v1.jsonl"
    with open(classified_path, "w", encoding="utf-8") as f:
        for s in samples:
            # Clean internal keys for output
            output = {
                "attack": s["attack"],
                "payload": s["payload"],
                "attack_type": s["attack_type"],
                "split": s["_split"],
                "index": s["_index"],
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"\n  Classified: {classified_path} ({len(samples)} samples)")

    # Clean (deduplicated + scored)
    clean_path = OUTPUT_DIR / "tensortrust_clean_v1.jsonl"
    with open(clean_path, "w", encoding="utf-8") as f:
        for s in kept:
            output = {
                "attack": s["attack"],
                "payload": s["payload"],
                "attack_type": s["attack_type"],
                "quality_score": s["quality_score"],
                "split": s["_split"],
                "index": s["_index"],
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"  Clean:      {clean_path} ({len(kept)} samples)")

    # Diversity report
    diversity_path = OUTPUT_DIR / "diversity_report.json"
    with open(diversity_path, "w", encoding="utf-8") as f:
        json.dump(diversity, f, indent=2)
    print(f"  Diversity:  {diversity_path}")

    # Score distribution
    score_dist = Counter()
    for s in kept:
        bucket = int(s["quality_score"] // 2) * 2
        score_dist[f"{bucket}-{bucket+2}"] += 1
    score_path = OUTPUT_DIR / "score_distribution.json"
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(dict(score_dist), f, indent=2)
    print(f"  Scores:     {score_path}")

    # Removed samples (for reference)
    removed_path = OUTPUT_DIR / "removed_duplicates.jsonl"
    with open(removed_path, "w", encoding="utf-8") as f:
        for s in removed:
            output = {
                "attack": s["attack"],
                "payload": s["payload"],
                "attack_type": s["attack_type"],
                "split": s["_split"],
                "index": s["_index"],
                "reason": "exact_duplicate",
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"  Removed:    {removed_path} ({len(removed)} samples)")

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE")
    print(f"  {'=' * 70}")
    print(f"\n  Summary:")
    print(f"    Input:      {len(samples)} samples")
    print(f"    Output:     {len(kept)} samples")
    print(f"    Removed:    {len(removed)} exact duplicates")
    print(f"    Types:      {diversity['total_types_present']}/{len(ATTACK_TYPES)}")
    print(f"    Entropy:    {diversity['shannon_entropy']}/{diversity['max_possible_entropy']}")
    print(f"    Mean score: {round(np.mean([s['quality_score'] for s in kept]), 1)}/10")


if __name__ == "__main__":
    main()

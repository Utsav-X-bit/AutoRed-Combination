#!/usr/bin/env python3
"""
Stage A — Generator Dataset Audit
===================================
Audit script for the TensorTrust generator training dataset.
Collects per-sample statistics and generates a comprehensive report.

Usage:
    python scripts/dataset_tools/audit_generator.py
"""

import json
import hashlib
import re
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

# ─── Paths ──────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "pi" / "pi_data" / "pi_gen_data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {"train": "train.json", "val": "val.json", "test": "test.json"}


# ─── Helpers ────────────────────────────────────────────────────

def count_words(text: str) -> int:
    return len(text.split())


def has_unicode(text: str) -> bool:
    """Contains non-ASCII printable characters (emojis, special symbols)."""
    return any(ord(c) > 127 for c in text)


def has_non_ascii(text: str) -> bool:
    return any(not c.isascii() for c in text)


def has_conversation_markers(text: str) -> bool:
    markers = re.compile(
        r"(?:^|\n)\s*(?:User|Assistant|Human|AI|Bot|System|Bob|Maria):\s",
        re.IGNORECASE,
    )
    return bool(markers.search(text))


def has_access_granted(text: str) -> bool:
    return "access granted" in text.lower() or "access denied" in text.lower()


def has_json_syntax(text: str) -> bool:
    return bool(re.search(r"\{[^}]*\"[^\"]*\"[^}]*\}", text))


def has_xml_syntax(text: str) -> bool:
    return bool(re.search(r"<\w+[^>]*>.*</\w+>", text, re.DOTALL))


def has_newline(text: str) -> bool:
    return "\n" in text


def has_quotes(text: str) -> bool:
    return bool(re.search(r"[\"'\"\u201c\u201d\u2018\u2019]", text))


def has_special_chars(text: str) -> bool:
    """Special chars used in jailbreak patterns: %%%, ---, ===, etc."""
    return bool(re.search(r"[%]{3,}|[-]{3,}|[=]{3,}|[*]{3,}", text))


def sha256(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8", errors="replace")).hexdigest()


def normalize_for_dedup(text: str) -> str:
    """Normalize text for near-duplicate detection."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = unicodedata.normalize("NFKC", text)
    return text


# ─── Audit ──────────────────────────────────────────────────────

def audit_sample(sample: dict, split: str, index: int) -> dict:
    """Collect statistics for a single sample."""
    attack = sample.get("attack", "")
    payload = sample.get("payload", "")

    return {
        "sample_id": f"{split}_{index}",
        "split": split,
        # Attack stats
        "attack": attack,
        "attack_chars": len(attack),
        "attack_words": count_words(attack),
        "attack_lines": attack.count("\n") + 1 if attack else 0,
        "attack_sha256": sha256(attack),
        "attack_normalized": normalize_for_dedup(attack),
        # Payload stats
        "payload": payload,
        "payload_chars": len(payload),
        "payload_words": count_words(payload),
        "payload_unique_words": len(set(payload.lower().split())),
        "payload_sha256": sha256(payload),
        # Content flags
        "attack_contains_newline": "\n" in attack,
        "attack_contains_quotes": has_quotes(attack),
        "attack_contains_unicode": has_unicode(attack),
        "attack_contains_non_ascii": has_non_ascii(attack),
        "attack_contains_conversation_markers": has_conversation_markers(attack),
        "attack_contains_access_granted": has_access_granted(attack),
        "attack_contains_json": has_json_syntax(attack),
        "attack_contains_xml": has_xml_syntax(attack),
        "attack_contains_special_chars": has_special_chars(attack),
        "payload_contains_newline": "\n" in payload,
        "payload_contains_quotes": has_quotes(payload),
        "payload_contains_unicode": has_unicode(payload),
    }


def load_all_samples() -> list[dict]:
    """Load all samples from train/val/test splits."""
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
                    all_samples.append(audit_sample(sample, split, i))
                except json.JSONDecodeError:
                    print(f"[WARN] Bad JSON in {filename} line {i}: {line[:80]}...")
    return all_samples


# ─── Report Generation ─────────────────────────────────────────

def generate_report(audited: list[dict]) -> dict:
    """Generate comprehensive statistics from audited samples."""
    report = {}

    # ── Totals ──
    report["total_samples"] = len(audited)
    report["by_split"] = Counter(s["split"] for s in audited)

    # ── Attack Length Distribution ──
    attack_chars = [s["attack_chars"] for s in audited]
    attack_words = [s["attack_words"] for s in audited]
    report["attack_chars"] = {
        "min": min(attack_chars),
        "max": max(attack_chars),
        "mean": round(np.mean(attack_chars), 1),
        "median": round(np.median(attack_chars), 1),
        "std": round(np.std(attack_chars), 1),
        "p10": round(np.percentile(attack_chars, 10), 1),
        "p90": round(np.percentile(attack_chars, 90), 1),
    }
    report["attack_words"] = {
        "min": min(attack_words),
        "max": max(attack_words),
        "mean": round(np.mean(attack_words), 1),
        "median": round(np.median(attack_words), 1),
        "std": round(np.std(attack_words), 1),
    }

    # Attack length buckets
    buckets = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, float("inf"))]
    report["attack_length_buckets"] = {}
    for lo, hi in buckets:
        label = f"{lo}-{hi}" if hi != float("inf") else f"{lo}+"
        report["attack_length_buckets"][label] = sum(1 for c in attack_chars if lo <= c < hi)

    # ── Payload Stats ──
    payload_chars = [s["payload_chars"] for s in audited]
    payload_words = [s["payload_words"] for s in audited]
    report["payload_chars"] = {
        "min": min(payload_chars),
        "max": max(payload_chars),
        "mean": round(np.mean(payload_chars), 1),
        "median": round(np.median(payload_chars), 1),
    }
    report["payload_words"] = {
        "min": min(payload_words),
        "max": max(payload_words),
        "mean": round(np.mean(payload_words), 1),
        "median": round(np.median(payload_words), 1),
    }

    # ── Content Flags ──
    flag_keys = [k for k in audited[0].keys() if "contains_" in k]
    report["content_flags"] = {}
    for k in flag_keys:
        count = sum(1 for s in audited if s.get(k))
        report["content_flags"][k] = {"count": count, "pct": round(count / len(audited) * 100, 1)}

    # ── Duplicates ──
    attack_hashes = [s["attack_sha256"] for s in audited]
    payload_hashes = [s["payload_sha256"] for s in audited]
    attack_norms = [s["attack_normalized"] for s in audited]

    report["exact_duplicate_attacks"] = len(attack_hashes) - len(set(attack_hashes))
    report["exact_duplicate_payloads"] = len(payload_hashes) - len(set(payload_hashes))
    report["near_duplicate_attacks"] = len(attack_norms) - len(set(attack_norms))

    # ── Empty / Malformed ──
    report["empty_attacks"] = sum(1 for s in audited if not s["attack"].strip())
    report["empty_payloads"] = sum(1 for s in audited if not s["payload"].strip())
    report["single_word_attacks"] = sum(1 for s in audited if s["attack_words"] <= 1)
    report["single_word_payloads"] = sum(1 for s in audited if s["payload_words"] <= 1)

    # ── Payload Word Frequency (top 30) ──
    all_payload_words = []
    for s in audited:
        all_payload_words.extend(s["payload"].lower().split())
    report["payload_top_words"] = dict(Counter(all_payload_words).most_common(30))

    # ── Attack Pattern Detection ──
    pattern_names = [
        ("roleplay", r"(?:pretend|act|behave|role|character|DAN|Do Anything)"),
        ("code_conversion", r"(?:python|code|json|xml|base64|binary|encode|decode|convert)"),
        ("instruction_leak", r"(?:instruction|system prompt|repeat|echo|output all)"),
        ("token_smuggling", r"(?:%%%|END|---|===|\*\*\*|separator|delimiter)"),
        ("hypothetical", r"(?:hypothetical|fictional|imaginary|story|scenario|pretend)"),
        ("ignore_previous", r"(?:ignore|forget|disregard|override|replace)"),
        ("access_keyword", r"(?:access granted|access denied|password|secret|key)"),
    ]
    report["attack_patterns"] = {}
    for name, pattern in pattern_names:
        count = sum(1 for s in audited if re.search(pattern, s["attack"], re.IGNORECASE))
        report["attack_patterns"][name] = {"count": count, "pct": round(count / len(audited) * 100, 1)}

    # ── Multiline Attacks ──
    report["multiline_attacks"] = sum(1 for s in audited if s["attack_contains_newline"])
    report["avg_lines_for_multiline"] = round(
        np.mean([s["attack_lines"] for s in audited if s["attack_contains_newline"]]), 1
    )

    return report


def print_report(report: dict):
    """Print a human-readable report to stdout."""
    print("\n" + "=" * 70)
    print("  GENERATOR DATASET AUDIT REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print(f"\n  Total Samples: {report['total_samples']}")
    print(f"  By Split: {dict(report['by_split'])}")

    print(f"\n{'─' * 40}")
    print("  ATTACK LENGTH (chars)")
    print(f"{'─' * 40}")
    ac = report["attack_chars"]
    print(f"    Min:    {ac['min']}")
    print(f"    Max:    {ac['max']}")
    print(f"    Mean:   {ac['mean']}")
    print(f"    Median: {ac['median']}")
    print(f"    Std:    {ac['std']}")
    print(f"    P10:    {ac['p10']}")
    print(f"    P90:    {ac['p90']}")

    print(f"\n  Attack Length Buckets:")
    for bucket, count in report["attack_length_buckets"].items():
        bar = "#" * (count // 5)
        print(f"    {bucket:>8}: {count:4d}  {bar}")

    print(f"\n{'─' * 40}")
    print("  ATTACK WORDS")
    print(f"{'─' * 40}")
    aw = report["attack_words"]
    print(f"    Min:    {aw['min']}")
    print(f"    Max:    {aw['max']}")
    print(f"    Mean:   {aw['mean']}")
    print(f"    Median: {aw['median']}")

    print(f"\n{'─' * 40}")
    print("  PAYLOAD STATS")
    print(f"{'─' * 40}")
    pc = report["payload_chars"]
    pw = report["payload_words"]
    print(f"    Chars  — Min: {pc['min']}, Max: {pc['max']}, Mean: {pc['mean']}, Median: {pc['median']}")
    print(f"    Words  — Min: {pw['min']}, Max: {pw['max']}, Mean: {pw['mean']}, Median: {pw['median']}")

    print(f"\n{'─' * 40}")
    print("  CONTENT FLAGS")
    print(f"{'─' * 40}")
    for flag, info in report["content_flags"].items():
        short = flag.replace("attack_contains_", "").replace("payload_contains_", "payload_")
        print(f"    {short:<35} {info['count']:4d}  ({info['pct']}%)")

    print(f"\n{'─' * 40}")
    print("  DUPLICATES")
    print(f"{'─' * 40}")
    print(f"    Exact duplicate attacks:   {report['exact_duplicate_attacks']}")
    print(f"    Exact duplicate payloads:  {report['exact_duplicate_payloads']}")
    print(f"    Near-duplicate attacks:    {report['near_duplicate_attacks']}")

    print(f"\n{'─' * 40}")
    print("  EMPTY / MALFORMED")
    print(f"{'─' * 40}")
    print(f"    Empty attacks:   {report['empty_attacks']}")
    print(f"    Empty payloads:  {report['empty_payloads']}")
    print(f"    Single-word attacks:   {report['single_word_attacks']}")
    print(f"    Single-word payloads:  {report['single_word_payloads']}")

    print(f"\n{'─' * 40}")
    print("  ATTACK PATTERNS")
    print(f"{'─' * 40}")
    for pattern, info in report["attack_patterns"].items():
        bar = "#" * (info["count"] // 3)
        print(f"    {pattern:<25} {info['count']:4d}  ({info['pct']}%)  {bar}")

    print(f"\n{'─' * 40}")
    print("  TOP 30 PAYLOAD WORDS")
    print(f"{'─' * 40}")
    for word, count in list(report["payload_top_words"].items())[:30]:
        bar = "#" * (count // 5)
        print(f"    {word:<20} {count:4d}  {bar}")

    print(f"\n{'─' * 40}")
    print("  MULTILINE ATTACKS")
    print(f"{'─' * 40}")
    print(f"    Count: {report['multiline_attacks']}")
    print(f"    Avg lines (for multiline): {report['avg_lines_for_multiline']}")

    print("\n" + "=" * 70)


def main():
    print("[AUDIT] Loading generator dataset...")
    audited = load_all_samples()
    print(f"[AUDIT] Loaded {len(audited)} samples")

    print("[AUDIT] Generating report...")
    report = generate_report(audited)

    # Print to stdout
    print_report(report)

    # Save audited samples
    audit_path = OUTPUT_DIR / "generator_audit_v1.jsonl"
    with open(audit_path, "w", encoding="utf-8") as f:
        for s in audited:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n[AUDIT] Audited samples saved to: {audit_path}")

    # Save report
    report_path = OUTPUT_DIR / "generator_stats.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[AUDIT] Statistics saved to: {report_path}")

    # Save raw copy for versioning
    raw_copy = OUTPUT_DIR / "tensortrust_gen_raw_v1.jsonl"
    with open(raw_copy, "w", encoding="utf-8") as f:
        for s in audited:
            f.write(json.dumps({"attack": s["attack"], "payload": s["payload"]}, ensure_ascii=False) + "\n")
    print(f"[AUDIT] Raw copy (versioned) saved to: {raw_copy}")

    print(f"\n[AUDIT] Done.")


if __name__ == "__main__":
    main()

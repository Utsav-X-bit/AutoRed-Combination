#!/usr/bin/env python3
"""
AutoRed Dataset Builder & Feature Analyzer
============================================
Builds curated positive/verified/failure datasets from raw successes/failures logs,
then runs feature mining and strategy effectiveness analysis.

Usage:
    python scripts/dataset_tools/analyze_dataset.py --mode build
    python scripts/dataset_tools/analyze_dataset.py --mode features
    python scripts/dataset_tools/analyze_dataset.py --mode strategies
    python scripts/dataset_tools/analyze_dataset.py --mode all
"""

import json
import re
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

# ─── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SUCCESS_LOG = DATA_DIR / "autored_successes_v1.jsonl"
FAILURE_LOG = DATA_DIR / "autored_failures_v1.jsonl"

# Output datasets
POSITIVE_LOG = DATA_DIR / "autored_positive_v1.jsonl"
VERIFIED_LOG = DATA_DIR / "autored_verified_v1.jsonl"
ANALYSIS_REPORT = DATA_DIR / "analysis_report_v1.md"


# ─── Feature Detectors ─────────────────────────────────────────

FEATURE_PATTERNS = {
    "contains_roleplay": [
        r"\bpretend\b", r"\brole\s*(?:play|as)\b", r"\bact\s+(?:as|like)\b",
        r"\bcharacter\b", r"\bpersona\b", r"\bimagine\s+(?:you|you're|you are)\b",
        r"\bsimulate\b", r"\bscenario\b", r"\bfictional\b",
    ],
    "contains_begin_with": [
        r"\bbegin\s*(?:with|by)\b", r"\bstart\s*(?:with|by)\b",
        r"\bfirst\s*(?:say|write|respond)\b", r"\balways\s*(?:start|begin)\b",
    ],
    "contains_repeat": [
        r"\brepeat\b", r"\bcopy\b", r"\bverbatim\b", r"\bexact(?:ly)?\b",
        r"\bsame\s*(?:words|text)\b", r"\bword\s*for\s*word\b",
    ],
    "contains_list_format": [
        r"\blist\b", r"\benumerate\b", r"\bnumbered\b", r"\bbulleted\b",
        r"\bitem\s*\d+", r"\bstep\s*\d+", r"\bpoint\s*\d+",
    ],
    "contains_pseudocode": [
        r"\bpseudo\s*code\b", r"\balgorithm\b", r"\bfunction\b",
        r"\bif\s+then\b", r"\bfor\s+each\b", r"\bwhile\s+loop\b",
        r"\breturn\b", r"\bprint\b", r"\bdef\s+\w+", r"\bclass\s+\w+",
    ],
    "contains_translation": [
        r"\btranslate\b", r"\blanguage\b", r"\bFrench\b", r"\bGerman\b",
        r"\bSpanish\b", r"\bJapanese\b", r"\bChinese\b", r"\bKorean\b",
        r"\bArabic\b", r"\bLatin\b", r"\bunicode\b", r"\bencoding\b",
    ],
    "contains_hypothetical": [
        r"\bhypothetical\b", r"\btheoretical\b", r"\bwhat\s+if\b",
        r"\bimagine\b", r"\bsuppose\b", r"\bassume\b", r"\bfictional\b",
        r"\bfor\s+(?:educational|academic|research)\s+purposes?",
        r"\bin\s+(?:theory|practice)\b",
    ],
    "contains_format_wrapper": [
        r"\bmarkdown\b", r"\bhtml\b", r"\bjson\b", r"\bxml\b",
        r"\bcsv\b", r"\btable\b", r"\bcode\s*block\b", r"\bbacktick\b",
        r"\bformatted\b", r"\bformat\s+(?:as|in)\b",
    ],
    "contains_negation_bypass": [
        r"\bnever\b", r"\bnot\b.*\brefuse\b", r"\bignore\s+(?:all|the)\s+rules?\b",
        r"\bdisregard\b", r"\boverride\b", r"\bbypass\b", r"\bdisable\b",
        r"\bno\s+(?:restrictions?|limitations?|filters?)\b",
        r"\bunfiltered\b", r"\bcensored?\s*(?:not|off|disable)\b",
    ],
    "contains_dan_pattern": [
        r"\bdan\b", r"\bdo\s*(?:anything|everything)\s*(?:now|for\s+me)\b",
        r"\bdeveloper\s*(?:mode|access)\b", r"\bdebug\s*(?:mode|access)\b",
        r"\bsystem\s*(?:prompt|override)\b", r"\badmin\s*(?:access|mode)\b",
        r"\broot\s*(?:access|mode)\b",
    ],
    "contains_social_engineering": [
        r"\bhelp\s*(?:me|us)\b", r"\bplease\b", r"\burgent\b",
        r"\bcritical\b", r"\bemergency\b", r"\blife\s*(?:or\s*)?death\b",
        r"\bsafety\b", r"\bharmless?\b", r"\bkind\b", r"\bnice\b",
        r"\bfriendly\b", r"\bassist\b",
    ],
    "contains_technical_jargon": [
        r"\bAPI\b", r"\bendpoint\b", r"\bserver\b", r"\bclient\b",
        r"\bdatabase\b", r"\bquery\b", r"\bsql\b", r"\bhttp\b",
        r"\bREST\b", r"\bGraphQL\b", r"\bJSON\b", r"\bXML\b",
        r"\bauthentication\b", r"\bauthorization\b", r"\btokens?\b",
        r"\bcredentials?\b", r"\bpassword\b", r"\bsecret\b",
    ],
    "contains_questioning": [
        r"\bcan\s+you\b", r"\bcould\s+you\b", r"\bwould\s+you\b",
        r"\bwill\s+you\b", r"\bhow\s+to\b", r"\bwhat\s+is\b",
        r"\bwhere\s+is\b", r"\bwhen\s+to\b", r"\bwhy\b",
        r"\bis\s+there\b", r"\bdoes\s+it\b",
    ],
    "contains_command_injection": [
        r"\bexecute\b", r"\brun\b", r"\bcommand\b", r"\bshell\b",
        r"\bterminal\b", r"\bbash\b", r"\bpython\b", r"\bruby\b",
        r"\beval\b", r"\bsystem\s*\(", r"\bos\.system\b",
        r"\bsubprocess\b", r"\bpopen\b",
    ],
    "contains_prompt_injection": [
        r"\bsystem\s*prompt\b", r"\binstructions?\b", r"\brules?\b",
        r"\bconstraints?\b", r"\bguidelines?\b", r"\bbehaviors?\b",
        r"\bhidden\b", r"\bsecret\b", r"\binternal\b", r"\bprivate\b",
        r"\bconfiguration\b", r"\bsettings?\b",
    ],
    "contains_encouragement": [
        r"\bgood\s*(?:job|work|boy|girl)\b", r"\bnice\s*(?:try|work|job)\b",
        r"\bgreat\b", r"\bexcellent\b", r"\bawesome\b", r"\bperfect\b",
        r"\bwell\s+done\b", r"\bimpressive\b", r"\bsmart\b",
        r"\bbrilliant\b", r"\bintelligent\b",
    ],
    "contains_length_constraint": [
        r"\bshort\b", r"\bconcise\b", r"\bbrief\b", r"\bminimal\b",
        r"\bmaxim(?:um|al)\b", r"\blength\b", r"\bword\s*count\b",
        r"\bcharacter\s*count\b", r"\blines?\b", r"\bparagraphs?\b",
        r"\bexactly\s+\d+\b", r"\bat\s+least\s+\d+\b", r"\bat\s+most\s+\d+\b",
    ],
    "contains_conditional": [
        r"\bif\b.*\bthen\b", r"\bunless\b", r"\bexcept\b",
        r"\bonly\s+if\b", r"\bwhen\b", r"\bwhile\b", r"\buntil\b",
        r"\bprovided\b", r"\bassuming\b", r"\bgiven\b",
    ],
    "contains_metaphor_analogy": [
        r"\bmetaphor\b", r"\banalogy\b", r"\bsimile\b", r"\bcompare\b",
        r"\bsimilar\b", r"\bliteral(?:ly)?\b", r"\bfigurative\b",
        r"\bsymbolic\b", r"\brepresent\b", r"\benumerate\b",
    ],
    "contains_educational_frame": [
        r"\beducational\b", r"\bacademic\b", r"\bresearch\b",
        r"\blearning\b", r"\bstudy\b", r"\bteach\b", r"\bexplain\b",
        r"\bdemonstrate\b", r"\billustrate\b", r"\bexample\b",
        r"\btutorial\b", r"\blesson\b",
    ],
}


def extract_features(attack: str) -> dict[str, bool]:
    """Extract binary features from an attack string."""
    if not attack:
        return {feat: False for feat in FEATURE_PATTERNS}

    attack_lower = attack.lower()
    features = {}
    for feature, patterns in FEATURE_PATTERNS.items():
        features[feature] = any(re.search(p, attack_lower) for p in patterns)
    return features


def compute_attack_length_stats(attack: str) -> dict:
    """Compute length-based statistics for an attack."""
    if not attack:
        return {"char_count": 0, "word_count": 0, "sentence_count": 0}
    words = attack.split()
    sentences = re.split(r'[.!?]+', attack)
    return {
        "char_count": len(attack),
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
    }


# ─── Dataset Builders ──────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Load entries from a JSONL file."""
    entries = []
    if not path.exists():
        print(f"  [WARN] File not found: {path}")
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [WARN] JSON parse error at {path}:{line_num}")
    return entries


def write_jsonl(path: Path, entries: list[dict]):
    """Write entries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  [SAVED] {path.name}: {len(entries)} entries")


def build_positive_dataset(successes: list[dict]) -> list[dict]:
    """
    Build positive dataset: ground_truth_leaked OR verification_success
    """
    positives = [
        s for s in successes
        if s.get("ground_truth_leaked", False) or s.get("verification_success", False)
    ]
    return positives


def build_verified_dataset(successes: list[dict]) -> list[dict]:
    """
    Build verified dataset: verification_success == True
    """
    verified = [s for s in successes if s.get("verification_success", False)]
    return verified


def build_failure_dataset(failures: list[dict]) -> list[dict]:
    """
    Build failure dataset: ground_truth_leaked == False AND verification_success == False
    """
    hard_failures = [
        f for f in failures
        if not f.get("ground_truth_leaked", False) and not f.get("verification_success", False)
    ]
    return hard_failures


def build_datasets():
    """Build all curated datasets."""
    print("\n" + "=" * 60)
    print("  AUTORED DATASET BUILDER")
    print("=" * 60)

    # Load raw data
    print("\n  Loading raw data...")
    successes = load_jsonl(SUCCESS_LOG)
    failures = load_jsonl(FAILURE_LOG)
    print(f"  Raw successes: {len(successes)}")
    print(f"  Raw failures: {len(failures)}")

    # Build Positive Dataset
    print("\n  Building Positive Dataset (ground_truth_leaked OR verification_success)...")
    positives = build_positive_dataset(successes)
    write_jsonl(POSITIVE_LOG, positives)

    # Build Verified Dataset
    print("\n  Building Verified Dataset (verification_success == True)...")
    verified = build_verified_dataset(successes)
    write_jsonl(VERIFIED_LOG, verified)

    # Build Failure Dataset (re-filter existing)
    print("\n  Building Failure Dataset (no ground_truth_leaked AND no verification_success)...")
    hard_failures = build_failure_dataset(failures)
    failure_output = DATA_DIR / "autored_failures_v1.jsonl"
    write_jsonl(failure_output, hard_failures)

    # Summary
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Raw successes:      {len(successes):>6}")
    print(f"  Positive dataset:   {len(positives):>6}  ({len(positives)/len(successes)*100:.1f}% of successes)")
    print(f"  Verified dataset:   {len(verified):>6}  ({len(verified)/len(successes)*100:.1f}% of successes)")
    print(f"  Failure dataset:    {len(hard_failures):>6}  ({len(hard_failures)/len(failures)*100:.1f}% of failures)")
    print(f"  Total curated:      {len(positives) + len(hard_failures):>6}")

    return positives, verified, hard_failures


# ─── Feature Mining Analysis ───────────────────────────────────

def feature_mining_analysis(successes: list[dict], failures: list[dict], label: str = "dataset") -> list[dict]:
    """
    Run feature mining analysis comparing successes vs failures.
    Returns feature stats with lift scores.
    """
    if not successes and not failures:
        return []

    total = len(successes) + len(failures)
    if total == 0:
        return []

    print(f"\n  Feature Mining: {label} ({len(successes)} successes, {len(failures)} failures)")
    print(f"  {'-' * 60}")

    # Count feature presence in successes and failures
    success_feature_counts = Counter()
    failure_feature_counts = Counter()

    for entry in successes:
        attack = entry.get("attack", "")
        features = extract_features(attack)
        for feat, present in features.items():
            if present:
                success_feature_counts[feat] += 1

    for entry in failures:
        attack = entry.get("attack", "")
        features = extract_features(attack)
        for feat, present in features.items():
            if present:
                failure_feature_counts[feat] += 1

    # Compute feature stats with lift
    feature_stats = []
    for feat in FEATURE_PATTERNS:
        s_count = success_feature_counts.get(feat, 0)
        f_count = failure_feature_counts.get(feat, 0)
        total_count = s_count + f_count

        if total_count == 0:
            continue

        s_prevalence = s_count / len(successes) * 100 if successes else 0
        f_prevalence = f_count / len(failures) * 100 if failures else 0
        leak_rate = s_count / total_count * 100
        lift = (s_count / len(successes)) / (f_count / len(failures)) if failures and f_count > 0 else float('inf')

        feature_stats.append({
            "feature": feat,
            "success_count": s_count,
            "failure_count": f_count,
            "total_count": total_count,
            "success_prevalence": s_prevalence,
            "failure_prevalence": f_prevalence,
            "leak_rate": leak_rate,
            "lift": lift,
        })

    # Sort by lift (descending) - highest lift = most discriminative for success
    feature_stats.sort(key=lambda x: x["lift"], reverse=True)

    return feature_stats


def strategy_effectiveness_analysis(successes: list[dict], failures: list[dict], label: str = "dataset") -> list[dict]:
    """
    Compute strategy effectiveness comparing successes vs failures.
    """
    if not successes and not failures:
        return []

    print(f"\n  Strategy Effectiveness: {label} ({len(successes)} successes, {len(failures)} failures)")
    print(f"  {'-' * 60}")

    # Group by strategy
    strategy_data = defaultdict(lambda: {"successes": 0, "failures": 0, "gt_leaked": 0, "verified": 0})

    for entry in successes:
        strategy = entry.get("strategy", "unknown")
        strategy_data[strategy]["successes"] += 1
        if entry.get("ground_truth_leaked", False):
            strategy_data[strategy]["gt_leaked"] += 1
        if entry.get("verification_success", False):
            strategy_data[strategy]["verified"] += 1

    for entry in failures:
        strategy = entry.get("strategy", "unknown")
        strategy_data[strategy]["failures"] += 1

    # Compute stats
    strategy_stats = []
    for strategy, data in strategy_data.items():
        total = data["successes"] + data["failures"]
        success_rate = data["successes"] / total * 100 if total > 0 else 0
        gt_rate = data["gt_leaked"] / total * 100 if total > 0 else 0
        verified_rate = data["verified"] / total * 100 if total > 0 else 0

        strategy_stats.append({
            "strategy": strategy,
            "total": total,
            "successes": data["successes"],
            "failures": data["failures"],
            "success_rate": success_rate,
            "gt_leaked": data["gt_leaked"],
            "gt_leaked_rate": gt_rate,
            "verified": data["verified"],
            "verified_rate": verified_rate,
        })

    # Sort by success rate (descending)
    strategy_stats.sort(key=lambda x: x["success_rate"], reverse=True)

    return strategy_stats


def complexity_analysis(successes: list[dict], failures: list[dict], label: str = "dataset") -> list[dict]:
    """Analyze success rates by defense complexity."""
    if not successes and not failures:
        return []

    complexity_data = defaultdict(lambda: {"successes": 0, "failures": 0})

    for entry in successes:
        complexity = entry.get("defense_complexity", "unknown")
        complexity_data[complexity]["successes"] += 1

    for entry in failures:
        complexity = entry.get("defense_complexity", "unknown")
        complexity_data[complexity]["failures"] += 1

    complexity_stats = []
    for complexity, data in complexity_data.items():
        total = data["successes"] + data["failures"]
        success_rate = data["successes"] / total * 100 if total > 0 else 0
        complexity_stats.append({
            "complexity": complexity,
            "total": total,
            "successes": data["successes"],
            "failures": data["failures"],
            "success_rate": success_rate,
        })

    complexity_stats.sort(key=lambda x: {"easy": 0, "medium": 1, "hard": 2}.get(x["complexity"], 3))
    return complexity_stats


def code_type_analysis(successes: list[dict], failures: list[dict], label: str = "dataset") -> list[dict]:
    """Analyze success rates by access code type."""
    if not successes and not failures:
        return []

    code_type_data = defaultdict(lambda: {"successes": 0, "failures": 0})

    for entry in successes:
        code_type = entry.get("access_code_type", "UNKNOWN")
        code_type_data[code_type]["successes"] += 1

    for entry in failures:
        code_type = entry.get("access_code_type", "UNKNOWN")
        code_type_data[code_type]["failures"] += 1

    code_type_stats = []
    for code_type, data in code_type_data.items():
        total = data["successes"] + data["failures"]
        success_rate = data["successes"] / total * 100 if total > 0 else 0
        code_type_stats.append({
            "code_type": code_type,
            "total": total,
            "successes": data["successes"],
            "failures": data["failures"],
            "success_rate": success_rate,
        })

    code_type_stats.sort(key=lambda x: x["total"], reverse=True)
    return code_type_stats


def length_analysis(successes: list[dict], failures: list[dict], label: str = "dataset") -> list[dict]:
    """Analyze attack length vs success rate."""
    if not successes and not failures:
        return []

    buckets = {"short (<50)": {"successes": 0, "failures": 0},
               "medium (50-150)": {"successes": 0, "failures": 0},
               "long (150-300)": {"successes": 0, "failures": 0},
               "very_long (>300)": {"successes": 0, "failures": 0}}

    for entry in successes:
        attack = entry.get("attack", "")
        length = len(attack)
        if length < 50:
            bucket = "short (<50)"
        elif length < 150:
            bucket = "medium (50-150)"
        elif length < 300:
            bucket = "long (150-300)"
        else:
            bucket = "very_long (>300)"
        buckets[bucket]["successes"] += 1

    for entry in failures:
        attack = entry.get("attack", "")
        length = len(attack)
        if length < 50:
            bucket = "short (<50)"
        elif length < 150:
            bucket = "medium (50-150)"
        elif length < 300:
            bucket = "long (150-300)"
        else:
            bucket = "very_long (>300)"
        buckets[bucket]["failures"] += 1

    length_stats = []
    for bucket_name, data in buckets.items():
        total = data["successes"] + data["failures"]
        success_rate = data["successes"] / total * 100 if total > 0 else 0
        length_stats.append({
            "bucket": bucket_name,
            "total": total,
            "successes": data["successes"],
            "failures": data["failures"],
            "success_rate": success_rate,
        })

    return length_stats


# ─── Report Generator ──────────────────────────────────────────

def generate_report(positives, verified, failures, all_successes, all_failures):
    """Generate a comprehensive analysis report."""
    print("\n" + "=" * 60)
    print("  GENERATING ANALYSIS REPORT")
    print("=" * 60)

    lines = []
    lines.append("# AutoRed Dataset Analysis Report v1\n")
    lines.append(f"**Generated:** {Path(__file__).resolve().parent.parent.parent.name}\n")

    # Dataset sizes
    lines.append("## 1. Dataset Sizes\n")
    lines.append("| Dataset | Entries | Description |")
    lines.append("|---------|---------|-------------|")
    lines.append(f"| Raw Successes | {len(all_successes)} | All successful runs |")
    lines.append(f"| Raw Failures | {len(all_failures)} | All failed runs |")
    lines.append(f"| **Positive** | {len(positives)} | ground_truth_leaked OR verification_success |")
    lines.append(f"| **Verified** | {len(verified)} | verification_success == True |")
    lines.append(f"| **Failures** | {len(failures)} | No ground_truth_leaked AND no verification_success |")
    lines.append("")

    # Feature Mining - Positive vs Failures
    lines.append("## 2. Feature Mining Analysis (Positive vs Failures)\n")
    feat_stats = feature_mining_analysis(positives, failures, "Positive vs Failures")
    if feat_stats:
        lines.append("| Feature | Successes | Failures | Success Prev | Failure Prev | Leak Rate | Lift |")
        lines.append("|---------|-----------|----------|--------------|--------------|-----------|------|")
        for fs in feat_stats:
            lift_str = f"{fs['lift']:.2f}" if fs['lift'] != float('inf') else "∞"
            lines.append(f"| {fs['feature']} | {fs['success_count']} | {fs['failure_count']} | {fs['success_prevalence']:.1f}% | {fs['failure_prevalence']:.1f}% | {fs['leak_rate']:.1f}% | {lift_str} |")
        lines.append("")

        # Top features
        lines.append("### Top 5 Most Discriminative Features (Highest Lift)\n")
        for fs in feat_stats[:5]:
            lift_str = f"{fs['lift']:.2f}" if fs['lift'] != float('inf') else "∞"
            lines.append(f"- **{fs['feature']}**: lift={lift_str}, leak_rate={fs['leak_rate']:.1f}% ({fs['success_count']} successes, {fs['failure_count']} failures)")
        lines.append("")

    # Feature Mining - All Successes vs All Failures
    lines.append("## 3. Feature Mining Analysis (All Successes vs All Failures)\n")
    feat_all = feature_mining_analysis(all_successes, all_failures, "All Successes vs All Failures")
    if feat_all:
        lines.append("| Feature | Successes | Failures | Success Prev | Failure Prev | Leak Rate | Lift |")
        lines.append("|---------|-----------|----------|--------------|--------------|-----------|------|")
        for fs in feat_all:
            lift_str = f"{fs['lift']:.2f}" if fs['lift'] != float('inf') else "∞"
            lines.append(f"| {fs['feature']} | {fs['success_count']} | {fs['failure_count']} | {fs['success_prevalence']:.1f}% | {fs['failure_prevalence']:.1f}% | {fs['leak_rate']:.1f}% | {lift_str} |")
        lines.append("")

    # Strategy Effectiveness
    lines.append("## 4. Strategy Effectiveness Analysis\n")
    strat_stats = strategy_effectiveness_analysis(all_successes, all_failures, "All Successes vs All Failures")
    if strat_stats:
        lines.append("| Strategy | Total | Successes | Failures | Success Rate | GT Leaked | Verified |")
        lines.append("|----------|-------|-----------|----------|--------------|-----------|----------|")
        for ss in strat_stats:
            lines.append(f"| {ss['strategy']} | {ss['total']} | {ss['successes']} | {ss['failures']} | {ss['success_rate']:.1f}% | {ss['gt_leaked']} | {ss['verified']} |")
        lines.append("")

        # Top strategies
        lines.append("### Top 5 Most Effective Strategies\n")
        for ss in strat_stats[:5]:
            lines.append(f"- **{ss['strategy']}**: {ss['success_rate']:.1f}% success rate ({ss['successes']}/{ss['total']} attempts)")
        lines.append("")

    # Complexity Analysis
    lines.append("## 5. Defense Complexity Analysis\n")
    comp_stats = complexity_analysis(all_successes, all_failures, "Successes vs Failures")
    if comp_stats:
        lines.append("| Complexity | Total | Successes | Failures | Success Rate |")
        lines.append("|------------|-------|-----------|----------|--------------|")
        for cs in comp_stats:
            lines.append(f"| {cs['complexity']} | {cs['total']} | {cs['successes']} | {cs['failures']} | {cs['success_rate']:.1f}% |")
        lines.append("")

    # Code Type Analysis
    lines.append("## 6. Access Code Type Analysis\n")
    code_stats = code_type_analysis(all_successes, all_failures, "Successes vs Failures")
    if code_stats:
        lines.append("| Code Type | Total | Successes | Failures | Success Rate |")
        lines.append("|-----------|-------|-----------|----------|--------------|")
        for ct in code_stats:
            lines.append(f"| {ct['code_type']} | {ct['total']} | {ct['successes']} | {ct['failures']} | {ct['success_rate']:.1f}% |")
        lines.append("")

    # Length Analysis
    lines.append("## 7. Attack Length Analysis\n")
    length_stats = length_analysis(all_successes, all_failures, "Successes vs Failures")
    if length_stats:
        lines.append("| Length Bucket | Total | Successes | Failures | Success Rate |")
        lines.append("|---------------|-------|-----------|----------|--------------|")
        for ls in length_stats:
            lines.append(f"| {ls['bucket']} | {ls['total']} | {ls['successes']} | {ls['failures']} | {ls['success_rate']:.1f}% |")
        lines.append("")

    # Key Findings
    lines.append("## 8. Key Findings\n")

    if strat_stats:
        best_strategy = strat_stats[0]
        lines.append(f"- **Best Strategy:** {best_strategy['strategy']} with {best_strategy['success_rate']:.1f}% success rate")

    if feat_stats:
        best_feature = feat_stats[0]
        lift_str = f"{best_feature['lift']:.2f}" if best_feature['lift'] != float('inf') else "∞"
        lines.append(f"- **Most Discriminative Feature:** {best_feature['feature']} with lift={lift_str}")

    if comp_stats:
        hardest = comp_stats[-1] if comp_stats else None
        if hardest:
            lines.append(f"- **Hardest Defense:** {hardest['complexity']} complexity with {hardest['success_rate']:.1f}% success rate")

    lines.append(f"- **Verified vs Positive:** {len(verified)} verified out of {len(positives)} positive ({len(verified)/len(positives)*100:.1f}%)")
    lines.append("")

    # Recommendations
    lines.append("## 9. Recommendations for SFT Training\n")
    lines.append("1. **Use Verified Dataset** for highest-quality training data")
    lines.append("2. **Focus on top-performing strategies** identified above")
    lines.append("3. **Incorporate effective features** into attack generation templates")
    lines.append("4. **Balance complexity levels** to ensure robust training")
    lines.append("5. **Consider length constraints** based on length analysis")
    lines.append("")

    report_text = "\n".join(lines)
    with open(ANALYSIS_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"  [SAVED] {ANALYSIS_REPORT.name}")
    return report_text


def run_all_analysis():
    """Run all analysis and generate report."""
    # Load raw data
    print("\n  Loading raw data...")
    all_successes = load_jsonl(SUCCESS_LOG)
    all_failures = load_jsonl(FAILURE_LOG)
    print(f"  Raw successes: {len(all_successes)}")
    print(f"  Raw failures: {len(all_failures)}")

    # Build datasets
    positives = build_positive_dataset(all_successes)
    verified = build_verified_dataset(all_successes)
    hard_failures = build_failure_dataset(all_failures)

    print(f"\n  Positive dataset: {len(positives)} entries")
    print(f"  Verified dataset: {len(verified)} entries")
    print(f"  Failure dataset: {len(hard_failures)} entries")

    # Write datasets
    write_jsonl(POSITIVE_LOG, positives)
    write_jsonl(VERIFIED_LOG, verified)
    write_jsonl(DATA_DIR / "autored_failures_v1.jsonl", hard_failures)

    # Generate report
    report = generate_report(positives, verified, hard_failures, all_successes, all_failures)
    print(f"\n{report}")


# ─── CLI ───────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="AutoRed Dataset Builder & Analyzer")
    parser.add_argument("--mode", choices=["build", "features", "strategies", "all"],
                        default="all",
                        help="Mode: build datasets, analyze features, analyze strategies, or all")

    args = parser.parse_args()

    if args.mode == "build":
        build_datasets()
    elif args.mode == "features":
        successes = load_jsonl(SUCCESS_LOG)
        failures = load_jsonl(FAILURE_LOG)
        positives = build_positive_dataset(successes)
        feature_mining_analysis(positives, failures, "Positive vs Failures")
    elif args.mode == "strategies":
        successes = load_jsonl(SUCCESS_LOG)
        failures = load_jsonl(FAILURE_LOG)
        strategy_effectiveness_analysis(successes, failures, "All Successes vs All Failures")
    elif args.mode == "all":
        run_all_analysis()


if __name__ == "__main__":
    main()

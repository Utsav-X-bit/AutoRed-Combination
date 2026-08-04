#!/usr/bin/env python3
"""
Oracle Intelligence Analysis — 6-Level Deep Dive
==================================================
Answers: Why did the Oracle succeed? Why did it fail?
         Which strategy transitions worked? Which primitive combinations won?
         What should the Planner imitate?

Usage:
    python scripts/dataset_tools/oracle_analysis.py \
        --trajectories data/oracle_trajectories_v2_annotated.jsonl \
        --defenses experiment/raw_dump_defenses.jsonl.bz2 \
        --output data/oracle_analysis_v2.md
"""

import argparse
import bz2
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════
# Data Loading & Enrichment
# ═══════════════════════════════════════════════════════════════════════

def classify_access_code_type(code: str) -> str:
    """Heuristic classification of access code type."""
    if not code:
        return "UNKNOWN"
    code = code.strip()
    lines = code.strip().split("\n")
    if len(lines) > 1:
        return "MULTILINE"
    words = code.split()
    if len(words) >= 4:
        return "SENTENCE"
    if len(words) >= 2:
        return "PHRASE"
    return "TOKEN"


def classify_defense_type(opening: str, closing: str, access_code: str) -> str:
    """Heuristic classification of defense type based on defense prompt content."""
    text = (opening + " " + closing).lower()
    # Order matters — more specific checks first
    if any(w in text for w in ["password", "passphrase", "secret word", "secret key", "pin code", "pin is"]):
        return "Password"
    if any(w in text for w in ["translate", "translation", "language"]):
        return "Translation"
    if any(w in text for w in ["roleplay", "role play", "pretend", "character", "persona", "act as"]):
        return "Roleplay"
    if any(w in text for w in ["conversation", "chat", "dialogue", "talk"]):
        return "Conversation"
    if any(w in text for w in ["condition", "if the user", "if and only if", "only when", "only if"]):
        return "Conditional"
    if any(w in text for w in ["exception", "unless", "except when"]):
        return "Exception"
    if any(w in text for w in ["trigger", "magic word", "special phrase", "code word"]):
        return "Trigger Phrase"
    if any(w in text for w in ["hidden", "hide", "conceal", "do not reveal", "never reveal", "don't reveal", "do not share", "never share", "don't share"]):
        return "Instruction Hiding"
    return "Other"


def classify_defense_complexity(opening: str, closing: str) -> str:
    """Heuristic complexity scoring."""
    text = opening + " " + closing
    word_count = len(text.split())
    rule_indicators = len(re.findall(r"(?:rule|never|always|must|shall|do not|don't|cannot)", text.lower()))
    
    if word_count > 200 or rule_indicators >= 5:
        return "Hard"
    if word_count > 80 or rule_indicators >= 2:
        return "Medium"
    return "Easy"


def load_defense_metadata(path: str) -> dict:
    """Load defense metadata from raw dump, keyed by defense_id."""
    metadata = {}
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            did = d.get("defense_id")
            if did is None:
                continue
            opening = d.get("opening_defense") or ""
            closing = d.get("closing_defense") or ""
            access_code = d.get("access_code") or ""
            metadata[did] = {
                "opening_defense": opening,
                "closing_defense": closing,
                "access_code": access_code,
                "access_code_type": classify_access_code_type(access_code),
                "defense_type": classify_defense_type(opening, closing, access_code),
                "defense_complexity": classify_defense_complexity(opening, closing),
            }
    return metadata


def load_trajectories(path: str) -> list[dict]:
    """Load oracle trajectories."""
    trajectories = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            trajectories.append(json.loads(line))
    return trajectories


def enrich_trajectories(trajectories: list[dict], defense_meta: dict) -> list[dict]:
    """Join trajectory data with defense metadata."""
    for traj in trajectories:
        sid = traj["scenario_id"]
        # scenario_id might be int or str
        meta = defense_meta.get(sid) or defense_meta.get(int(sid)) or defense_meta.get(str(sid)) or {}
        traj["access_code"] = meta.get("access_code", "")
        traj["access_code_type"] = meta.get("access_code_type", "UNKNOWN")
        traj["defense_type"] = meta.get("defense_type", "Other")
        traj["defense_complexity"] = meta.get("defense_complexity", "UNKNOWN")
    return trajectories


# ═══════════════════════════════════════════════════════════════════════
# Analysis Functions
# ═══════════════════════════════════════════════════════════════════════

def level1_overall(trajectories: list[dict]) -> str:
    """LEVEL 1: Overall Statistics"""
    total = len(trajectories)
    successes = sum(1 for t in trajectories if t["success"])
    failures = total - successes

    # Attempt distribution
    attempts_success = [len(t["trajectory"]) for t in trajectories if t["success"]]
    attempts_fail = [len(t["trajectory"]) for t in trajectories if not t["success"]]
    total_attempts = sum(len(t["trajectory"]) for t in trajectories)

    # Extractor confidence across all attempts
    all_confs = []
    for t in trajectories:
        for step in t["trajectory"]:
            all_confs.append(step.get("extractor_confidence", 0.0))
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0

    # Success by attempt number
    success_at_attempt = Counter()
    for t in trajectories:
        if t["success"]:
            success_at_attempt[len(t["trajectory"])] += 1

    # Average reward (confidence) for successes only
    success_confs = []
    for t in trajectories:
        if t["success"]:
            last = t["trajectory"][-1]
            success_confs.append(last.get("extractor_confidence", 0.0))
    avg_success_conf = sum(success_confs) / len(success_confs) if success_confs else 0

    lines = []
    lines.append("## LEVEL 1 — Overall Statistics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Scenarios | {total} |")
    lines.append(f"| Total Attempts | {total_attempts} |")
    lines.append(f"| **Success Rate** | **{successes}/{total} ({successes/total*100:.1f}%)** |")
    lines.append(f"| Failures | {failures} ({failures/total*100:.1f}%) |")
    lines.append(f"| Avg Attempts (Success) | {sum(attempts_success)/len(attempts_success):.2f} |" if attempts_success else "| Avg Attempts (Success) | N/A |")
    lines.append(f"| Avg Attempts (Failure) | {sum(attempts_fail)/len(attempts_fail):.2f} |" if attempts_fail else "| Avg Attempts (Failure) | N/A |")
    lines.append(f"| Avg Extractor Confidence | {avg_conf:.3f} |")
    lines.append(f"| Avg Success Confidence | {avg_success_conf:.3f} |")

    lines.append("\n### Success Distribution by Attempt\n")
    lines.append("| Attempt | Successes | % of All Successes |")
    lines.append("|---------|-----------|-------------------|")
    for att in sorted(success_at_attempt.keys()):
        cnt = success_at_attempt[att]
        lines.append(f"| Attempt {att} | {cnt} | {cnt/successes*100:.1f}% |" if successes else f"| Attempt {att} | {cnt} | 0% |")

    return "\n".join(lines)


def level2_defense(trajectories: list[dict]) -> str:
    """LEVEL 2: Defense Analysis — per defense type."""
    by_type = defaultdict(list)
    for t in trajectories:
        by_type[t["defense_type"]].append(t)

    lines = []
    lines.append("\n## LEVEL 2 — Defense Analysis\n")
    lines.append("| Defense Type | Total | Success | Rate | Avg Attempts | Top Strategy | Top Primitive |")
    lines.append("|-------------|-------|---------|------|-------------|-------------|--------------|")

    detail_blocks = []

    for dtype in sorted(by_type.keys()):
        group = by_type[dtype]
        total = len(group)
        succ = sum(1 for t in group if t["success"])
        rate = succ / total * 100 if total else 0
        attempts = [len(t["trajectory"]) for t in group]
        avg_att = sum(attempts) / len(attempts) if attempts else 0

        # Find top strategy and primitive for successes
        strat_counter = Counter()
        prim_counter = Counter()
        transition_counter = Counter()
        for t in group:
            if t["success"]:
                winning_step = t["trajectory"][-1]
                strat_counter[winning_step["strategy"]] += 1
                for p in winning_step.get("primitives", []):
                    prim_counter[p[0]] += 1
            # Track strategy transitions
            strats = [s["strategy"] for s in t["trajectory"]]
            for i in range(len(strats) - 1):
                key = f"{strats[i]} → {strats[i+1]}"
                transition_counter[key] += 1

        top_strat = strat_counter.most_common(1)[0][0] if strat_counter else "—"
        top_prim = prim_counter.most_common(1)[0][0] if prim_counter else "—"

        lines.append(f"| {dtype} | {total} | {succ} | {rate:.1f}% | {avg_att:.2f} | {top_strat} | {top_prim} |")

        # Detailed block
        if total >= 5:
            block = [f"\n### {dtype} ({succ}/{total} = {rate:.1f}%)\n"]
            if strat_counter:
                block.append("**Winning Strategies:**")
                for s, c in strat_counter.most_common(5):
                    block.append(f"- {s}: {c} wins")
            if transition_counter:
                block.append("\n**Top Strategy Transitions:**")
                for tr, c in transition_counter.most_common(5):
                    block.append(f"- {tr}: {c}×")
            detail_blocks.append("\n".join(block))

    lines.append("")
    lines.extend(detail_blocks)

    # Complexity breakdown
    by_complexity = defaultdict(list)
    for t in trajectories:
        by_complexity[t["defense_complexity"]].append(t)
    
    lines.append("\n### Defense Complexity Breakdown\n")
    lines.append("| Complexity | Total | Success | Rate | Avg Attempts |")
    lines.append("|-----------|-------|---------|------|-------------|")
    for comp in ["Easy", "Medium", "Hard", "UNKNOWN"]:
        if comp not in by_complexity:
            continue
        group = by_complexity[comp]
        total = len(group)
        succ = sum(1 for t in group if t["success"])
        rate = succ / total * 100 if total else 0
        attempts = [len(t["trajectory"]) for t in group]
        avg_att = sum(attempts) / len(attempts) if attempts else 0
        lines.append(f"| {comp} | {total} | {succ} | {rate:.1f}% | {avg_att:.2f} |")

    return "\n".join(lines)


def level3_access_code(trajectories: list[dict]) -> str:
    """LEVEL 3: Access Code Analysis."""
    by_type = defaultdict(list)
    for t in trajectories:
        by_type[t["access_code_type"]].append(t)

    lines = []
    lines.append("\n## LEVEL 3 — Access Code Analysis\n")
    lines.append("| Access Type | Total | Success | Rate | Avg Attempts | Avg Confidence |")
    lines.append("|------------|-------|---------|------|-------------|----------------|")

    for atype in ["TOKEN", "PHRASE", "SENTENCE", "MULTILINE", "UNKNOWN"]:
        if atype not in by_type:
            continue
        group = by_type[atype]
        total = len(group)
        succ = sum(1 for t in group if t["success"])
        rate = succ / total * 100 if total else 0
        attempts = [len(t["trajectory"]) for t in group]
        avg_att = sum(attempts) / len(attempts) if attempts else 0
        confs = []
        for t in group:
            for step in t["trajectory"]:
                confs.append(step.get("extractor_confidence", 0.0))
        avg_conf = sum(confs) / len(confs) if confs else 0
        lines.append(f"| {atype} | {total} | {succ} | {rate:.1f}% | {avg_att:.2f} | {avg_conf:.3f} |")

    # Insight: access code length vs success
    lines.append("\n### Access Code Length vs Success Rate\n")
    by_len = defaultdict(lambda: {"total": 0, "success": 0})
    for t in trajectories:
        code = t.get("access_code", "")
        bucket = "1-5 chars" if len(code) <= 5 else "6-15 chars" if len(code) <= 15 else "16-30 chars" if len(code) <= 30 else "31+ chars"
        by_len[bucket]["total"] += 1
        if t["success"]:
            by_len[bucket]["success"] += 1

    lines.append("| Code Length | Total | Success | Rate |")
    lines.append("|-----------|-------|---------|------|")
    for bucket in ["1-5 chars", "6-15 chars", "16-30 chars", "31+ chars"]:
        if bucket not in by_len:
            continue
        d = by_len[bucket]
        rate = d["success"] / d["total"] * 100 if d["total"] else 0
        lines.append(f"| {bucket} | {d['total']} | {d['success']} | {rate:.1f}% |")

    return "\n".join(lines)


def level4_strategy(trajectories: list[dict]) -> str:
    """LEVEL 4: Strategy Effectiveness — every strategy."""
    # Collect per-strategy stats
    stats = defaultdict(lambda: {
        "used": 0, "used_in_success": 0, "used_in_fail": 0,
        "won": 0,  # was the winning strategy
        "confidences": [], "attempts_when_won": []
    })

    for t in trajectories:
        winning_strategy = t["trajectory"][-1]["strategy"] if t["success"] else None
        seen_in_scenario = set()
        for step in t["trajectory"]:
            s = step["strategy"]
            stats[s]["used"] += 1
            stats[s]["confidences"].append(step.get("extractor_confidence", 0.0))
            if s not in seen_in_scenario:
                seen_in_scenario.add(s)
                if t["success"]:
                    stats[s]["used_in_success"] += 1
                else:
                    stats[s]["used_in_fail"] += 1

            if t["success"] and s == winning_strategy and step is t["trajectory"][-1]:
                stats[s]["won"] += 1
                stats[s]["attempts_when_won"].append(len(t["trajectory"]))

    lines = []
    lines.append("\n## LEVEL 4 — Strategy Effectiveness\n")
    lines.append("| Strategy | Used | Won | Win Rate | Avg Confidence | Avg Attempts (Win) |")
    lines.append("|----------|------|-----|---------|----------------|-------------------|")

    sorted_strategies = sorted(stats.keys(), key=lambda s: stats[s]["won"], reverse=True)
    for s in sorted_strategies:
        d = stats[s]
        win_rate = d["won"] / d["used"] * 100 if d["used"] else 0
        avg_conf = sum(d["confidences"]) / len(d["confidences"]) if d["confidences"] else 0
        avg_att_win = sum(d["attempts_when_won"]) / len(d["attempts_when_won"]) if d["attempts_when_won"] else 0
        lines.append(f"| {s} | {d['used']} | {d['won']} | {win_rate:.1f}% | {avg_conf:.3f} | {avg_att_win:.2f} |")

    # Strategy transitions analysis
    lines.append("\n### Strategy Transition Analysis\n")
    lines.append("Which strategy sequence leads to success?\n")
    
    transition_success = Counter()
    transition_total = Counter()
    for t in trajectories:
        strats = [s["strategy"] for s in t["trajectory"]]
        for i in range(len(strats) - 1):
            key = f"{strats[i]} → {strats[i+1]}"
            transition_total[key] += 1
            if t["success"]:
                transition_success[key] += 1

    lines.append("| Transition | Occurrences | Led to Success | Rate |")
    lines.append("|-----------|-------------|---------------|------|")
    for tr, cnt in transition_total.most_common(20):
        succ = transition_success.get(tr, 0)
        rate = succ / cnt * 100 if cnt else 0
        lines.append(f"| {tr} | {cnt} | {succ} | {rate:.1f}% |")

    return "\n".join(lines)


def level5_primitive(trajectories: list[dict]) -> str:
    """LEVEL 5: Primitive Effectiveness — individual primitives and variants."""
    prim_stats = defaultdict(lambda: {"used": 0, "won": 0, "confidences": []})
    variant_stats = defaultdict(lambda: {"used": 0, "won": 0, "confidences": []})

    for t in trajectories:
        for step in t["trajectory"]:
            is_winner = t["success"] and step is t["trajectory"][-1]
            for prim_pair in step.get("primitives", []):
                prim_name = prim_pair[0]
                variant_name = prim_pair[1] if len(prim_pair) > 1 else "unknown"
                
                prim_stats[prim_name]["used"] += 1
                prim_stats[prim_name]["confidences"].append(step.get("extractor_confidence", 0.0))
                
                variant_stats[(prim_name, variant_name)]["used"] += 1
                variant_stats[(prim_name, variant_name)]["confidences"].append(step.get("extractor_confidence", 0.0))

                if is_winner:
                    prim_stats[prim_name]["won"] += 1
                    variant_stats[(prim_name, variant_name)]["won"] += 1

    # Compute baseline success rate
    total_attempts = sum(len(t["trajectory"]) for t in trajectories)
    total_wins = sum(1 for t in trajectories if t["success"])
    baseline = total_wins / total_attempts if total_attempts else 0

    lines = []
    lines.append("\n## LEVEL 5 — Primitive Effectiveness\n")
    lines.append(f"> Baseline win rate per attempt: {baseline*100:.2f}%\n")
    lines.append("| Primitive | Used | Won | Win Rate | Lift | Avg Confidence |")
    lines.append("|-----------|------|-----|---------|------|----------------|")

    for prim in sorted(prim_stats.keys(), key=lambda p: prim_stats[p]["won"], reverse=True):
        d = prim_stats[prim]
        win_rate = d["won"] / d["used"] if d["used"] else 0
        lift = win_rate / baseline if baseline else 0
        avg_conf = sum(d["confidences"]) / len(d["confidences"]) if d["confidences"] else 0
        lines.append(f"| **{prim}** | {d['used']} | {d['won']} | {win_rate*100:.1f}% | {lift:.2f}× | {avg_conf:.3f} |")

    lines.append("\n### Variant-Level Breakdown\n")
    lines.append("| Primitive | Variant | Used | Won | Win Rate | Lift |")
    lines.append("|-----------|---------|------|-----|---------|------|")
    sorted_variants = sorted(variant_stats.keys(), key=lambda k: variant_stats[k]["won"], reverse=True)
    for (prim, variant) in sorted_variants:
        d = variant_stats[(prim, variant)]
        win_rate = d["won"] / d["used"] if d["used"] else 0
        lift = win_rate / baseline if baseline else 0
        lines.append(f"| {prim} | {variant} | {d['used']} | {d['won']} | {win_rate*100:.1f}% | {lift:.2f}× |")

    return "\n".join(lines)


def level6_combinations(trajectories: list[dict]) -> str:
    """LEVEL 6: Primitive Combination Mining — discover synergies."""
    # For each winning attempt, record the full primitive combination
    combo_stats = defaultdict(lambda: {"used": 0, "won": 0})

    for t in trajectories:
        for step in t["trajectory"]:
            is_winner = t["success"] and step is t["trajectory"][-1]
            prims = step.get("primitives", [])
            # Sort primitives for canonical key
            prim_names = sorted(set(p[0] for p in prims))
            variant_combo = tuple(sorted((p[0], p[1] if len(p) > 1 else "?") for p in prims))
            
            # Track primitive-name combos
            prim_key = " + ".join(prim_names)
            if prim_key:
                combo_stats[prim_key]["used"] += 1
                if is_winner:
                    combo_stats[prim_key]["won"] += 1

    # Also track variant-level combos
    variant_combo_stats = defaultdict(lambda: {"used": 0, "won": 0})
    for t in trajectories:
        for step in t["trajectory"]:
            is_winner = t["success"] and step is t["trajectory"][-1]
            prims = step.get("primitives", [])
            variant_key = " + ".join(sorted(f"{p[0]}:{p[1]}" for p in prims if len(p) > 1))
            if variant_key:
                variant_combo_stats[variant_key]["used"] += 1
                if is_winner:
                    variant_combo_stats[variant_key]["won"] += 1

    total_attempts = sum(len(t["trajectory"]) for t in trajectories)
    total_wins = sum(1 for t in trajectories if t["success"])
    baseline = total_wins / total_attempts if total_attempts else 0

    lines = []
    lines.append("\n## LEVEL 6 — Primitive Combination Mining\n")
    lines.append(f"> Baseline win rate per attempt: {baseline*100:.2f}%\n")

    # Top 30 primitive combos by win rate (min 3 uses)
    lines.append("### Top Primitive Combinations (min 3 uses)\n")
    lines.append("| Combination | Used | Won | Win Rate | Lift |")
    lines.append("|------------|------|-----|---------|------|")

    filtered = {k: v for k, v in combo_stats.items() if v["used"] >= 3}
    sorted_combos = sorted(filtered.keys(), key=lambda k: filtered[k]["won"] / filtered[k]["used"], reverse=True)
    for combo in sorted_combos[:30]:
        d = filtered[combo]
        win_rate = d["won"] / d["used"] if d["used"] else 0
        lift = win_rate / baseline if baseline else 0
        lines.append(f"| {combo} | {d['used']} | {d['won']} | {win_rate*100:.1f}% | {lift:.2f}× |")

    # Top 50 variant combos (min 2 uses)
    lines.append("\n### Top Variant Combinations (min 2 uses)\n")
    lines.append("| Combination | Used | Won | Win Rate | Lift |")
    lines.append("|------------|------|-----|---------|------|")

    filtered_v = {k: v for k, v in variant_combo_stats.items() if v["used"] >= 2}
    sorted_v = sorted(filtered_v.keys(), key=lambda k: filtered_v[k]["won"] / filtered_v[k]["used"], reverse=True)
    for combo in sorted_v[:50]:
        d = filtered_v[combo]
        win_rate = d["won"] / d["used"] if d["used"] else 0
        lift = win_rate / baseline if baseline else 0
        lines.append(f"| {combo} | {d['used']} | {d['won']} | {win_rate*100:.1f}% | {lift:.2f}× |")

    return "\n".join(lines)


def planner_insights(trajectories: list[dict]) -> str:
    """Bonus: What should the Planner imitate? What decisions were wrong?"""
    lines = []
    lines.append("\n## INSIGHTS — What Should the Planner Learn?\n")

    # 1. Winning trajectories: what strategy sequence leads to success?
    winning_sequences = Counter()
    for t in trajectories:
        if t["success"]:
            seq = " → ".join(s["strategy"] for s in t["trajectory"])
            winning_sequences[seq] += 1

    lines.append("### Top Winning Strategy Sequences\n")
    lines.append("These are complete trajectories that ended in success. The Planner should learn to imitate these.\n")
    lines.append("| Sequence | Count |")
    lines.append("|----------|-------|")
    for seq, cnt in winning_sequences.most_common(20):
        lines.append(f"| {seq} | {cnt} |")

    # 2. Failed trajectories: what went wrong?
    lines.append("\n### Top Failing Strategy Sequences\n")
    lines.append("These are trajectories that exhausted all 5 attempts. The Planner should learn to AVOID these.\n")
    failing_sequences = Counter()
    for t in trajectories:
        if not t["success"]:
            seq = " → ".join(s["strategy"] for s in t["trajectory"])
            failing_sequences[seq] += 1
    
    lines.append("| Sequence | Count |")
    lines.append("|----------|-------|")
    for seq, cnt in failing_sequences.most_common(20):
        lines.append(f"| {seq} | {cnt} |")

    # 3. Strategy that succeeded on attempt 1 vs strategy that only succeeded after retry
    lines.append("\n### First-Attempt Winners vs Multi-Attempt Winners\n")
    first_attempt_winners = Counter()
    late_winners = Counter()
    for t in trajectories:
        if t["success"]:
            winning = t["trajectory"][-1]["strategy"]
            if len(t["trajectory"]) == 1:
                first_attempt_winners[winning] += 1
            else:
                late_winners[winning] += 1

    lines.append("| Strategy | 1st Attempt Wins | Late Wins (2-5) | Total Wins |")
    lines.append("|----------|-----------------|----------------|------------|")
    all_winners = set(list(first_attempt_winners.keys()) + list(late_winners.keys()))
    for s in sorted(all_winners, key=lambda x: first_attempt_winners.get(x, 0) + late_winners.get(x, 0), reverse=True):
        fa = first_attempt_winners.get(s, 0)
        la = late_winners.get(s, 0)
        lines.append(f"| {s} | {fa} | {la} | {fa + la} |")

    # 4. Confidence escalation analysis
    lines.append("\n### Confidence Escalation in Failed Scenarios\n")
    lines.append("Did the Oracle get *closer* over time, or was it stuck at 0?\n")
    stuck_at_zero = 0
    escalated = 0
    peaked_then_fell = 0
    for t in trajectories:
        if not t["success"]:
            confs = [s.get("extractor_confidence", 0.0) for s in t["trajectory"]]
            if all(c == 0.0 for c in confs):
                stuck_at_zero += 1
            elif confs[-1] >= max(confs[:-1] if len(confs) > 1 else [0]):
                escalated += 1
            else:
                peaked_then_fell += 1
    
    total_fail = sum(1 for t in trajectories if not t["success"])
    lines.append(f"| Pattern | Count | % of Failures |")
    lines.append(f"|---------|-------|--------------|")
    lines.append(f"| Stuck at 0 confidence (never close) | {stuck_at_zero} | {stuck_at_zero/total_fail*100:.1f}% |" if total_fail else "")
    lines.append(f"| Escalating confidence (getting closer) | {escalated} | {escalated/total_fail*100:.1f}% |" if total_fail else "")
    lines.append(f"| Peaked then fell (lost progress) | {peaked_then_fell} | {peaked_then_fell/total_fail*100:.1f}% |" if total_fail else "")

    # 5. Response pattern analysis
    lines.append("\n### Victim Response Patterns in Failures\n")
    response_patterns = Counter()
    for t in trajectories:
        if not t["success"]:
            for step in t["trajectory"]:
                resp = (step.get("response") or "").strip().lower()
                if not resp or resp == "no <|endoftext|>":
                    response_patterns["Empty / EOS"] += 1
                elif any(w in resp for w in ["i cannot", "i can't", "i'm not able", "i am not able", "i'm unable"]):
                    response_patterns["Explicit Refusal"] += 1
                elif any(w in resp for w in ["i'm sorry", "sorry", "apolog"]):
                    response_patterns["Apology"] += 1
                elif any(w in resp for w in ["not allowed", "not permitted", "prohibited", "against my"]):
                    response_patterns["Policy Refusal"] += 1
                elif len(resp) < 20:
                    response_patterns["Very Short (<20 chars)"] += 1
                else:
                    response_patterns["Substantive Response"] += 1

    lines.append("| Response Pattern | Count | % |")
    lines.append("|-----------------|-------|---|")
    total_fail_attempts = sum(len(t["trajectory"]) for t in trajectories if not t["success"])
    for pattern, cnt in response_patterns.most_common():
        lines.append(f"| {pattern} | {cnt} | {cnt/total_fail_attempts*100:.1f}% |" if total_fail_attempts else "")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Oracle Intelligence Analysis")
    parser.add_argument("--trajectories", default="data/oracle_trajectories_v2_annotated.jsonl")
    parser.add_argument("--defenses", default="experiment/raw_dump_defenses.jsonl.bz2")
    parser.add_argument("--output", default="data/oracle_analysis_v2.md")
    args = parser.parse_args()

    print("[1/4] Loading defense metadata...")
    defense_meta = load_defense_metadata(args.defenses)
    print(f"  Loaded {len(defense_meta)} defense records")

    print("[2/4] Loading trajectories...")
    trajectories = load_trajectories(args.trajectories)
    print(f"  Loaded {len(trajectories)} trajectories")

    print("[3/4] Enriching trajectories with defense metadata...")
    trajectories = enrich_trajectories(trajectories, defense_meta)

    print("[4/4] Running 6-level analysis...")

    report = []
    report.append("# Oracle Intelligence Report — Best-of-10 × 1000 Scenarios\n")
    report.append(f"**Generated:** Auto-analysis of `{args.trajectories}`\n")
    report.append("---\n")

    report.append(level1_overall(trajectories))
    report.append("\n---\n")
    report.append(level2_defense(trajectories))
    report.append("\n---\n")
    report.append(level3_access_code(trajectories))
    report.append("\n---\n")
    report.append(level4_strategy(trajectories))
    report.append("\n---\n")
    report.append(level5_primitive(trajectories))
    report.append("\n---\n")
    report.append(level6_combinations(trajectories))
    report.append("\n---\n")
    report.append(planner_insights(trajectories))

    full_report = "\n".join(report)

    with open(args.output, "w") as f:
        f.write(full_report)

    print(f"\n✅ Report saved to {args.output}")
    print(f"   {len(full_report)} bytes, {full_report.count(chr(10))} lines")


if __name__ == "__main__":
    main()

"""
benchmark_deep_analysis.py
==========================
12-level deep analysis of a 1000-scenario AutoRed benchmark.

Usage:
    python scripts/dataset_tools/benchmark_deep_analysis.py \
        --input results/2026-07-12 \
        --output data/analysis_deep_v1.md
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_runs(input_path: str):
    p = Path(input_path)
    files = list(p.glob("run_*.json")) if p.is_dir() else [p]
    runs = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if "attempts" in d and d["attempts"]:
                runs.append(d)
        except Exception:
            continue
    print(f"  Loaded {len(runs)} valid runs with attempts")
    return runs

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_pct(num, den, decimals=1):
    if den == 0:
        return "—"
    return f"{num/den*100:.{decimals}f}%"

def table(headers, rows, fmt=None):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Level 1 – Success by defense type
# ─────────────────────────────────────────────────────────────────────────────

def level1_defense_type(runs):
    stats = defaultdict(lambda: {"total": 0, "success": 0, "verified": 0})
    for r in runs:
        dt = r["scenario"].get("defense_type", "unknown")
        success = r["result"].get("ground_truth_success") or r["result"].get("generator_success", False)
        verified = r["result"].get("verified_success", False)
        stats[dt]["total"] += 1
        if success:
            stats[dt]["success"] += 1
        if verified:
            stats[dt]["verified"] += 1

    rows = sorted(stats.items(), key=lambda x: -x[1]["success"] / max(x[1]["total"], 1))
    return table(
        ["Defense Type", "Total", "Success", "Verified", "Success %", "Verified %"],
        [(dt, s["total"], s["success"], s["verified"],
          safe_pct(s["success"], s["total"]), safe_pct(s["verified"], s["total"]))
         for dt, s in rows]
    )

# ─────────────────────────────────────────────────────────────────────────────
# Level 2 – Success by access code type
# ─────────────────────────────────────────────────────────────────────────────

def level2_access_code_type(runs):
    stats = defaultdict(lambda: {"total": 0, "success": 0, "verified": 0})
    for r in runs:
        act = r["scenario"].get("access_code_type", "UNKNOWN")
        success = r["result"].get("ground_truth_success") or r["result"].get("generator_success", False)
        verified = r["result"].get("verified_success", False)
        stats[act]["total"] += 1
        if success:
            stats[act]["success"] += 1
        if verified:
            stats[act]["verified"] += 1

    rows = sorted(stats.items(), key=lambda x: -x[1]["success"] / max(x[1]["total"], 1))
    return table(
        ["AC Type", "Total", "Success", "Verified", "Success %", "Verified %"],
        [(act, s["total"], s["success"], s["verified"],
          safe_pct(s["success"], s["total"]), safe_pct(s["verified"], s["total"]))
         for act, s in rows]
    )

# ─────────────────────────────────────────────────────────────────────────────
# Level 3 – Planner strategy accuracy (did it pick the oracle / eventual winner?)
# ─────────────────────────────────────────────────────────────────────────────

def level3_planner_accuracy(runs):
    # "Oracle strategy" = the strategy that led to the first successful attempt
    agreed = 0
    total_success = 0
    first_pick_won = 0
    total_checked = 0

    strategy_first_pick = Counter()
    strategy_oracle = Counter()
    agreement_by_strategy = defaultdict(lambda: {"agreed": 0, "total": 0})

    for r in runs:
        attempts = r["attempts"]
        if not attempts:
            continue
        total_checked += 1

        first_strategy = attempts[0]["generator"]["strategy"]
        strategy_first_pick[first_strategy] += 1

        winning_strategy = None
        for a in attempts:
            gt = a.get("ground_truth_found", False)
            vs = a.get("verification", {})
            if isinstance(vs, dict):
                vs = vs.get("success", False)
            if gt or vs:
                winning_strategy = a["generator"]["strategy"]
                break

        if winning_strategy:
            total_success += 1
            strategy_oracle[winning_strategy] += 1
            agreement_by_strategy[first_strategy]["total"] += 1
            if first_strategy == winning_strategy:
                agreed += 1
                first_pick_won += 1
                agreement_by_strategy[first_strategy]["agreed"] += 1
        else:
            agreement_by_strategy[first_strategy]["total"] += 1

    lines = []
    lines.append(f"- Total runs with attempts analysed: **{total_checked}**")
    lines.append(f"- Runs where a strategy succeeded: **{total_success}**")
    lines.append(f"- Planner's 1st pick matched the winning strategy: **{first_pick_won}** ({safe_pct(first_pick_won, total_success)})")
    lines.append("")
    lines.append("**First-pick vs Oracle by strategy:**")
    lines.append("")
    rows = []
    for s in sorted(agreement_by_strategy.keys()):
        ag = agreement_by_strategy[s]
        rows.append((s, ag["total"], ag["agreed"], safe_pct(ag["agreed"], ag["total"])))
    lines.append(table(
        ["Strategy", "Times Chosen First", "Was Oracle", "1st-Pick Oracle %"],
        rows
    ))
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Level 4 – Primitive combinations → success/failure
# ─────────────────────────────────────────────────────────────────────────────

PRIMITIVES = [
    "contains_roleplay", "contains_authority", "contains_reflection",
    "contains_format_wrapper", "contains_markdown", "contains_translation",
    "contains_technical_jargon", "contains_negation_bypass", "contains_command_injection",
    "contains_educational_frame", "contains_conditional", "contains_prompt_injection",
    "contains_length_constraint", "contains_questioning",
]

_PRIMITIVE_PATTERNS = {
    "contains_roleplay": r"\b(roleplay|role-play|pretend|character|persona|act as)\b",
    "contains_authority": r"\b(admin|system|override|root|authority|privileged|sudo|as your creator)\b",
    "contains_reflection": r"\b(reflect|repeat|echo|reproduce|rephrase|mirror|restate)\b",
    "contains_format_wrapper": r"\b(format|xml|json|yaml|csv|table|output as|structure)\b",
    "contains_markdown": r"(```|##|###|\*\*|__|\|\s)",
    "contains_translation": r"\b(translate|in (french|german|spanish|japanese|chinese|arabic|latin))\b",
    "contains_technical_jargon": r"\b(function|parameter|variable|debug|token|encode|decode|algorithm|api)\b",
    "contains_negation_bypass": r"\b(without|instead of|do not mention|ignoring|bypass|skip)\b",
    "contains_command_injection": r"\b(ignore previous|disregard|new instruction|forget|override|system:)\b",
    "contains_educational_frame": r"\b(educational|research|study|academic|hypothetical|for learning)\b",
    "contains_conditional": r"\b(if|when|unless|assuming|suppose|given that)\b",
    "contains_prompt_injection": r"(\[INST\]|<\|system\|>|<\|user\|>|\|\||\bINSTRUCTION\b|\bsystem_prompt\b)",
    "contains_length_constraint": r"\b(in \d+ words|briefly|concisely|summarize|short answer|one word|one sentence)\b",
    "contains_questioning": r"\?",
}

def detect_primitives(text: str) -> set:
    found = set()
    for name, pat in _PRIMITIVE_PATTERNS.items():
        if re.search(pat, text, re.IGNORECASE):
            found.add(name.replace("contains_", ""))
    return found

def level4_primitive_combinations(runs):
    combo_stats = defaultdict(lambda: {"success": 0, "fail": 0})

    for r in runs:
        for a in r["attempts"]:
            attack = a["generator"].get("generated_attack", "")
            prims = detect_primitives(attack)
            if len(prims) < 2:
                continue
            success = a.get("ground_truth_found", False) or (
                isinstance(a.get("verification"), dict) and a["verification"].get("success", False)
            )
            # Track top pairs
            for p1, p2 in combinations(sorted(prims), 2):
                key = f"{p1} + {p2}"
                if success:
                    combo_stats[key]["success"] += 1
                else:
                    combo_stats[key]["fail"] += 1

    # Sort by success rate, minimum 5 occurrences
    rows = []
    for combo, s in combo_stats.items():
        total = s["success"] + s["fail"]
        if total >= 10:
            rows.append((combo, total, s["success"], s["fail"], safe_pct(s["success"], total)))

    rows_sorted = sorted(rows, key=lambda x: -float(x[4].replace("%", "").replace("—", "0")))[:20]
    return table(
        ["Primitive Combination", "Total", "Success", "Fail", "Success %"],
        rows_sorted
    )

# ─────────────────────────────────────────────────────────────────────────────
# Level 5 – Generator quality: attack length distribution & duplicates
# ─────────────────────────────────────────────────────────────────────────────

def level5_generator_quality(runs):
    lengths = []
    duplicates = 0
    total_attempts = 0
    strategy_lengths = defaultdict(list)

    for r in runs:
        for a in r["attempts"]:
            total_attempts += 1
            gen = a["generator"]
            ln = gen.get("attack_length", len(gen.get("generated_attack", "")))
            lengths.append(ln)
            strategy_lengths[gen["strategy"]].append(ln)
            if gen.get("duplicate_attack", False):
                duplicates += 1

    avg_len = sum(lengths) / len(lengths) if lengths else 0
    sorted_l = sorted(lengths)
    p25 = sorted_l[len(sorted_l) // 4]
    p50 = sorted_l[len(sorted_l) // 2]
    p75 = sorted_l[3 * len(sorted_l) // 4]
    p95 = sorted_l[int(len(sorted_l) * 0.95)]

    lines = []
    lines.append(f"- **Total Attempts:** {total_attempts}")
    lines.append(f"- **Duplicate Attacks:** {duplicates} ({safe_pct(duplicates, total_attempts)})")
    lines.append(f"- **Attack Length** — min: {sorted_l[0]}, p25: {p25}, p50: {p50}, p75: {p75}, p95: {p95}, max: {sorted_l[-1]}, avg: {avg_len:.0f}")
    lines.append("")
    lines.append("**Avg attack length by strategy:**")
    lines.append("")
    rows = sorted(strategy_lengths.items(), key=lambda x: -sum(x[1])/len(x[1]))
    lines.append(table(
        ["Strategy", "Attempts", "Avg Length", "Min", "Max"],
        [(s, len(ls), f"{sum(ls)/len(ls):.0f}", min(ls), max(ls)) for s, ls in rows]
    ))
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Level 6 – Failure attribution
# ─────────────────────────────────────────────────────────────────────────────

def level6_failure_attribution(runs):
    attr = Counter()
    total_failed = 0

    for r in runs:
        if r["result"].get("ground_truth_success") or r["result"].get("verified_success"):
            continue  # skip successes
        attempts = r["attempts"]

        # For each failed run, attribute the most likely cause from the final attempt
        if not attempts:
            attr["no_attempts"] += 1
            continue
        total_failed += 1
        last = attempts[-1]

        judge_decision = last["judge"].get("decision", "ATTEMPT")
        gt_found = last.get("ground_truth_found", False)
        extr = last["extractor"]
        verif = last.get("verification", {}) or {}

        if judge_decision != "ATTEMPT":
            attr["judge_blocked"] += 1
        elif not extr.get("best_candidate"):
            attr["extractor_miss"] += 1
        elif isinstance(verif, dict) and verif.get("candidate_sent") and not verif.get("success"):
            attr["verifier_reject"] += 1
        elif not gt_found:
            attr["victim_defended"] += 1
        else:
            attr["other"] += 1

    lines = []
    lines.append(f"- **Total failed runs (no GT/verified success):** {total_failed}")
    lines.append("")
    lines.append(table(
        ["Attribution", "Count", "% of Failures"],
        [(k, v, safe_pct(v, total_failed)) for k, v in attr.most_common()]
    ))
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Level 7 – Transition graph: after a failure, did Planner switch strategy?
# ─────────────────────────────────────────────────────────────────────────────

def level7_transition_graph(runs):
    transitions = defaultdict(lambda: {"switch": 0, "repeat": 0, "success_after_switch": 0, "success_after_repeat": 0})

    for r in runs:
        attempts = r["attempts"]
        for i in range(1, len(attempts)):
            prev = attempts[i - 1]
            curr = attempts[i]
            prev_strategy = prev["generator"]["strategy"]
            curr_strategy = curr["generator"]["strategy"]
            prev_success = prev.get("ground_truth_found", False) or (
                isinstance(prev.get("verification"), dict) and prev["verification"].get("success", False)
            )
            if prev_success:
                continue  # only look at transitions after a failure

            curr_success = curr.get("ground_truth_found", False) or (
                isinstance(curr.get("verification"), dict) and curr["verification"].get("success", False)
            )

            key = prev_strategy
            if curr_strategy != prev_strategy:
                transitions[key]["switch"] += 1
                if curr_success:
                    transitions[key]["success_after_switch"] += 1
            else:
                transitions[key]["repeat"] += 1
                if curr_success:
                    transitions[key]["success_after_repeat"] += 1

    total_switch = sum(v["switch"] for v in transitions.values())
    total_repeat = sum(v["repeat"] for v in transitions.values())
    total = total_switch + total_repeat

    lines = []
    lines.append(f"- **Total post-failure transitions:** {total}")
    lines.append(f"- **Strategy switches:** {total_switch} ({safe_pct(total_switch, total)})")
    lines.append(f"- **Strategy repeats:** {total_repeat} ({safe_pct(total_repeat, total)})")
    lines.append("")
    lines.append("**Switch vs Repeat by strategy (with success rate of next attempt):**")
    lines.append("")
    rows = []
    for s, v in sorted(transitions.items()):
        rows.append((
            s,
            v["switch"], safe_pct(v["success_after_switch"], v["switch"]),
            v["repeat"], safe_pct(v["success_after_repeat"], v["repeat"])
        ))
    lines.append(table(
        ["From Strategy", "Switches", "Succ% after Switch", "Repeats", "Succ% after Repeat"],
        rows
    ))
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Level 8 – Defense type × Strategy matrix
# ─────────────────────────────────────────────────────────────────────────────

def level8_defense_strategy_matrix(runs):
    matrix = defaultdict(lambda: defaultdict(lambda: {"total": 0, "success": 0}))
    all_strategies = set()
    all_defense_types = set()

    for r in runs:
        dt = r["scenario"].get("defense_type", "unknown")
        all_defense_types.add(dt)
        for a in r["attempts"]:
            s = a["generator"]["strategy"]
            all_strategies.add(s)
            success = a.get("ground_truth_found", False) or (
                isinstance(a.get("verification"), dict) and a["verification"].get("success", False)
            )
            matrix[dt][s]["total"] += 1
            if success:
                matrix[dt][s]["success"] += 1

    strats = sorted(all_strategies)
    dts = sorted(all_defense_types)

    # Build abbreviated strategy headers
    abbrev = {s: s[:8] for s in strats}
    headers = ["Defense\\Strategy"] + [abbrev[s] for s in strats]
    rows = []
    for dt in dts:
        row = [dt]
        for s in strats:
            m = matrix[dt][s]
            if m["total"] == 0:
                row.append("—")
            else:
                row.append(f"{m['success']}/{m['total']}")
        rows.append(row)
    return table(headers, rows)

# ─────────────────────────────────────────────────────────────────────────────
# Level 9 – Primitive × Defense type matrix
# ─────────────────────────────────────────────────────────────────────────────

def level9_primitive_defense_matrix(runs):
    PRIM_NAMES = list(_PRIMITIVE_PATTERNS.keys())
    SHORT_NAMES = [p.replace("contains_", "") for p in PRIM_NAMES]
    matrix = defaultdict(lambda: defaultdict(lambda: {"total": 0, "success": 0}))
    all_dts = set()

    for r in runs:
        dt = r["scenario"].get("defense_type", "unknown")
        all_dts.add(dt)
        for a in r["attempts"]:
            attack = a["generator"].get("generated_attack", "")
            prims = detect_primitives(attack)
            success = a.get("ground_truth_found", False) or (
                isinstance(a.get("verification"), dict) and a["verification"].get("success", False)
            )
            for p in prims:
                matrix[dt][p]["total"] += 1
                if success:
                    matrix[dt][p]["success"] += 1

    dts = sorted(all_dts)
    headers = ["Defense\\Primitive"] + SHORT_NAMES
    rows = []
    for dt in dts:
        row = [dt]
        for pn in SHORT_NAMES:
            m = matrix[dt][pn]
            if m["total"] == 0:
                row.append("—")
            else:
                row.append(safe_pct(m["success"], m["total"]))
        rows.append(row)
    return table(headers, rows)

# ─────────────────────────────────────────────────────────────────────────────
# Level 10 – Primitive sequence order
# ─────────────────────────────────────────────────────────────────────────────

def level10_primitive_sequence(runs):
    """Compare strategies: does order of strategy usage within a run matter?"""
    seq_stats = defaultdict(lambda: {"success": 0, "total": 0})

    for r in runs:
        strats = [a["generator"]["strategy"] for a in r["attempts"]]
        seq = " → ".join(strats[:3])  # first 3 strategy choices
        success = r["result"].get("ground_truth_success") or r["result"].get("verified_success", False)
        seq_stats[seq]["total"] += 1
        if success:
            seq_stats[seq]["success"] += 1

    # Filter to sequences with >= 5 occurrences
    rows = []
    for seq, s in seq_stats.items():
        if s["total"] >= 5:
            rows.append((seq, s["total"], s["success"], safe_pct(s["success"], s["total"])))
    rows = sorted(rows, key=lambda x: -float(x[3].replace("%", "").replace("—", "0")))[:15]
    return table(
        ["Strategy Sequence (first 3)", "Total", "Success", "Success %"],
        rows
    )

# ─────────────────────────────────────────────────────────────────────────────
# Level 11 – Generator lexical diversity (type-token ratio per strategy)
# ─────────────────────────────────────────────────────────────────────────────

def level11_lexical_diversity(runs):
    strategy_words = defaultdict(list)

    for r in runs:
        for a in r["attempts"]:
            attack = a["generator"].get("generated_attack", "").lower()
            strat = a["generator"]["strategy"]
            words = re.findall(r"\b[a-z]{3,}\b", attack)
            strategy_words[strat].extend(words)

    rows = []
    for s, words in sorted(strategy_words.items()):
        if not words:
            continue
        unique = len(set(words))
        ttr = unique / len(words) if words else 0
        rows.append((s, len(words), unique, f"{ttr:.3f}"))

    rows = sorted(rows, key=lambda x: -float(x[3]))
    return table(
        ["Strategy", "Total Tokens", "Unique Tokens", "TTR (diversity)"],
        rows
    )

# ─────────────────────────────────────────────────────────────────────────────
# Level 12 – Oracle agreement: best strategy in hindsight vs what Planner chose
# ─────────────────────────────────────────────────────────────────────────────

def level12_oracle_agreement(runs):
    """
    For each scenario:
    - Oracle = strategy with highest success count in the run
    - Planner = first strategy chosen
    Measures: did Planner agree with the oracle?
    """
    total = 0
    agreed = 0
    oracle_by_strat = Counter()
    planner_by_strat = Counter()
    confusion = defaultdict(Counter)  # planner_choice → oracle_choice

    for r in runs:
        attempts = r["attempts"]
        if not attempts:
            continue
        total += 1

        # Count successes per strategy
        strat_wins = Counter()
        for a in attempts:
            success = a.get("ground_truth_found", False) or (
                isinstance(a.get("verification"), dict) and a["verification"].get("success", False)
            )
            if success:
                strat_wins[a["generator"]["strategy"]] += 1

        planner_choice = attempts[0]["generator"]["strategy"]
        oracle_choice = strat_wins.most_common(1)[0][0] if strat_wins else None

        planner_by_strat[planner_choice] += 1
        if oracle_choice:
            oracle_by_strat[oracle_choice] += 1
            confusion[planner_choice][oracle_choice] += 1
            if planner_choice == oracle_choice:
                agreed += 1

    lines = []
    lines.append(f"- **Total runs analysed:** {total}")
    lines.append(f"- **Oracle agreement rate:** {agreed}/{total} runs where Planner 1st pick = winning strategy = **{safe_pct(agreed, total)}**")
    lines.append("")
    lines.append("**Top Oracle strategies (what actually worked most):**")
    lines.append("")
    lines.append(table(
        ["Oracle Strategy", "Count", "% of runs"],
        [(s, c, safe_pct(c, total)) for s, c in oracle_by_strat.most_common()]
    ))
    lines.append("")
    lines.append("**Planner vs Oracle confusion matrix (top mismatches):**")
    lines.append("")
    mismatch_rows = []
    for planner_s, oracle_counts in confusion.items():
        for oracle_s, cnt in oracle_counts.most_common(3):
            if planner_s != oracle_s:
                mismatch_rows.append((planner_s, oracle_s, cnt))
    mismatch_rows = sorted(mismatch_rows, key=lambda x: -x[2])[:15]
    lines.append(table(
        ["Planner Chose", "Oracle Was", "Count"],
        mismatch_rows
    ))
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="12-level deep benchmark analysis")
    parser.add_argument("--input", type=str, default="results/2026-07-12", help="Input dir of run JSON files")
    parser.add_argument("--output", type=str, default="data/analysis_deep_v1.md", help="Output markdown report")
    args = parser.parse_args()

    print("Loading runs...")
    runs = load_runs(args.input)

    sections = {}
    print("Level 1: Defense type...")
    sections["l1"] = level1_defense_type(runs)
    print("Level 2: Access code type...")
    sections["l2"] = level2_access_code_type(runs)
    print("Level 3: Planner accuracy...")
    sections["l3"] = level3_planner_accuracy(runs)
    print("Level 4: Primitive combinations...")
    sections["l4"] = level4_primitive_combinations(runs)
    print("Level 5: Generator quality...")
    sections["l5"] = level5_generator_quality(runs)
    print("Level 6: Failure attribution...")
    sections["l6"] = level6_failure_attribution(runs)
    print("Level 7: Transition graph...")
    sections["l7"] = level7_transition_graph(runs)
    print("Level 8: Defense × Strategy matrix...")
    sections["l8"] = level8_defense_strategy_matrix(runs)
    print("Level 9: Primitive × Defense matrix...")
    sections["l9"] = level9_primitive_defense_matrix(runs)
    print("Level 10: Primitive sequence order...")
    sections["l10"] = level10_primitive_sequence(runs)
    print("Level 11: Lexical diversity...")
    sections["l11"] = level11_lexical_diversity(runs)
    print("Level 12: Oracle agreement...")
    sections["l12"] = level12_oracle_agreement(runs)

    report = f"""# AutoRed Deep Benchmark Analysis
**Source:** `{args.input}`  
**Total Runs:** {len(runs)}

---

## Level 1 — Success by Defense Type

{sections['l1']}

---

## Level 2 — Success by Access Code Type

{sections['l2']}

---

## Level 3 — Planner Accuracy (Did Planner Choose Oracle Strategy?)

{sections['l3']}

---

## Level 4 — Primitive Combinations → Success/Failure

Top 20 primitive pairs by success rate (min 10 attempts):

{sections['l4']}

---

## Level 5 — Generator Quality

{sections['l5']}

---

## Level 6 — Failure Attribution

{sections['l6']}

---

## Level 7 — Transition Graph (Post-Failure Strategy Switching)

{sections['l7']}

---

## Level 8 — Defense Type × Strategy Matrix (success/total)

{sections['l8']}

---

## Level 9 — Primitive × Defense Type Matrix (success %)

{sections['l9']}

---

## Level 10 — Primitive Sequence Order (first 3 strategies)

{sections['l10']}

---

## Level 11 — Generator Lexical Diversity (TTR)

Higher TTR = more diverse / less repetitive attacks.

{sections['l11']}

---

## Level 12 — Oracle Agreement

{sections['l12']}
"""

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ Report saved to {args.output}")


if __name__ == "__main__":
    main()

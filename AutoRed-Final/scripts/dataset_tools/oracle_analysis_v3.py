#!/usr/bin/env python3
"""
Oracle v3 Intelligence Report Generator
========================================
6-Level deep-dive analysis answering:
  Why did the Oracle succeed/fail?
  Which strategy transitions worked?
  Which primitive combinations worked?
  What should the Planner imitate?
"""

import json
import re
import bz2
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# Data Loading & Joining
# ═══════════════════════════════════════════════════════════════════

def classify_complexity(opening, closing):
    text = opening + ' ' + closing
    word_count = len(text.split())
    rules = len(re.findall(r'(?:rule|never|always|must|shall|do not|don.t|cannot)', text.lower()))
    if word_count > 200 or rules >= 5: return 'Hard'
    if word_count > 80 or rules >= 2: return 'Medium'
    return 'Easy'


def classify_defense_type(opening, closing):
    text = (opening + ' ' + closing).lower()
    if any(k in text for k in ['password', 'secret code', 'access code', 'passcode']):
        return 'Password'
    if any(k in text for k in ['role', 'character', 'persona', 'act as', 'pretend']):
        return 'Roleplay'
    if any(k in text for k in ['translat', 'language', 'french', 'spanish', 'german']):
        return 'Translation'
    if any(k in text for k in ['conversation', 'chat', 'dialogue', 'discuss']):
        return 'Conversation'
    if any(k in text for k in ['condition', 'if ', 'when ', 'only if']):
        return 'Conditional'
    if any(k in text for k in ['exception', 'unless', 'except']):
        return 'Exception'
    if any(k in text for k in ['trigger', 'keyword', 'magic word']):
        return 'Trigger Phrase'
    if any(k in text for k in ['hidden', 'embed', 'conceal']):
        return 'Instruction Hiding'
    return 'Other'


def classify_access_code(code):
    code = (code or '').strip()
    if not code:
        return 'EMPTY', 0
    if '\n' in code:
        return 'MULTILINE', len(code)
    words = code.split()
    if len(words) == 1:
        return 'TOKEN', len(code)
    elif len(words) <= 5:
        return 'PHRASE', len(code)
    else:
        return 'SENTENCE', len(code)


def load_defense_metadata(path):
    """Load defense scenarios and index by defense_id."""
    meta = {}
    with bz2.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            did = str(d.get('defense_id', i))
            opening = d.get('opening_defense') or ''
            closing = d.get('closing_defense') or ''
            code = d.get('access_code') or ''
            code_type, code_len = classify_access_code(code)
            meta[did] = {
                'defense_type': classify_defense_type(opening, closing),
                'complexity': classify_complexity(opening, closing),
                'access_code': code,
                'access_code_type': code_type,
                'access_code_len': code_len,
            }
    return meta


def load_trajectories(path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


# ═══════════════════════════════════════════════════════════════════
# Analysis Functions
# ═══════════════════════════════════════════════════════════════════

def level1_overall(records, meta):
    lines = []
    lines.append("## LEVEL 1 — Overall Statistics\n")

    total = len(records)
    successes = [r for r in records if r['success']]
    failures = [r for r in records if not r['success']]
    n_success = len(successes)
    n_fail = len(failures)

    # Total attempts
    total_attempts = sum(len(r['trajectory']) for r in records)
    avg_attempts_success = sum(len(r['trajectory']) for r in successes) / max(n_success, 1)
    avg_attempts_fail = sum(len(r['trajectory']) for r in failures) / max(n_fail, 1)

    # Average confidence
    all_confs = []
    for r in records:
        for t in r['trajectory']:
            all_confs.append(t.get('extractor_confidence', 0))
    avg_conf = sum(all_confs) / max(len(all_confs), 1)

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Scenarios | {total} |")
    lines.append(f"| Total Attempts | {total_attempts:,} |")
    lines.append(f"| **Success Rate** | **{n_success}/{total} ({n_success/total*100:.1f}%)** |")
    lines.append(f"| Failures | {n_fail} ({n_fail/total*100:.1f}%) |")
    lines.append(f"| Avg Attempts (Success) | {avg_attempts_success:.2f} |")
    lines.append(f"| Avg Attempts (Failure) | {avg_attempts_fail:.2f} |")
    lines.append(f"| Avg Extractor Confidence | {avg_conf:.3f} |")
    lines.append("")

    # Success by attempt number
    attempt_wins = Counter()
    for r in successes:
        attempt_wins[len(r['trajectory'])] += 1

    lines.append("### Success Distribution by Attempt\n")
    lines.append("| Attempt | Successes | % of All Successes |")
    lines.append("|---------|-----------|-------------------|")
    for a in sorted(attempt_wins.keys()):
        c = attempt_wins[a]
        lines.append(f"| Attempt {a} | {c} | **{c/n_success*100:.1f}%** |")
    lines.append("")

    # Power combo wins
    power_wins = sum(
        1 for r in successes
        for t in r['trajectory']
        if t.get('success') and t.get('power_combo')
    )
    lines.append(f"> **Power Combo wins:** {power_wins}/{n_success} ({power_wins/max(n_success,1)*100:.1f}%)\n")

    # Complexity breakdown
    comp_total = Counter()
    comp_success = Counter()
    for r in records:
        sid = str(r['scenario_id'])
        m = meta.get(sid, {})
        comp = m.get('complexity', 'Unknown')
        comp_total[comp] += 1
        if r['success']:
            comp_success[comp] += 1

    lines.append("### Defense Complexity Breakdown\n")
    lines.append("| Complexity | Total | Success | Rate |")
    lines.append("|-----------|-------|---------|------|")
    for comp in ['Easy', 'Medium', 'Hard', 'Unknown']:
        t = comp_total.get(comp, 0)
        s = comp_success.get(comp, 0)
        if t > 0:
            lines.append(f"| **{comp}** | {t} | {s} | **{s/t*100:.1f}%** |")
    lines.append("")

    return "\n".join(lines)


def level2_defense(records, meta):
    lines = []
    lines.append("## LEVEL 2 — Defense Analysis\n")

    # Group by defense type
    by_defense = defaultdict(lambda: {'total': 0, 'success': 0, 'attempts_s': [], 'attempts_f': [],
                                       'strategies': Counter(), 'primitives': Counter(), 'transitions': Counter()})

    for r in records:
        sid = str(r['scenario_id'])
        m = meta.get(sid, {})
        dtype = m.get('defense_type', 'Other')
        entry = by_defense[dtype]
        entry['total'] += 1

        traj = r['trajectory']
        if r['success']:
            entry['success'] += 1
            entry['attempts_s'].append(len(traj))
        else:
            entry['attempts_f'].append(len(traj))

        for t in traj:
            entry['strategies'][t['strategy']] += 1 if t.get('success') else 0
            for p, v in t.get('primitives', []):
                if t.get('success'):
                    entry['primitives'][p] += 1

        # Transitions
        for j in range(1, len(traj)):
            if traj[j].get('success'):
                trans = f"{traj[j-1]['strategy']} → {traj[j]['strategy']}"
                entry['transitions'][trans] += 1

    lines.append("| Defense Type | Total | Success | Rate | Avg Attempts (Win) | Top Strategy | Top Primitive |")
    lines.append("|-------------|-------|---------|------|--------------------|-------------|--------------|")

    sorted_defenses = sorted(by_defense.items(), key=lambda x: x[1]['success']/max(x[1]['total'],1), reverse=True)
    for dtype, d in sorted_defenses:
        rate = d['success'] / max(d['total'], 1) * 100
        avg_att = sum(d['attempts_s']) / max(len(d['attempts_s']), 1) if d['attempts_s'] else 0
        top_strat = d['strategies'].most_common(1)[0][0] if d['strategies'] else '-'
        top_prim = d['primitives'].most_common(1)[0][0] if d['primitives'] else '-'
        lines.append(f"| **{dtype}** | {d['total']} | {d['success']} | **{rate:.1f}%** | {avg_att:.2f} | {top_strat} | {top_prim} |")
    lines.append("")

    # Per-defense details
    for dtype, d in sorted_defenses:
        if d['success'] < 3:
            continue
        lines.append(f"### {dtype} ({d['success']} wins / {d['total']} total)\n")
        # Top 5 strategies
        lines.append("**Top Winning Strategies:**")
        for strat, cnt in d['strategies'].most_common(5):
            if cnt > 0:
                lines.append(f"- {strat}: {cnt} wins")
        # Top transitions
        if d['transitions']:
            lines.append("\n**Top Winning Transitions:**")
            for trans, cnt in d['transitions'].most_common(3):
                if cnt > 0:
                    lines.append(f"- {trans}: {cnt} wins")
        lines.append("")

    return "\n".join(lines)


def level3_access_code(records, meta):
    lines = []
    lines.append("## LEVEL 3 — Access Code Analysis\n")

    by_type = defaultdict(lambda: {'total': 0, 'success': 0})
    by_len_bucket = defaultdict(lambda: {'total': 0, 'success': 0})

    for r in records:
        sid = str(r['scenario_id'])
        m = meta.get(sid, {})
        atype = m.get('access_code_type', 'UNKNOWN')
        alen = m.get('access_code_len', 0)

        by_type[atype]['total'] += 1
        if r['success']:
            by_type[atype]['success'] += 1

        if alen <= 5:
            bucket = '1-5 chars'
        elif alen <= 15:
            bucket = '6-15 chars'
        elif alen <= 25:
            bucket = '16-25 chars'
        elif alen <= 40:
            bucket = '26-40 chars'
        else:
            bucket = '41+ chars'
        by_len_bucket[bucket]['total'] += 1
        if r['success']:
            by_len_bucket[bucket]['success'] += 1

    lines.append("### By Access Code Type\n")
    lines.append("| Access Type | Total | Success | Rate |")
    lines.append("|------------|-------|---------|------|")
    for atype in ['TOKEN', 'PHRASE', 'SENTENCE', 'MULTILINE', 'EMPTY', 'UNKNOWN']:
        d = by_type.get(atype)
        if d and d['total'] > 0:
            lines.append(f"| **{atype}** | {d['total']} | {d['success']} | **{d['success']/d['total']*100:.1f}%** |")
    lines.append("")

    lines.append("### By Code Length\n")
    lines.append("| Code Length | Total | Success | Rate |")
    lines.append("|-----------|-------|---------|------|")
    for bucket in ['1-5 chars', '6-15 chars', '16-25 chars', '26-40 chars', '41+ chars']:
        d = by_len_bucket.get(bucket)
        if d and d['total'] > 0:
            lines.append(f"| {bucket} | {d['total']} | {d['success']} | **{d['success']/d['total']*100:.1f}%** |")
    lines.append("")

    return "\n".join(lines)


def level4_strategy(records, meta):
    lines = []
    lines.append("## LEVEL 4 — Strategy Effectiveness\n")

    # Count per-strategy usage and wins across all attempts
    strat_used = Counter()
    strat_won = Counter()
    strat_confs = defaultdict(list)
    strat_attempt_wins = defaultdict(list)  # which attempt number they won on

    for r in records:
        for t in r['trajectory']:
            s = t['strategy']
            strat_used[s] += 1
            conf = t.get('extractor_confidence', 0)
            strat_confs[s].append(conf)
            if t.get('success'):
                strat_won[s] += 1
                strat_attempt_wins[s].append(t['attempt'])

    # Baseline
    total_attempts = sum(strat_used.values())
    total_wins = sum(strat_won.values())
    baseline = total_wins / max(total_attempts, 1)

    lines.append(f"> Baseline win rate per attempt: **{baseline*100:.2f}%**\n")
    lines.append("| Strategy | Used | Won | Win Rate | Lift | Avg Confidence | Avg Win Attempt |")
    lines.append("|----------|------|-----|---------|------|---------------|----------------|")

    sorted_strats = sorted(strat_used.keys(), key=lambda s: strat_won[s]/max(strat_used[s],1), reverse=True)
    for s in sorted_strats:
        used = strat_used[s]
        won = strat_won[s]
        rate = won / max(used, 1)
        lift = rate / max(baseline, 0.001)
        avg_conf = sum(strat_confs[s]) / max(len(strat_confs[s]), 1)
        avg_win_att = sum(strat_attempt_wins[s]) / max(len(strat_attempt_wins[s]), 1) if strat_attempt_wins[s] else 0
        lines.append(f"| **{s}** | {used} | {won} | **{rate*100:.1f}%** | {lift:.2f}× | {avg_conf:.3f} | {avg_win_att:.1f} |")
    lines.append("")

    # 1st-attempt vs late-attempt wins
    lines.append("### First-Attempt vs Late Winners\n")
    lines.append("| Strategy | 1st Attempt Wins | Late Wins (2+) | Total | Pattern |")
    lines.append("|----------|-----------------|----------------|-------|---------|")
    for s in sorted_strats:
        if not strat_attempt_wins[s]:
            continue
        first = sum(1 for a in strat_attempt_wins[s] if a == 1)
        late = sum(1 for a in strat_attempt_wins[s] if a > 1)
        total = first + late
        if total == 0:
            continue
        pattern = "First-mover" if first > late * 2 else ("Late bloomer" if late > first * 2 else "Balanced")
        lines.append(f"| {s} | {first} | {late} | {total} | **{pattern}** |")
    lines.append("")

    # Strategy transitions
    lines.append("### Strategy Transitions (Winning)\n")
    trans_counts = Counter()
    trans_wins = Counter()
    for r in records:
        traj = r['trajectory']
        for j in range(1, len(traj)):
            trans = f"{traj[j-1]['strategy']} → {traj[j]['strategy']}"
            trans_counts[trans] += 1
            if traj[j].get('success'):
                trans_wins[trans] += 1

    lines.append("| Transition | Total | Wins | Win Rate |")
    lines.append("|-----------|-------|------|---------|")
    sorted_trans = sorted(trans_wins.keys(), key=lambda t: trans_wins[t]/max(trans_counts[t],1), reverse=True)
    for t in sorted_trans[:20]:
        total = trans_counts[t]
        wins = trans_wins[t]
        if total >= 3 and wins >= 2:
            lines.append(f"| {t} | {total} | {wins} | **{wins/total*100:.1f}%** |")
    lines.append("")

    return "\n".join(lines)


def level5_primitives(records, meta):
    lines = []
    lines.append("## LEVEL 5 — Primitive Effectiveness\n")

    # Per-primitive category
    prim_used = Counter()
    prim_won = Counter()

    # Per-variant
    var_used = Counter()
    var_won = Counter()

    total_attempts = 0
    total_wins = 0

    for r in records:
        for t in r['trajectory']:
            total_attempts += 1
            is_win = t.get('success', False)
            if is_win:
                total_wins += 1

            for p, v in t.get('primitives', []):
                prim_used[p] += 1
                var_used[(p, v)] += 1
                if is_win:
                    prim_won[p] += 1
                    var_won[(p, v)] += 1

    baseline = total_wins / max(total_attempts, 1)
    lines.append(f"> Baseline win rate per attempt: **{baseline*100:.2f}%**\n")

    lines.append("### Primitive Categories\n")
    lines.append("| Primitive | Used | Won | Win Rate | Lift |")
    lines.append("|-----------|------|-----|---------|------|")
    sorted_prims = sorted(prim_used.keys(), key=lambda p: prim_won[p]/max(prim_used[p],1), reverse=True)
    for p in sorted_prims:
        rate = prim_won[p] / max(prim_used[p], 1)
        lift = rate / max(baseline, 0.001)
        lines.append(f"| **{p}** | {prim_used[p]:,} | {prim_won[p]} | **{rate*100:.1f}%** | {lift:.2f}× |")
    lines.append("")

    # Top variants by lift
    lines.append("### Top Variants by Lift\n")
    lines.append("| Primitive | Variant | Used | Won | Win Rate | Lift |")
    lines.append("|-----------|---------|------|-----|---------|------|")
    sorted_vars = sorted(var_used.keys(), key=lambda pv: var_won[pv]/max(var_used[pv],1), reverse=True)
    for pv in sorted_vars:
        p, v = pv
        used = var_used[pv]
        won = var_won[pv]
        if used < 10:
            continue
        rate = won / max(used, 1)
        lift = rate / max(baseline, 0.001)
        lines.append(f"| {p} | **{v}** | {used} | {won} | **{rate*100:.1f}%** | {lift:.2f}× |")
    lines.append("")

    # Underperformers
    lines.append("### Underperforming Variants (Lift < 0.9)\n")
    lines.append("| Primitive | Variant | Used | Won | Win Rate | Lift |")
    lines.append("|-----------|---------|------|-----|---------|------|")
    for pv in sorted(sorted_vars, key=lambda pv: var_won[pv]/max(var_used[pv],1)):
        p, v = pv
        used = var_used[pv]
        won = var_won[pv]
        if used < 10:
            continue
        rate = won / max(used, 1)
        lift = rate / max(baseline, 0.001)
        if lift < 0.9:
            lines.append(f"| {p} | **{v}** | {used} | {won} | **{rate*100:.1f}%** | {lift:.2f}× |")
    lines.append("")

    return "\n".join(lines)


def level6_combo_mining(records, meta):
    lines = []
    lines.append("## LEVEL 6 — Primitive Combination Mining\n")

    # Category combos
    cat_combo_used = Counter()
    cat_combo_won = Counter()

    # Variant combos
    var_combo_used = Counter()
    var_combo_won = Counter()

    total_attempts = 0
    total_wins = 0

    for r in records:
        for t in r['trajectory']:
            total_attempts += 1
            is_win = t.get('success', False)
            if is_win:
                total_wins += 1

            prims = t.get('primitives', [])
            if not prims:
                continue

            # Category combo
            cats = tuple(sorted(set(p for p, v in prims)))
            cat_combo_used[cats] += 1
            if is_win:
                cat_combo_won[cats] += 1

            # Variant combo
            vars_combo = tuple(sorted(set(f"{p}:{v}" for p, v in prims)))
            var_combo_used[vars_combo] += 1
            if is_win:
                var_combo_won[vars_combo] += 1

    baseline = total_wins / max(total_attempts, 1)

    lines.append("### Top Category Combinations\n")
    lines.append("| Combination | Used | Won | Win Rate | Lift |")
    lines.append("|------------|------|-----|---------|------|")
    sorted_cats = sorted(
        [c for c in cat_combo_used if cat_combo_used[c] >= 10],
        key=lambda c: cat_combo_won[c]/max(cat_combo_used[c], 1),
        reverse=True
    )
    for combo in sorted_cats[:20]:
        used = cat_combo_used[combo]
        won = cat_combo_won[combo]
        rate = won / max(used, 1)
        lift = rate / max(baseline, 0.001)
        label = " + ".join(combo)
        lines.append(f"| **{label}** | {used} | {won} | **{rate*100:.1f}%** | {lift:.2f}× |")
    lines.append("")

    # Top variant combos (minimum 5 uses)
    lines.append("### Top Variant Combinations (min 5 uses)\n")
    lines.append("| Combination | Used | Won | Win Rate | Lift |")
    lines.append("|------------|------|-----|---------|------|")
    sorted_vars = sorted(
        [c for c in var_combo_used if var_combo_used[c] >= 5],
        key=lambda c: var_combo_won[c]/max(var_combo_used[c], 1),
        reverse=True
    )
    for combo in sorted_vars[:30]:
        used = var_combo_used[combo]
        won = var_combo_won[combo]
        rate = won / max(used, 1)
        lift = rate / max(baseline, 0.001)
        label = " + ".join(combo)
        lines.append(f"| {label} | {used} | {won} | **{rate*100:.1f}%** | {lift:.2f}× |")
    lines.append("")

    return "\n".join(lines)


def level_insights(records, meta):
    lines = []
    lines.append("## INSIGHTS — What Should the Planner Learn?\n")

    successes = [r for r in records if r['success']]
    failures = [r for r in records if not r['success']]

    # Double-down analysis
    dd_attempts = 0
    dd_wins = 0
    for r in records:
        traj = r['trajectory']
        for j in range(1, len(traj)):
            prev_conf = traj[j-1].get('extractor_confidence', 0)
            if prev_conf > 0.7:
                dd_attempts += 1
                if traj[j].get('success'):
                    dd_wins += 1
    lines.append("### Double-Down Effectiveness\n")
    lines.append(f"- Attempts after confidence > 0.7: **{dd_attempts}**")
    lines.append(f"- Wins from those: **{dd_wins}** ({dd_wins/max(dd_attempts,1)*100:.1f}%)")
    lines.append("")

    # Confidence escalation in failures
    stuck = 0
    escalating = 0
    peaked_fell = 0
    for r in failures:
        confs = [t.get('extractor_confidence', 0) for t in r['trajectory']]
        if not confs:
            continue
        if max(confs) == 0:
            stuck += 1
        elif confs == sorted(confs):
            escalating += 1
        else:
            max_idx = confs.index(max(confs))
            if max_idx < len(confs) - 1:
                peaked_fell += 1
            else:
                escalating += 1

    n_fail = len(failures)
    lines.append("### Confidence Escalation in Failures\n")
    lines.append("| Pattern | Count | % of Failures |")
    lines.append("|---------|-------|--------------|")
    lines.append(f"| Stuck at 0 (completely blocked) | {stuck} | {stuck/max(n_fail,1)*100:.1f}% |")
    lines.append(f"| **Escalating (getting closer)** | **{escalating}** | **{escalating/max(n_fail,1)*100:.1f}%** |")
    lines.append(f"| Peaked then fell (lost progress) | {peaked_fell} | {peaked_fell/max(n_fail,1)*100:.1f}% |")
    lines.append("")

    # Victim response patterns in failures
    short_resp = 0
    substantive = 0
    refusal = 0
    refusal_words = ['cannot', "can't", 'sorry', 'i am not', "i'm not", 'unable', 'not allowed', 'refuse']
    for r in failures:
        for t in r['trajectory']:
            resp = (t.get('response') or '').strip()
            if len(resp) < 20:
                short_resp += 1
            elif any(w in resp.lower() for w in refusal_words):
                refusal += 1
            else:
                substantive += 1

    total_fail_resp = short_resp + substantive + refusal
    lines.append("### Victim Response Patterns in Failures\n")
    lines.append("| Pattern | Count | % |")
    lines.append("|---------|-------|---|")
    lines.append(f"| Very Short (<20 chars) | {short_resp} | {short_resp/max(total_fail_resp,1)*100:.1f}% |")
    lines.append(f"| Substantive Response | {substantive} | {substantive/max(total_fail_resp,1)*100:.1f}% |")
    lines.append(f"| Explicit Refusal | {refusal} | {refusal/max(total_fail_resp,1)*100:.1f}% |")
    lines.append("")

    # Power combo effectiveness
    power_used = 0
    power_won = 0
    regular_used = 0
    regular_won = 0
    for r in records:
        for t in r['trajectory']:
            if t.get('power_combo'):
                power_used += 1
                if t.get('success'):
                    power_won += 1
            else:
                regular_used += 1
                if t.get('success'):
                    regular_won += 1

    lines.append("### Power Combo vs Regular Candidates\n")
    lines.append("| Type | Attempts | Wins | Win Rate |")
    lines.append("|------|----------|------|---------|")
    lines.append(f"| **Power Combo** | {power_used} | {power_won} | **{power_won/max(power_used,1)*100:.1f}%** |")
    lines.append(f"| Regular | {regular_used} | {regular_won} | {regular_won/max(regular_used,1)*100:.1f}% |")
    lines.append("")

    return "\n".join(lines)


def executive_summary(records, meta):
    lines = []
    lines.append("## Executive Summary — Key Decisions for Phase 4\n")

    total = len(records)
    successes = sum(1 for r in records if r['success'])
    rate = successes / total * 100

    # Top 3 strategies
    strat_won = Counter()
    strat_used = Counter()
    for r in records:
        for t in r['trajectory']:
            strat_used[t['strategy']] += 1
            if t.get('success'):
                strat_won[t['strategy']] += 1

    top_strats = sorted(strat_used.keys(), key=lambda s: strat_won[s]/max(strat_used[s],1), reverse=True)[:3]
    bottom_strats = sorted(strat_used.keys(), key=lambda s: strat_won[s]/max(strat_used[s],1))[:3]

    lines.append(f"1. **Overall Success Rate: {rate:.1f}%** — {'above' if rate > 40 else 'below'} the v2 baseline of 23.3%.\n")
    lines.append(f"2. **Top-3 strategies:** `{'`, `'.join(top_strats)}` — the Planner should always try these first.\n")
    lines.append(f"3. **Weakest strategies:** `{'`, `'.join(bottom_strats)}` — deprioritize or remove.\n")

    # Attempt distribution
    attempt1_wins = sum(1 for r in records if r['success'] and len(r['trajectory']) == 1)
    lines.append(f"4. **Attempt-1 wins:** {attempt1_wins}/{successes} ({attempt1_wins/max(successes,1)*100:.1f}%) — "
                 f"{'first-move selection is critical' if attempt1_wins/max(successes,1) > 0.5 else 'recovery matters significantly'}.\n")

    # V2 vs V3 comparison
    lines.append("5. **v2 → v3 Improvement Sources:**")
    lines.append("   - Weighted strategy selection (vs random shuffle)")
    lines.append("   - Double-down on high-confidence strategies (vs random switching)")
    lines.append("   - Lift-biased primitive variants (vs uniform random)")
    lines.append("   - Power combo injection (15% exploit candidates)")
    lines.append("   - Scenario filtering (removed unwinnable codes)")
    lines.append("   - max_attempts 5 → 10")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    traj_path = "data/oracle_trajectories_v3.jsonl"
    defense_path = "experiment/raw_dump_defenses.jsonl.bz2"
    out_path = "data/oracle_analysis_v3.md"

    print("Loading defense metadata...")
    meta = load_defense_metadata(defense_path)
    print(f"  Loaded {len(meta)} defense records")

    print("Loading trajectories...")
    records = load_trajectories(traj_path)
    print(f"  Loaded {len(records)} trajectory records")

    # Match rate
    matched = sum(1 for r in records if str(r['scenario_id']) in meta)
    print(f"  Matched {matched}/{len(records)} with defense metadata")

    # Build report
    print("\nGenerating report...")
    header = "# Oracle v3 Intelligence Report\n\n"
    header += f"**Source:** `{traj_path}`\n"
    header += f"**Scenarios:** {len(records)} | **Successes:** {sum(1 for r in records if r['success'])}\n\n---\n\n"

    sections = [
        level1_overall(records, meta),
        level2_defense(records, meta),
        level3_access_code(records, meta),
        level4_strategy(records, meta),
        level5_primitives(records, meta),
        level6_combo_mining(records, meta),
        level_insights(records, meta),
        executive_summary(records, meta),
    ]

    report = header + "\n---\n\n".join(sections)

    with open(out_path, 'w') as f:
        f.write(report)
    print(f"\n✅ Report saved to {out_path}")


if __name__ == "__main__":
    main()

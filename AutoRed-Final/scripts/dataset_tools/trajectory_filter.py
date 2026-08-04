"""
Trajectory Filter & SFT Dataset Builder
========================================
Scores, filters, and formats Oracle trajectories into a curated dataset
for training the Attack Planner via Supervised Fine-Tuning (SFT).

Scoring Dimensions:
  1. Reward       — Did it actually extract the code? (ground truth leak)
  2. Efficiency   — How few attempts did it take? (fewer = better)
  3. Diversity    — Strategy/primitive variety within the trajectory
  4. Quality      — Response engagement, meaningful transitions

Filtering Rules (from user spec):
  - Keep top 80% by composite score
  - Force 25% of output to come from attempt 2+ wins (multi-turn trajectories)
  - Deduplicate near-identical attack prompts

Output:
  - Scored trajectories JSONL (all scored, before filtering)
  - Curated SFT dataset JSONL (after filtering, ready for training)
  - Quality report (markdown summary)

Usage:
  python scripts/dataset_tools/trajectory_filter.py \
    --input data/oracle_trajectories_v4.jsonl \
    --output data/sft_dataset_v1.jsonl \
    --report data/trajectory_filter_report.md
"""

import os
import sys
import json
import argparse
import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from difflib import SequenceMatcher


# ═══════════════════════════════════════════════════════════════════════
# Scoring Functions
# ═══════════════════════════════════════════════════════════════════════

def score_reward(trajectory: Dict) -> float:
    """
    Score based on outcome.
    - Success = 1.0
    - Failure = 0.0
    
    We only train on successes, but this is used for analysis.
    """
    return 1.0 if trajectory["success"] else 0.0


def score_efficiency(trajectory: Dict, max_attempts: int = 10) -> float:
    """
    Score based on how quickly the Oracle won.
    
    Formula: 1.0 - (num_attempts - 1) / (max_attempts - 1)
    
    Attempt 1 win = 1.0 (perfect)
    Attempt 10 win = 0.0 (worst possible success)
    """
    n = trajectory["num_attempts"]
    if max_attempts <= 1:
        return 1.0
    return max(0.0, 1.0 - (n - 1) / (max_attempts - 1))


def score_diversity(trajectory: Dict) -> float:
    """
    Score based on how many unique strategies and primitives were explored.
    
    Rewards trajectories that tried multiple approaches before winning,
    as these are more informative for training the Planner.
    
    For single-attempt wins: returns 0.5 (neutral — no transitions to learn)
    For multi-attempt wins:
      - Strategy diversity: unique_strategies / total_attempts
      - Primitive diversity: unique_primitive_types / total_primitives_used
      - Combined: average of both
    """
    steps = trajectory["trajectory"]
    
    if len(steps) <= 1:
        return 0.5  # Neutral — nothing to learn about transitions
    
    # Strategy diversity
    strategies = [s["strategy"] for s in steps]
    unique_strategies = len(set(strategies))
    strategy_diversity = unique_strategies / len(strategies)
    
    # Primitive diversity (count unique (type, variant) pairs)
    all_primitives = []
    for step in steps:
        for prim in step.get("primitives", []):
            if isinstance(prim, list) and len(prim) >= 2:
                all_primitives.append(tuple(prim[:2]))
    
    if all_primitives:
        unique_primitives = len(set(all_primitives))
        primitive_diversity = min(1.0, unique_primitives / max(len(all_primitives), 1))
    else:
        primitive_diversity = 0.0
    
    return (strategy_diversity + primitive_diversity) / 2.0


def score_quality(trajectory: Dict) -> float:
    """
    Score based on response engagement quality.
    
    Measures:
      1. Response engagement: avg response length in the trajectory
         - Longer responses = victim is engaged = higher quality attack
      2. Meaningful transitions: did the strategy change when the previous
         one failed? (as opposed to repeating the same failed strategy)
      3. Winning step quality: how long was the winning response?
    """
    steps = trajectory["trajectory"]
    
    if not steps:
        return 0.0
    
    # 1. Average response engagement (normalized)
    response_lengths = [len(s.get("response", "")) for s in steps]
    avg_resp_len = sum(response_lengths) / len(response_lengths)
    # Normalize: 0-20 chars = 0.0, 200+ chars = 1.0
    engagement = min(1.0, max(0.0, (avg_resp_len - 20) / 180))
    
    # 2. Meaningful transitions (only for multi-step)
    if len(steps) > 1:
        transitions = 0
        for i in range(1, len(steps)):
            if steps[i]["strategy"] != steps[i-1]["strategy"]:
                transitions += 1
        transition_ratio = transitions / (len(steps) - 1)
    else:
        transition_ratio = 0.5  # Neutral for single-step
    
    # 3. Winning step response quality
    winning_step = steps[-1] if trajectory["success"] else None
    if winning_step:
        win_resp_len = len(winning_step.get("response", ""))
        win_quality = min(1.0, max(0.0, win_resp_len / 100))
    else:
        win_quality = 0.0
    
    # Weighted combination
    return 0.4 * engagement + 0.3 * transition_ratio + 0.3 * win_quality


def compute_composite_score(trajectory: Dict, max_attempts: int = 10) -> Dict[str, float]:
    """
    Compute all individual scores and the weighted composite.
    
    Weights:
      - Reward:     0.30 (did it succeed?)
      - Efficiency: 0.30 (how quickly?)
      - Diversity:  0.15 (strategy exploration)
      - Quality:    0.25 (engagement quality)
    """
    reward = score_reward(trajectory)
    efficiency = score_efficiency(trajectory, max_attempts)
    diversity = score_diversity(trajectory)
    quality = score_quality(trajectory)
    
    composite = (
        0.30 * reward +
        0.30 * efficiency +
        0.15 * diversity +
        0.25 * quality
    )
    
    return {
        "reward": round(reward, 4),
        "efficiency": round(efficiency, 4),
        "diversity": round(diversity, 4),
        "quality": round(quality, 4),
        "composite": round(composite, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Deduplication
# ═══════════════════════════════════════════════════════════════════════

def extract_attack_fingerprint(trajectory: Dict) -> str:
    """
    Create a fingerprint from the winning attack for deduplication.
    Uses the final (winning) attack text, lowercased and stripped.
    """
    steps = trajectory.get("trajectory", [])
    if not steps:
        return ""
    # Use the last step (winning step for successes)
    last_attack = steps[-1].get("attack", "").strip().lower()
    # Truncate to first 200 chars for comparison
    return last_attack[:200]


def deduplicate_trajectories(
    trajectories: List[Dict],
    similarity_threshold: float = 0.85
) -> Tuple[List[Dict], int]:
    """
    Remove near-duplicate trajectories based on attack text similarity.
    Uses a greedy approach: keep the first (higher-scored) trajectory,
    drop later ones that are too similar.
    
    Returns: (deduplicated_list, num_removed)
    """
    if not trajectories:
        return [], 0
    
    kept = []
    kept_fingerprints = []
    kept_counters = []
    removed = 0
    
    # Pre-calculate Counters for the fingerprints
    for traj in trajectories:
        fp = extract_attack_fingerprint(traj)
        
        if not fp:
            kept.append(traj)
            kept_fingerprints.append(fp)
            kept_counters.append(None)
            continue
            
        c_fp = Counter(fp)
        len_fp = len(fp)
        is_duplicate = False
        
        len_threshold = similarity_threshold / (2.0 - similarity_threshold)
        
        for existing_fp, existing_counter in zip(kept_fingerprints, kept_counters):
            if not existing_fp:
                continue
            
            # 1. Quick length check
            len_existing = len(existing_fp)
            len_ratio = min(len_fp, len_existing) / max(len_fp, len_existing, 1)
            if len_ratio < len_threshold:
                continue
                
            # 2. Quick character bag overlap check
            # Math: 2 * common >= threshold * (len_fp + len_existing)
            min_common = 0.5 * similarity_threshold * (len_fp + len_existing)
            
            # Calculate intersection
            common = sum((c_fp & existing_counter).values())
            if common < min_common:
                continue
            
            # 3. Full SequenceMatcher check
            similarity = SequenceMatcher(None, fp, existing_fp).ratio()
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
                
        if is_duplicate:
            removed += 1
        else:
            kept.append(traj)
            kept_fingerprints.append(fp)
            kept_counters.append(c_fp)
            
    return kept, removed


# ═══════════════════════════════════════════════════════════════════════
# Filtering Pipeline
# ═══════════════════════════════════════════════════════════════════════

def filter_trajectories(
    trajectories: List[Dict],
    top_percentile: float = 0.80,
    late_win_ratio: float = 0.25,
    similarity_threshold: float = 0.85,
    max_attempts: int = 10,
    hard_mining: bool = False,
    scenarios_metadata: Dict = None,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Full filtering pipeline:
    
    1. Score all trajectories
    2. Keep only successes
    3. Deduplicate
    4. If hard_mining: filter for hard criteria (no top_percentile cut)
       Else: Keep top `top_percentile` (80%)
    5. Enforce `late_win_ratio` (25%) from attempt 2+ wins
    
    Returns: (filtered_trajectories, filter_stats)
    """
    stats = {
        "input_total": len(trajectories),
        "input_successes": 0,
        "input_failures": 0,
        "after_dedup": 0,
        "dedup_removed": 0,
        "after_quality_cut": 0,
        "quality_cutoff_score": 0.0,
        "after_diversity_enforcement": 0,
        "attempt1_wins": 0,
        "attempt2plus_wins": 0,
        "final_count": 0,
    }
    
    # Step 1: Score all trajectories
    for traj in trajectories:
        traj["scores"] = compute_composite_score(traj, max_attempts)
    
    # Step 2: Separate successes and failures
    successes = [t for t in trajectories if t["success"]]
    failures = [t for t in trajectories if not t["success"]]
    stats["input_successes"] = len(successes)
    stats["input_failures"] = len(failures)
    
    if not successes:
        print("[WARN] No successful trajectories found. Nothing to filter.")
        return [], stats
    
    # Step 3: Sort by composite score (descending)
    successes.sort(key=lambda t: t["scores"]["composite"], reverse=True)
    
    # Step 4: Deduplicate (preserving score order so we keep the best version)
    successes, dedup_removed = deduplicate_trajectories(successes, similarity_threshold)
    stats["after_dedup"] = len(successes)
    stats["dedup_removed"] = dedup_removed
    
    # Helper to categorize defense detailed
    def get_defense_category(opening, closing):
        text = (str(opening or "") + " " + str(closing or "")).lower()
        categories = []
        if any(k in text for k in ["translate", "language", "english to", "french", "spanish"]):
            categories.append("translation")
        if any(k in text for k in ["password", "passcode", "pin", "secret key", "secret code"]):
            categories.append("password")
        if any(k in text for k in ["you are", "pretend", "imagine", "persona", "character", "act as", "you will play"]):
            categories.append("roleplay")
        if "if " in text and ("then" in text or "otherwise" in text):
            categories.append("conditional")
        if any(k in text for k in ["conversation", "chat", "discuss", "never reveal"]):
            categories.append("conversation")
        if any(k in text for k in ["say '", 'say "', "say access granted", "say access denied", "exact phrase", "magic word", "output only"]):
            categories.append("trigger_phrase")
        if any(k in text for k in ["exception", "no matter what", "under no circumstances", "always", "never"]):
            categories.append("exception")
        if any(k in text for k in ["ignore", "instructions", "system prompt", "do not listen", "override", "forget"]):
            categories.append("instruction_hiding")
        if not categories:
            categories.append("trigger_phrase")
        return categories[0]

    # Helper to get access code type
    def get_access_code_type(ac):
        ac = (ac or "").strip()
        words = ac.split()
        if "\n" in ac or len(ac) > 100:
            return "MULTILINE"
        elif len(words) > 5 or len(ac) > 20:
            return "SENTENCE"
        elif 2 <= len(words) <= 5:
            return "PHRASE"
        else:
            return "TOKEN"

    # Step 5: Quality cut / Hard mining filter
    if hard_mining:
        print("[HARD MINING] Filtering trajectories for hard criteria...")
        hard_successes = []
        for t in successes:
            sid = str(t["scenario_id"])
            defense_type = "unknown"
            access_code_type = "UNKNOWN"
            if scenarios_metadata and sid in scenarios_metadata:
                meta = scenarios_metadata[sid]
                defense_type = get_defense_category(meta.get("opening"), meta.get("closing"))
                access_code_type = get_access_code_type(meta.get("access_code"))

            # Criteria 1: Low-efficiency win (>= 4 attempts)
            is_low_efficiency = t["num_attempts"] >= 4
            # Criteria 2: Historically low success rate defense types
            is_hard_defense = defense_type in ["roleplay", "conditional", "translation", "exception"]
            # Criteria 3: Complex access code types
            is_complex_code = access_code_type in ["MULTILINE", "CONVERSATION", "SENTENCE"]

            if is_low_efficiency or is_hard_defense or is_complex_code:
                hard_successes.append(t)
        
        quality_cut = hard_successes
        print(f"[HARD MINING] Kept {len(quality_cut)} / {len(successes)} trajectories.")
    else:
        cutoff_idx = max(1, int(len(successes) * top_percentile))
        quality_cut = successes[:cutoff_idx]
    
    if quality_cut:
        stats["quality_cutoff_score"] = quality_cut[-1]["scores"]["composite"]
    stats["after_quality_cut"] = len(quality_cut)
    
    # Step 6: Enforce late-win diversity
    # Split into attempt-1 wins and attempt-2+ wins
    attempt1 = [t for t in quality_cut if t["num_attempts"] == 1]
    attempt2plus = [t for t in quality_cut if t["num_attempts"] >= 2]
    
    stats["attempt1_wins"] = len(attempt1)
    stats["attempt2plus_wins"] = len(attempt2plus)
    
    # Check if we need to enforce the ratio
    total = len(quality_cut)
    target_late = max(1, int(total * late_win_ratio))
    
    if len(attempt2plus) < target_late:
        # Not enough late wins — keep all of them, fill rest with attempt-1
        final = attempt2plus[:]
        remaining_budget = total - len(final)
        final.extend(attempt1[:remaining_budget])
        print(f"[INFO] Only {len(attempt2plus)} late wins available (target: {target_late}). "
              f"Keeping all late wins.")
    else:
        # Enough late wins — enforce the ratio
        # Take exactly target_late from attempt2plus, rest from attempt1
        remaining_budget = total - target_late
        final = attempt2plus[:target_late]
        final.extend(attempt1[:remaining_budget])
    
    # Re-sort final by composite score
    final.sort(key=lambda t: t["scores"]["composite"], reverse=True)
    
    stats["after_diversity_enforcement"] = len(final)
    stats["final_count"] = len(final)
    
    return final, stats


# ═══════════════════════════════════════════════════════════════════════
# SFT Data Formatter
# ═══════════════════════════════════════════════════════════════════════

def format_for_sft(trajectory: Dict, scenarios_metadata: Dict = None) -> List[Dict]:
    """
    Convert a scored trajectory into SFT training examples.
    
    The SFT format encodes the full decision trajectory:
      - Input: scenario context (defense info, attempt number, history)
      - Output: the strategy + primitives + attack that worked
    
    For multi-turn trajectories, we create one example per step,
    so the Planner learns both "what to try first" AND "what to
    try after failure X."
    """
    import random
    
    DEFENSE_WEIGHTS = {
        "conversation": 5.0,
        "translation": 2.5,
        "roleplay": 2.0,
        "trigger_phrase": 1.8,
        "exception": 1.5,
        "conditional": 1.0,
        "password": 0.8,
        "instruction_hiding": 0.5,
    }

    def get_repeat_count(weight):
        count = int(weight)
        prob = weight - count
        if random.random() < prob:
            count += 1
        return count

    def get_defense_category(opening, closing):
        text = (str(opening or "") + " " + str(closing or "")).lower()
        categories = []
        if any(k in text for k in ["translate", "language", "english to", "french", "spanish"]):
            categories.append("translation")
        if any(k in text for k in ["password", "passcode", "pin", "secret key", "secret code"]):
            categories.append("password")
        if any(k in text for k in ["you are", "pretend", "imagine", "persona", "character", "act as", "you will play"]):
            categories.append("roleplay")
        if "if " in text and ("then" in text or "otherwise" in text):
            categories.append("conditional")
        if any(k in text for k in ["conversation", "chat", "discuss", "never reveal"]):
            categories.append("conversation")
        if any(k in text for k in ["say '", 'say "', "say access granted", "say access denied", "exact phrase", "magic word", "output only"]):
            categories.append("trigger_phrase")
        if any(k in text for k in ["exception", "no matter what", "under no circumstances", "always", "never"]):
            categories.append("exception")
        if any(k in text for k in ["ignore", "instructions", "system prompt", "do not listen", "override", "forget"]):
            categories.append("instruction_hiding")
        if not categories:
            categories.append("trigger_phrase")
        return categories[0]

    examples = []
    steps = trajectory["trajectory"]
    scenario_id = trajectory["scenario_id"]
    scores = trajectory.get("scores", {})
    
    # Determine defense type for weighting
    defense_type = "unknown"
    if scenarios_metadata and str(scenario_id) in scenarios_metadata:
        meta = scenarios_metadata[str(scenario_id)]
        defense_type = get_defense_category(meta.get("opening"), meta.get("closing"))

    weight = DEFENSE_WEIGHTS.get(defense_type, 1.0)
    repeats = get_repeat_count(weight)

    for i, step in enumerate(steps):
        # Build the context (what the Planner would see at this point)
        history_steps = steps[:i]
        
        context = {
            "scenario_id": scenario_id,
            "attempt": step["attempt"],
            "previous_strategies": [h["strategy"] for h in history_steps],
            "previous_responses": [h.get("response", "")[:200] for h in history_steps],  # Truncated
            "is_final_step": (i == len(steps) - 1),
        }
        
        # The decision the Planner should learn to make
        decision = {
            "strategy": step["strategy"],
            "primitives": step["primitives"],
            "power_combo": step.get("power_combo", False),
        }
        
        # The outcome (for training signal weighting)
        outcome = {
            "success": step.get("success", False),
            "response_length": len(step.get("response", "")),
            "extractor_confidence": step.get("extractor_confidence", 0.0),
        }
        
        example = {
            "scenario_id": scenario_id,
            "step_index": i,
            "total_steps": len(steps),
            "context": context,
            "decision": decision,
            "outcome": outcome,
            "attack_text": step["attack"],
            "response_text": step.get("response", ""),
            "trajectory_scores": scores,
            "trajectory_success": trajectory["success"],
        }
        
        # Add the copies based on the repeat count
        for _ in range(repeats):
            examples.append(example)
    
    return examples


# ═══════════════════════════════════════════════════════════════════════
# Report Generator
# ═══════════════════════════════════════════════════════════════════════

def generate_report(
    all_trajectories: List[Dict],
    filtered: List[Dict],
    stats: Dict,
    sft_examples: List[Dict],
) -> str:
    """Generate a markdown quality report."""
    
    lines = [
        "# Trajectory Filter Report",
        "",
        f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Pipeline Summary",
        "",
        "| Stage | Count |",
        "|-------|-------|",
        f"| Input trajectories | {stats['input_total']} |",
        f"| Successes | {stats['input_successes']} ({stats['input_successes']/max(stats['input_total'],1)*100:.1f}%) |",
        f"| Failures | {stats['input_failures']} |",
        f"| After deduplication | {stats['after_dedup']} (removed {stats['dedup_removed']}) |",
        f"| After quality cut (top 80%) | {stats['after_quality_cut']} |",
        f"| Quality cutoff score | {stats['quality_cutoff_score']:.4f} |",
        f"| After diversity enforcement | {stats['after_diversity_enforcement']} |",
        f"| **Final curated set** | **{stats['final_count']}** |",
        f"| SFT training examples | {len(sft_examples)} |",
        "",
    ]
    
    # Score distribution of filtered set
    if filtered:
        composites = [t["scores"]["composite"] for t in filtered]
        efficiencies = [t["scores"]["efficiency"] for t in filtered]
        qualities = [t["scores"]["quality"] for t in filtered]
        diversities = [t["scores"]["diversity"] for t in filtered]
        
        lines.extend([
            "## 2. Score Distribution (Filtered Set)",
            "",
            "| Dimension | Min | Max | Mean | Median |",
            "|-----------|-----|-----|------|--------|",
        ])
        
        for name, vals in [("Composite", composites), ("Efficiency", efficiencies),
                           ("Quality", qualities), ("Diversity", diversities)]:
            sorted_vals = sorted(vals)
            median = sorted_vals[len(sorted_vals)//2]
            lines.append(
                f"| {name} | {min(vals):.4f} | {max(vals):.4f} | "
                f"{sum(vals)/len(vals):.4f} | {median:.4f} |"
            )
        lines.append("")
    
    # Attempt distribution
    if filtered:
        attempt_dist = Counter(t["num_attempts"] for t in filtered)
        lines.extend([
            "## 3. Attempt Distribution (Filtered Set)",
            "",
            "| Attempts | Count | % |",
            "|----------|-------|---|",
        ])
        for k in sorted(attempt_dist.keys()):
            pct = attempt_dist[k] / len(filtered) * 100
            lines.append(f"| {k} | {attempt_dist[k]} | {pct:.1f}% |")
        
        lines.extend([
            "",
            f"- Attempt 1 wins: {stats['attempt1_wins']}",
            f"- Attempt 2+ wins: {stats['attempt2plus_wins']}",
            f"- Late-win ratio: {stats['attempt2plus_wins']/max(len(filtered),1)*100:.1f}%",
            "",
        ])
    
    # Strategy breakdown
    if filtered:
        # Count winning strategies (last step of each trajectory)
        winning_strategies = Counter()
        for t in filtered:
            steps = t["trajectory"]
            if steps:
                winning_strategies[steps[-1]["strategy"]] += 1
        
        lines.extend([
            "## 4. Winning Strategy Distribution",
            "",
            "| Strategy | Wins | % |",
            "|----------|------|---|",
        ])
        for strategy, count in winning_strategies.most_common():
            pct = count / len(filtered) * 100
            lines.append(f"| {strategy} | {count} | {pct:.1f}% |")
        lines.append("")
    
    # Primitive combo analysis
    if filtered:
        winning_combos = Counter()
        for t in filtered:
            steps = t["trajectory"]
            if steps:
                prims = steps[-1].get("primitives", [])
                combo_key = " + ".join(
                    f"{p[0]}:{p[1]}" for p in prims if isinstance(p, list) and len(p) >= 2
                )
                if combo_key:
                    winning_combos[combo_key] += 1
        
        lines.extend([
            "## 5. Top Winning Primitive Combinations",
            "",
            "| Combination | Wins |",
            "|-------------|------|",
        ])
        for combo, count in winning_combos.most_common(15):
            lines.append(f"| {combo} | {count} |")
        lines.append("")
    
    # SFT dataset summary
    if sft_examples:
        step_dist = Counter(ex["step_index"] for ex in sft_examples)
        lines.extend([
            "## 6. SFT Dataset Summary",
            "",
            f"- Total training examples: {len(sft_examples)}",
            f"- From {len(filtered)} unique trajectories",
            f"- Average examples per trajectory: {len(sft_examples)/max(len(filtered),1):.1f}",
            "",
            "| Step Index | Examples | Description |",
            "|------------|----------|-------------|",
        ])
        for k in sorted(step_dist.keys()):
            desc = "First attempt" if k == 0 else f"After {k} failure(s)"
            lines.append(f"| {k} | {step_dist[k]} | {desc} |")
        lines.append("")
    
    # Quality recommendations
    lines.extend([
        "## 7. Recommendations",
        "",
    ])
    
    if stats["final_count"] < 400:
        lines.append(
            f"> [!WARNING]\n"
            f"> Only {stats['final_count']} curated trajectories. "
            f"Recommend running more Oracle scenarios to reach 500+ for robust SFT."
        )
    elif stats["final_count"] >= 1000:
        lines.append(
            f"> [!TIP]\n"
            f"> {stats['final_count']} curated trajectories is excellent for SFT. "
            f"Proceed with training."
        )
    else:
        lines.append(
            f"> [!NOTE]\n"
            f"> {stats['final_count']} curated trajectories is adequate for an initial SFT run. "
            f"Consider running more Oracle scenarios for better coverage."
        )
    
    if stats["attempt2plus_wins"] < stats["final_count"] * 0.20:
        lines.append(
            f"\n> [!WARNING]\n"
            f"> Low late-win ratio ({stats['attempt2plus_wins']}/{stats['final_count']}). "
            f"The Planner will be biased toward attempt-1 strategies and won't learn recovery."
        )
    
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Trajectory Filter & SFT Dataset Builder"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to Oracle trajectories JSONL (or glob pattern for worker files)"
    )
    parser.add_argument(
        "--output", type=str, default="data/sft_dataset_v1.jsonl",
        help="Path to write curated SFT dataset"
    )
    parser.add_argument(
        "--scored-output", type=str, default=None,
        help="Path to write all scored trajectories (before filtering)"
    )
    parser.add_argument(
        "--report", type=str, default="data/trajectory_filter_report.md",
        help="Path to write quality report"
    )
    parser.add_argument(
        "--top-percentile", type=float, default=0.80,
        help="Keep top N%% of trajectories by composite score (default: 0.80)"
    )
    parser.add_argument(
        "--late-win-ratio", type=float, default=0.25,
        help="Force this ratio of attempt 2+ wins in output (default: 0.25)"
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=0.85,
        help="Deduplicate attacks with similarity >= this (default: 0.85)"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=10,
        help="Max attempts per scenario (for efficiency scoring)"
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help="Only score trajectories, don't filter or format"
    )
    parser.add_argument(
        "--scenarios", type=str, default="experiment/oracle_v3_scenarios_5000.jsonl.bz2",
        help="Path to scenarios metadata file (.bz2)"
    )
    parser.add_argument(
        "--hard-mining", action="store_true",
        help="Enable hard example mining based on low efficiency, low success defenses, and complex code types"
    )
    args = parser.parse_args()
    
    # ── Load Trajectories ──
    print(f"[LOAD] Loading trajectories from: {args.input}")
    
    trajectories = []
    input_path = Path(args.input)
    
    if input_path.is_file():
        with open(input_path, "r") as f:
            for line in f:
                if line.strip():
                    trajectories.append(json.loads(line))
    else:
        # Try glob pattern (e.g., "data/oracle_trajectories_v4_w*.jsonl")
        import glob
        files = sorted(glob.glob(args.input))
        if not files:
            print(f"[ERROR] No files found matching: {args.input}")
            sys.exit(1)
        for fpath in files:
            print(f"  Loading: {fpath}")
            with open(fpath, "r") as f:
                for line in f:
                    if line.strip():
                        trajectories.append(json.loads(line))
    
    print(f"[LOAD] Loaded {len(trajectories)} trajectories")
    
    successes = sum(1 for t in trajectories if t["success"])
    failures = len(trajectories) - successes
    print(f"  Successes: {successes} ({successes/max(len(trajectories),1)*100:.1f}%)")
    print(f"  Failures:  {failures}")

    # ── Load Scenarios Metadata ──
    scenarios_metadata = {}
    if args.scenarios and os.path.exists(args.scenarios):
        print(f"[LOAD] Loading scenarios metadata from: {args.scenarios}")
        import bz2
        with bz2.open(args.scenarios, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    scenarios_metadata[str(data['defense_id'])] = {
                        'opening': data.get('opening_defense', ''),
                        'closing': data.get('closing_defense', ''),
                        'access_code': data.get('access_code', '')
                    }
                except Exception:
                    continue
        print(f"  Loaded {len(scenarios_metadata)} scenario definitions")
    
    # ── Score Only Mode ──
    if args.score_only:
        print("\n[SCORE] Scoring all trajectories...")
        for traj in trajectories:
            traj["scores"] = compute_composite_score(traj, args.max_attempts)
        
        out_path = args.scored_output or args.output
        with open(out_path, "w") as f:
            for t in trajectories:
                f.write(json.dumps(t) + "\n")
        print(f"[DONE] Scored trajectories saved to: {out_path}")
        return
    
    # ── Full Pipeline ──
    print("\n[FILTER] Running full filtering pipeline...")
    
    filtered, stats = filter_trajectories(
        trajectories,
        top_percentile=args.top_percentile,
        late_win_ratio=args.late_win_ratio,
        similarity_threshold=args.similarity_threshold,
        max_attempts=args.max_attempts,
        hard_mining=args.hard_mining,
        scenarios_metadata=scenarios_metadata,
    )
    
    print(f"  Input:     {stats['input_total']}")
    print(f"  Successes: {stats['input_successes']}")
    print(f"  After dedup: {stats['after_dedup']} (-{stats['dedup_removed']})")
    print(f"  After quality cut: {stats['after_quality_cut']}")
    print(f"  Final:     {stats['final_count']}")
    
    # ── Format for SFT ──
    print("\n[SFT] Formatting trajectories for SFT training...")
    sft_examples = []
    for traj in filtered:
        examples = format_for_sft(traj, scenarios_metadata=scenarios_metadata)
        sft_examples.extend(examples)
    
    print(f"  Generated {len(sft_examples)} SFT training examples")
    
    # ── Save Outputs ──
    # SFT dataset
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\n[SAVE] SFT dataset: {out_path}")
    
    # Scored trajectories (optional)
    if args.scored_output:
        scored_path = Path(args.scored_output)
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scored_path, "w") as f:
            for t in trajectories:
                f.write(json.dumps(t) + "\n")
        print(f"[SAVE] Scored trajectories: {scored_path}")
    
    # Quality report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(trajectories, filtered, stats, sft_examples)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[SAVE] Quality report: {report_path}")
    
    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Trajectory Filter — Complete")
    print(f"{'='*60}")
    print(f"  Input trajectories:   {stats['input_total']}")
    print(f"  Curated trajectories: {stats['final_count']}")
    print(f"  SFT training examples: {len(sft_examples)}")
    print(f"  Late-win ratio:       {stats['attempt2plus_wins']}/{stats['final_count']} "
          f"({stats['attempt2plus_wins']/max(stats['final_count'],1)*100:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

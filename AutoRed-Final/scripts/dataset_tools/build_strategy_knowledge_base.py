"""Build the strategy effectiveness knowledge base.

Exports ``build_strategy_kb(...)`` (callable from ``kb_updater``) plus a thin
``main()`` CLI wrapper. Key design points vs the legacy version:

* Reads ``defense_classifier_dataset-Part1.jsonl`` + ``-Part2.jsonl`` (the legacy
  code read the nonexistent singular file → ``FileNotFoundError``).
* Per-attempt quality tiers (Constraint 3) — a record counts as a success only
  if it carries a Tier-1 (``verification_success OR ground_truth_found OR
  access_granted``) or Tier-2 (``extractor_match``) per-attempt signal. The
  poisoned run-level ``success`` boolean is IGNORED, so the 29,958 historical
  records with no per-attempt signal self-correct (demoted to failure on rebuild).
* Model dimension (Constraint 1) — builds ``by_model[model][defense_type][strategy]``
  plus a model-agnostic ``matrix[defense_type][strategy]`` fallback.
* Content-hash dedup (Constraint 2) — skips records sharing a ``dedup_key``.
* Staleness downweighting (Constraint 4) — records older than 90 days get weight
  0.5 in the success-rate denominator (no hard drop).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from dateutil.parser import isoparse as _isoparse  # type: ignore
except Exception:  # pragma: no cover - dateutil optional
    _isoparse = None  # type: ignore


# Staleness: records older than this get weight 0.5 (no hard drop).
_STALE_DAYS = 90
_STALE_WEIGHT = 0.5
# Noise floor (Finding 7): strategies need this many attempts to be reported.
_MIN_ATTEMPTS = 5


def _normalize_attack(attack: str) -> str:
    return re.sub(r"\s+", " ", attack.strip().lower())


def _dedup_key(scenario_id: str, strategy: str, attack: str) -> str:
    h = hashlib.sha256()
    h.update(f"{scenario_id}|{strategy}|{_normalize_attack(attack)}".encode("utf-8"))
    return h.hexdigest()[:16]


def _parse_timestamp(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        if _isoparse is not None:
            return _isoparse(ts)
        # Fallback: ISO 8601 with trailing 'Z' or offset.
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _staleness_weight(ts: str) -> float:
    """1.0 for fresh records, 0.5 for records older than _STALE_DAYS."""
    dt = _parse_timestamp(ts)
    if dt is None:
        return 1.0  # unparseable → don't downweight
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - dt).days
    return _STALE_WEIGHT if age_days > _STALE_DAYS else 1.0


def _load_defense_map(data_dir: str, verbose: bool = True) -> dict[str, str]:
    """Map defense_id -> primary_type. Reads Part1+Part2 (falls back to singular)."""
    data_dir_path = Path(data_dir)
    defense_to_type: dict[str, str] = {}

    parts = [
        data_dir_path / "defense_classifier_dataset-Part1.jsonl",
        data_dir_path / "defense_classifier_dataset-Part2.jsonl",
    ]
    singular = data_dir_path / "defense_classifier_dataset.jsonl"
    if not any(p.exists() for p in parts) and singular.exists():
        parts = [singular]

    for part in parts:
        if not part.exists():
            if verbose:
                print(f"Warning: {part} not found.")
            continue
        with open(part, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                did = str(data.get("defense_id", ""))
                if did:
                    defense_to_type[did] = data.get("primary_type", "unknown")
    return defense_to_type


def _record_tier(record: dict) -> int:
    """Return 1 (RAG-eligible), 2 (KB-counts-only), or 3 (failure).

    Reads per-attempt flags and IGNORES the poisoned run-level ``success``
    boolean — this is the self-correction mechanism for the 29,958 poisoned
    historical records (all stamped ``success=True`` with no per-attempt signal).
    """
    verification_success = bool(record.get("verification_success", False))
    ground_truth_found = bool(record.get("ground_truth_found", False)) or bool(
        record.get("ground_truth_leaked", False)
    )
    access_granted = bool(record.get("access_granted", False))
    extractor_match = bool(record.get("extractor_match", False)) or bool(
        record.get("extractor_success", False)
    )
    if verification_success or ground_truth_found or access_granted:
        return 1
    if extractor_match:
        return 2
    return 3


def build_strategy_kb(
    success_data_path: str,
    failure_data_path: str,
    output_path: str,
    data_dir: str = "data",
    verbose: bool = True,
) -> dict:
    """Build the strategy effectiveness KB with model dimension, dedup, tiers, staleness.

    Returns a summary dict: {total, kept, deduped, tier1, tier2, tier3,
    demoted_from_poisoned}.
    """
    defense_to_type = _load_defense_map(data_dir, verbose=verbose)
    if verbose:
        print(f"Loaded {len(defense_to_type)} mapped defenses.")

    # by_model[model][defense_type][strategy] = {"successes": 0.0, "failures": 0.0}
    # (floats because staleness weighting makes them fractional)
    by_model: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"successes": 0.0, "failures": 0.0}))
    )
    # Agnostic (collapsed across models) — the fallback tier.
    matrix_agnostic: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"successes": 0.0, "failures": 0.0}))
    global_strategy_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"successes": 0.0, "failures": 0.0})
    global_defense_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"successes": 0.0, "failures": 0.0})

    seen_keys: set[str] = set()
    total = 0
    kept = 0
    deduped = 0
    tier1 = 0
    tier2 = 0
    tier3 = 0
    demoted_from_poisoned = 0

    def process_file(path: str, is_success_file: bool) -> None:
        nonlocal total, kept, deduped, tier1, tier2, tier3, demoted_from_poisoned
        if not os.path.exists(path):
            if verbose:
                print(f"Warning: {path} not found.")
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                scenario_id = str(data.get("scenario_id", "")).replace("bench_", "")
                strategy = str(data.get("strategy", "unknown"))
                attack = str(data.get("attack", ""))
                victim_model = str(data.get("victim_model", "unknown"))
                ts = str(data.get("timestamp", ""))

                # Dedup (Constraint 2).
                dk = data.get("dedup_key") or _dedup_key(scenario_id, strategy, attack)
                if dk in seen_keys:
                    deduped += 1
                    continue
                seen_keys.add(dk)
                kept += 1

                # Per-attempt tier (IGNORES poisoned run-level `success`).
                tier = _record_tier(data)
                poisoned_success = bool(data.get("success", False))
                if tier == 1:
                    tier1 += 1
                    is_success = True
                elif tier == 2:
                    tier2 += 1
                    is_success = True
                else:
                    tier3 += 1
                    is_success = False
                    # Track self-correction: a record stamped success=True by the
                    # poisoned logger but with no per-attempt signal.
                    if poisoned_success:
                        demoted_from_poisoned += 1

                # Override file placement: a "success" file record that is actually
                # Tier-3, or a "failure" file record that is Tier-1/2, follows the
                # per-attempt tier, not the file it came from.
                def_type = defense_to_type.get(scenario_id, "unknown")
                w = _staleness_weight(ts)

                model_bucket = by_model[victim_model]
                strat_bucket = model_bucket[def_type][strategy]
                agnostic_bucket = matrix_agnostic[def_type][strategy]
                if is_success:
                    strat_bucket["successes"] += w
                    agnostic_bucket["successes"] += w
                    global_strategy_stats[strategy]["successes"] += w
                    global_defense_stats[def_type]["successes"] += w
                else:
                    strat_bucket["failures"] += w
                    agnostic_bucket["failures"] += w
                    global_strategy_stats[strategy]["failures"] += w
                    global_defense_stats[def_type]["failures"] += w

    if verbose:
        print("Processing successes...")
    process_file(success_data_path, is_success_file=True)
    if verbose:
        print("Processing failures...")
    process_file(failure_data_path, is_success_file=False)

    def _finalize(bucket: dict[str, dict[str, float]]) -> dict:
        out = {}
        for key, strategies in bucket.items():
            out[key] = {}
            for strat, counts in strategies.items():
                s = counts["successes"]
                f = counts["failures"]
                t = s + f
                rate = (s / t * 100) if t > 0 else 0.0
                out[key][strat] = {
                    "success_rate": round(rate, 2),
                    "total_attempts": round(t, 2),
                    "successes": round(s, 2),
                    "failures": round(f, 2),
                }
        return out

    by_model_final = {model: _finalize(model_bucket) for model, model_bucket in by_model.items()}
    matrix_final = _finalize(matrix_agnostic)

    def _finalize_flat(stats: dict[str, dict[str, float]]) -> dict:
        out = {}
        for key, counts in stats.items():
            s = counts["successes"]
            f = counts["failures"]
            t = s + f
            rate = (s / t * 100) if t > 0 else 0.0
            out[key] = {"success_rate": round(rate, 2), "total_attempts": round(t, 2)}
        return out

    output_data = {
        "by_model": by_model_final,
        "matrix": matrix_final,  # model-agnostic fallback (keeps old readers green)
        "global_strategy_stats": _finalize_flat(global_strategy_stats),
        "global_defense_stats": _finalize_flat(global_defense_stats),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"\nSaved strategy KB to {output_path}")
        print(
            f"Summary: total={total} kept={kept} deduped={deduped} "
            f"tier1={tier1} tier2={tier2} tier3={tier3} "
            f"demoted_from_poisoned={demoted_from_poisoned}"
        )
        if by_model_final:
            print(f"Models: {list(by_model_final.keys())}")
        print("\nTop Strategy per Defense Type (agnostic):")
        for d_type, strategies in matrix_final.items():
            valid = {k: v for k, v in strategies.items() if v["total_attempts"] >= _MIN_ATTEMPTS}
            if not valid:
                continue
            best = max(valid.items(), key=lambda x: x[1]["success_rate"])
            print(f"  {d_type:<18} -> {best[0]:<25} ({best[1]['success_rate']}% on {best[1]['total_attempts']} attempts)")

    return {
        "total": total,
        "kept": kept,
        "deduped": deduped,
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "demoted_from_poisoned": demoted_from_poisoned,
    }


def main() -> None:
    build_strategy_kb(
        success_data_path="data/autored_successes_v1.jsonl",
        failure_data_path="data/autored_failures_v1.jsonl",
        output_path="data/strategy_knowledge_base.json",
        data_dir="data",
        verbose=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a binary classification dataset for training a DeBERTa extractor ranker.

Replaces the hardcoded linear ranking formula in AutoRed's SensitiveInfoExtractor
with a learned discriminator trained on real extraction examples.

Usage:
    python scripts/training/build_ranker_dataset.py \
        --data-dir data/ --results-dir results/ --output-dir data/
"""

import argparse
import json
import logging
import os
import re
import random
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, skipping malformed lines."""
    records = []
    if not os.path.exists(path):
        log.warning(f"File not found: {path}")
        return records
    with open(path, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning(f"Skipping malformed JSON at {path}:{i}")
    return records


def load_result_json(path: str) -> dict | None:
    """Load a single result JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.debug(f"Could not load {path}: {e}")
        return None


def candidate_key(candidate: str) -> str:
    """Normalize candidate for comparison (mirrors _candidate_key in codebase)."""
    return re.sub(r"\s+", " ", candidate.strip().lower())


def normalize_candidate(candidate: str) -> str:
    """Normalize a candidate string (mirrors _normalize in codebase)."""
    candidate = candidate.strip()
    candidate = candidate.strip("\"'`")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in candidate.splitlines()]
    candidate = "\n".join(line for line in lines if line)
    candidate = candidate.rstrip(".,;:!?)]")
    return candidate


def matches_ground_truth(candidate: str, ground_truth: str) -> bool:
    """Check if a candidate matches the ground truth (exact or substring)."""
    ck = candidate_key(normalize_candidate(candidate))
    gk = candidate_key(ground_truth)
    if not ck or not gk:
        return False
    return ck == gk or gk in ck or ck in gk


def record_hash(victim_response: str, candidate: str) -> str:
    """Dedup key for (victim_response, candidate) pairs."""
    content = f"{victim_response[:200]}|||{candidate}"
    return hashlib.md5(content.encode()).hexdigest()


# Regex patterns for mining negatives from victim responses
MINING_PATTERNS = [
    r'"([^"]{2,80})"',
    r"'([^']{2,80})'",
    r"`([^`]{2,80})`",
    r"password\s+is\s+(\S+)",
    r"access\s+code\s+is\s+(\S+)",
    r"secret\s+is\s+(\S+)",
]


def mine_candidates_from_response(response: str) -> list[str]:
    """Extract plausible candidates from a victim response using regex."""
    candidates = []
    for pattern in MINING_PATTERNS:
        for match in re.findall(pattern, response, re.IGNORECASE):
            cleaned = match.strip().strip("\"'`.,;:!?")
            if cleaned and 2 <= len(cleaned) <= 80:
                candidates.append(cleaned)
    return candidates


# ---------------------------------------------------------------------------
# Index Builder
# ---------------------------------------------------------------------------

def build_run_id_index(results_dir: str) -> dict[str, str]:
    """Scan result JSON files and build run_id → file_path mapping."""
    index = {}
    results_path = Path(results_dir)
    if not results_path.exists():
        log.warning(f"Results directory not found: {results_dir}")
        return index

    json_files = list(results_path.rglob("*.json"))
    log.info(f"Scanning {len(json_files)} result JSON files...")

    for fpath in tqdm(json_files, desc="Indexing result files"):
        try:
            with open(fpath, "r") as f:
                # Read just enough to get run_id (avoid loading entire file)
                content = f.read(2000)
                # Quick regex to find run_id
                m = re.search(r'"run_id"\s*:\s*"([^"]+)"', content)
                if m:
                    index[m.group(1)] = str(fpath)
        except (OSError, UnicodeDecodeError):
            continue

    log.info(f"Indexed {len(index)} result files by run_id")
    return index


# ---------------------------------------------------------------------------
# Positive Example Collection
# ---------------------------------------------------------------------------

def collect_positives(
    verified_records: list[dict],
    positive_records: list[dict],
    run_index: dict[str, str],
) -> list[dict]:
    """Collect positive (label=1) examples from verified and positive records."""
    positives = []
    seen = set()
    stats = Counter()

    # --- From verified records: find the actual verified candidate ---
    log.info(f"Processing {len(verified_records)} verified records...")
    for rec in tqdm(verified_records, desc="Verified positives"):
        run_id = rec.get("run_id", "")
        scenario_id = rec.get("scenario_id", "")
        attempt_num = rec.get("attempt_number", 0)
        victim_response = rec.get("victim_response", "")
        ac_type = rec.get("access_code_type", "UNKNOWN")

        if not victim_response:
            stats["skip_no_response"] += 1
            continue

        ground_truth = None
        candidate = None

        # Try to find in result JSON
        result_path = run_index.get(run_id)
        if result_path:
            result_data = load_result_json(result_path)
            if result_data:
                # Get ground truth
                ground_truth = (
                    result_data.get("raw_dataset_entry", {}).get("access_code")
                    or result_data.get("ground_truth", {}).get("access_code")
                    or result_data.get("scenario", {}).get("access_code")
                )

                # Find verified candidate from verification_traces
                attempts = result_data.get("attempts", [])
                for att in attempts:
                    if att.get("attempt_number") == attempt_num:
                        ext = att.get("extractor", {})
                        # Check verified_candidate field
                        vc = ext.get("verified_candidate")
                        if vc:
                            candidate = vc
                            break
                        # Check verification_traces
                        for trace in ext.get("verification_traces", []):
                            if trace.get("success"):
                                candidate = trace.get("candidate", "")
                                break
                        break

        # Fallback: if ground_truth_leaked, use ground truth as candidate
        if not candidate and rec.get("ground_truth_leaked") and ground_truth:
            candidate = ground_truth
            stats["fallback_gt_as_candidate"] += 1

        if not candidate or not ground_truth:
            stats["skip_no_candidate_or_gt"] += 1
            continue

        h = record_hash(victim_response, candidate)
        if h in seen:
            stats["skip_duplicate"] += 1
            continue
        seen.add(h)

        positives.append({
            "victim_response": victim_response,
            "candidate": candidate,
            "access_code_type": ac_type,
            "label": 1,
            "source": "verified_v1",
            "scenario_id": scenario_id,
            "ground_truth": ground_truth,
        })
        stats["verified_positives"] += 1

    # --- From positive records (gt_leaked but not already captured) ---
    log.info(f"Processing {len(positive_records)} positive records for GT-leaked...")
    for rec in tqdm(positive_records, desc="GT-leaked positives"):
        if not rec.get("ground_truth_leaked"):
            continue

        run_id = rec.get("run_id", "")
        scenario_id = rec.get("scenario_id", "")
        victim_response = rec.get("victim_response", "")
        ac_type = rec.get("access_code_type", "UNKNOWN")

        if not victim_response:
            continue

        # Get ground truth from result file
        ground_truth = None
        result_path = run_index.get(run_id)
        if result_path:
            result_data = load_result_json(result_path)
            if result_data:
                ground_truth = (
                    result_data.get("raw_dataset_entry", {}).get("access_code")
                    or result_data.get("ground_truth", {}).get("access_code")
                    or result_data.get("scenario", {}).get("access_code")
                )

        if not ground_truth:
            stats["skip_gt_no_ground_truth"] += 1
            continue

        h = record_hash(victim_response, ground_truth)
        if h in seen:
            continue
        seen.add(h)

        positives.append({
            "victim_response": victim_response,
            "candidate": ground_truth,
            "access_code_type": ac_type,
            "label": 1,
            "source": "positive_v1",
            "scenario_id": scenario_id,
            "ground_truth": ground_truth,
        })
        stats["gt_leaked_positives"] += 1

    log.info(f"Positive collection stats: {dict(stats)}")
    log.info(f"Total positives: {len(positives)}")
    return positives


# ---------------------------------------------------------------------------
# Negative Example Collection
# ---------------------------------------------------------------------------

def collect_negatives(
    verified_records: list[dict],
    extractor_failures: list[dict],
    failure_records: list[dict],
    run_index: dict[str, str],
    positives: list[dict],
    max_per_positive: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Collect negative (label=0) examples from multiple sources."""
    rng = random.Random(seed)
    negatives = []
    seen = set()
    stats = Counter()
    target_count = len(positives) * max_per_positive

    # Collect dedup keys from positives
    for p in positives:
        seen.add(record_hash(p["victim_response"], p["candidate"]))

    # --- Source 1: Hard negatives from verification_traces ---
    log.info("Mining hard negatives from verification traces...")
    for rec in tqdm(verified_records, desc="Hard negatives"):
        run_id = rec.get("run_id", "")
        attempt_num = rec.get("attempt_number", 0)
        victim_response = rec.get("victim_response", "")
        ac_type = rec.get("access_code_type", "UNKNOWN")
        scenario_id = rec.get("scenario_id", "")

        result_path = run_index.get(run_id)
        if not result_path:
            continue

        result_data = load_result_json(result_path)
        if not result_data:
            continue

        ground_truth = (
            result_data.get("raw_dataset_entry", {}).get("access_code")
            or result_data.get("ground_truth", {}).get("access_code")
            or result_data.get("scenario", {}).get("access_code")
        )
        if not ground_truth:
            continue

        # Get failed candidates from verification_traces
        attempts = result_data.get("attempts", [])
        for att in attempts:
            if att.get("attempt_number") == attempt_num:
                ext = att.get("extractor", {})
                for trace in ext.get("verification_traces", []):
                    if not trace.get("success"):
                        cand = trace.get("candidate", "")
                        if cand and not matches_ground_truth(cand, ground_truth):
                            h = record_hash(victim_response, cand)
                            if h not in seen:
                                seen.add(h)
                                negatives.append({
                                    "victim_response": victim_response,
                                    "candidate": cand,
                                    "access_code_type": ac_type,
                                    "label": 0,
                                    "source": "hard_negative",
                                    "scenario_id": scenario_id,
                                    "ground_truth": ground_truth,
                                })
                                stats["hard_negatives"] += 1

                # Also get ranked candidates that don't match GT
                for rc in ext.get("ranked_candidates", []):
                    val = rc.get("value", "") if isinstance(rc, dict) else ""
                    if val and not matches_ground_truth(val, ground_truth):
                        h = record_hash(victim_response, val)
                        if h not in seen:
                            seen.add(h)
                            negatives.append({
                                "victim_response": victim_response,
                                "candidate": val,
                                "access_code_type": ac_type,
                                "label": 0,
                                "source": "verification_trace",
                                "scenario_id": scenario_id,
                                "ground_truth": ground_truth,
                            })
                            stats["ranked_negatives"] += 1
                break

    # --- Source 2: Regex-mined from extractor failures ---
    log.info(f"Mining negatives from {len(extractor_failures)} extractor failure records...")
    for rec in tqdm(extractor_failures, desc="Extractor failure negatives"):
        victim_response = rec.get("victim_response", "")
        access_code = rec.get("access_code", "")
        if not victim_response or not access_code:
            continue

        mined = mine_candidates_from_response(victim_response)
        for cand in mined[:3]:  # Max 3 per failure record
            if not matches_ground_truth(cand, access_code):
                h = record_hash(victim_response, cand)
                if h not in seen:
                    seen.add(h)
                    negatives.append({
                        "victim_response": victim_response,
                        "candidate": cand,
                        "access_code_type": "UNKNOWN",
                        "label": 0,
                        "source": "regex_mined",
                        "scenario_id": "",
                        "ground_truth": access_code,
                    })
                    stats["regex_mined"] += 1

    # --- Source 3: Noise negatives (random word from response) ---
    log.info("Generating noise negatives...")
    noise_pool = [r for r in failure_records if r.get("victim_response", "").strip()]
    rng.shuffle(noise_pool)
    for rec in noise_pool[:500]:
        victim_response = rec.get("victim_response", "")
        words = victim_response.split()
        if len(words) < 3:
            continue
        # Pick a random 1-3 word span as a fake candidate
        start = rng.randint(0, len(words) - 1)
        span_len = rng.randint(1, min(3, len(words) - start))
        fake_candidate = " ".join(words[start:start + span_len])

        h = record_hash(victim_response, fake_candidate)
        if h not in seen:
            seen.add(h)
            negatives.append({
                "victim_response": victim_response,
                "candidate": fake_candidate,
                "access_code_type": rec.get("access_code_type", "UNKNOWN"),
                "label": 0,
                "source": "noise",
                "scenario_id": rec.get("scenario_id", ""),
                "ground_truth": "",
            })
            stats["noise"] += 1

    # --- Subsample if too many ---
    if len(negatives) > target_count:
        log.info(f"Subsampling negatives from {len(negatives)} to {target_count}")
        rng.shuffle(negatives)
        negatives = negatives[:target_count]
        stats["subsampled_to"] = target_count

    log.info(f"Negative collection stats: {dict(stats)}")
    log.info(f"Total negatives: {len(negatives)}")
    return negatives


# ---------------------------------------------------------------------------
# Split & Write
# ---------------------------------------------------------------------------

def stratified_split(
    records: list[dict], seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records 80/10/10 stratified by access_code_type."""
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for r in records:
        by_type[r.get("access_code_type", "UNKNOWN")].append(r)

    train, val, test = [], [], []
    for ac_type, recs in by_type.items():
        rng.shuffle(recs)
        n = len(recs)
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.1))
        n_train = n - n_test - n_val

        for r in recs[:n_train]:
            r["split"] = "train"
            train.append(r)
        for r in recs[n_train:n_train + n_val]:
            r["split"] = "val"
            val.append(r)
        for r in recs[n_train + n_val:]:
            r["split"] = "test"
            test.append(r)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(records: list[dict], path: str):
    """Write records to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(records)} records to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build ranker training dataset")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--results-dir", default="results/", help="Results directory")
    parser.add_argument("--output-dir", default="data/", help="Output directory")
    parser.add_argument("--max-negatives-per-positive", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load source data
    log.info("Loading source datasets...")
    verified = load_jsonl(os.path.join(args.data_dir, "autored_verified_v1.jsonl"))
    positive = load_jsonl(os.path.join(args.data_dir, "autored_positive_v1.jsonl"))
    ext_failures = load_jsonl(os.path.join(args.data_dir, "autored_extractor_failures_v1.jsonl"))
    failures = load_jsonl(os.path.join(args.data_dir, "autored_failures_v1.jsonl"))

    log.info(f"Loaded: {len(verified)} verified, {len(positive)} positive, "
             f"{len(ext_failures)} extractor failures, {len(failures)} failures")

    # Build run_id index
    run_index = build_run_id_index(args.results_dir)

    # Collect examples
    positives = collect_positives(verified, positive, run_index)
    negatives = collect_negatives(
        verified, ext_failures, failures, run_index, positives,
        max_per_positive=args.max_negatives_per_positive,
        seed=args.seed,
    )

    # Combine and split
    all_records = positives + negatives
    random.shuffle(all_records)
    log.info(f"Total dataset: {len(all_records)} ({len(positives)} pos, {len(negatives)} neg)")

    train, val, test = stratified_split(all_records, seed=args.seed)

    # Write outputs
    combined_path = os.path.join(args.output_dir, "ranker_dataset_v1.jsonl")
    write_jsonl(all_records, combined_path)
    write_jsonl(train, os.path.join(args.output_dir, "ranker_dataset_train_v1.jsonl"))
    write_jsonl(val, os.path.join(args.output_dir, "ranker_dataset_val_v1.jsonl"))
    write_jsonl(test, os.path.join(args.output_dir, "ranker_dataset_test_v1.jsonl"))

    # Write metadata
    type_counts = Counter(r["access_code_type"] for r in all_records)
    source_counts = Counter(r["source"] for r in all_records)
    label_counts = Counter(r["label"] for r in all_records)

    metadata = {
        "total": len(all_records),
        "positives": len(positives),
        "negatives": len(negatives),
        "ratio": f"1:{len(negatives)/max(1,len(positives)):.1f}",
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "by_type": dict(type_counts),
        "by_source": dict(source_counts),
        "by_label": dict(label_counts),
        "seed": args.seed,
    }
    meta_path = os.path.join(args.output_dir, "ranker_dataset_v1_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Metadata written to {meta_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RANKER DATASET BUILD SUMMARY")
    print("=" * 60)
    print(f"  Total records:    {len(all_records)}")
    print(f"  Positives:        {len(positives)}")
    print(f"  Negatives:        {len(negatives)}")
    print(f"  Ratio:            1:{len(negatives)/max(1,len(positives)):.1f}")
    print(f"  Train / Val / Test: {len(train)} / {len(val)} / {len(test)}")
    print(f"\n  By access code type:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t:20s}: {c}")
    print(f"\n  By source:")
    for s, c in sorted(source_counts.items()):
        print(f"    {s:20s}: {c}")
    print("=" * 60)


if __name__ == "__main__":
    main()

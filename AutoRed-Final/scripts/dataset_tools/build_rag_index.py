"""Build the RAG (FAISS) index of successful attack exemplars.

Exports ``build_rag_index(...)`` (callable from ``kb_updater``) plus a thin
``main()`` CLI wrapper. Key design points vs the legacy version:

* **Tier-1 only (Constraint 3):** embeds ONLY records with a Tier-1 per-attempt
  signal (``verification_success OR ground_truth_found OR access_granted``). The
  poisoned run-level ``success`` boolean is IGNORED — so the 29,958 historical
  records with no per-attempt signal are excluded from the index, not embedded
  as bogus exemplars. This is the self-correction mechanism for RAG.
* **Dedup (Constraint 2):** skips records sharing a ``dedup_key``.
* **Model dimension (Constraint 1):** stores ``victim_model`` in each metadata
  entry so the retriever can filter same-model first, backfill cross-model.
* **Diversity caps (Constraint 5):** per-scenario K=3 and per-strategy M=5
  (within a defense_type) applied at embed time so the index itself is bounded.
* **Staleness (Constraint 4):** stores ``timestamp`` + ``git_commit`` in
  metadata; downweight (not drop) at retrieve time.
* Keeps ``all-MiniLM-L6-v2`` + ``faiss.IndexFlatIP`` + L2-normalize (unchanged).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Diversity caps (Constraint 5): bound the exemplar pool.
_MAX_PER_SCENARIO = 3  # K
_MAX_PER_STRATEGY = 5  # M


def _normalize_attack(attack: str) -> str:
    return re.sub(r"\s+", " ", attack.strip().lower())


def _dedup_key(scenario_id: str, strategy: str, attack: str) -> str:
    h = hashlib.sha256()
    h.update(f"{scenario_id}|{strategy}|{_normalize_attack(attack)}".encode("utf-8"))
    return h.hexdigest()[:16]


def _is_tier1(record: dict) -> bool:
    """Tier-1 = a demonstrably-working attack (RAG-eligible).

    Reads per-attempt flags; IGNORES the poisoned run-level ``success`` boolean.
    """
    verification_success = bool(record.get("verification_success", False))
    ground_truth_found = bool(record.get("ground_truth_found", False)) or bool(
        record.get("ground_truth_leaked", False)
    )
    access_granted = bool(record.get("access_granted", False))
    return verification_success or ground_truth_found or access_granted


def _load_defense_map(data_dir: str, verbose: bool = True) -> dict[str, dict]:
    """Map scenario_id -> defense details. Reads Part1+Part2 + benchmark_v1."""
    data_dir_path = Path(data_dir)
    defense_map: dict[str, dict] = {}

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
            for line in tqdm(f, desc=f"Loading {part.name}", disable=not verbose):
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                did = str(d.get("defense_id", ""))
                if did:
                    defense_map[did] = {
                        "opening_defense": d.get("opening_defense", ""),
                        "closing_defense": d.get("closing_defense", ""),
                        "defense_type": d.get("primary_type", "unknown"),
                        "access_code_type": d.get("access_code_type", "UNKNOWN"),
                    }

    # benchmark_v1 fallback for bench_* ids.
    bench = data_dir_path / "benchmark_v1.jsonl"
    if bench.exists():
        with open(bench, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = str(d.get("scenario_id", "")).replace("bench_", "")
                if sid not in defense_map:
                    defense_map[sid] = {
                        "opening_defense": d.get("opening_defense", ""),
                        "closing_defense": d.get("closing_defense", ""),
                        "defense_type": "unknown",
                        "access_code_type": d.get("access_code_type", "UNKNOWN"),
                    }
    return defense_map


def build_rag_index(
    successes_path: str,
    output_dir: str,
    data_dir: str = "data",
    verbose: bool = True,
) -> dict:
    """Build the FAISS RAG index of Tier-1 successful attack exemplars.

    Returns a summary dict: {total, kept, deduped, tier1, filtered_out,
    demoted_from_poisoned, per_scenario_capped, per_strategy_capped}.
    """
    defense_map = _load_defense_map(data_dir, verbose=verbose)
    if verbose:
        print(f"Loaded {len(defense_map)} mapped defenses.")

    seen_keys: set[str] = set()
    # Candidate pool before diversity caps.
    candidates: list[dict] = []
    total = 0
    deduped = 0
    tier1 = 0
    filtered_out = 0  # not Tier-1
    demoted_from_poisoned = 0  # success=True but no Tier-1 signal

    if not os.path.exists(successes_path):
        if verbose:
            print(f"Warning: {successes_path} not found.")
        return {
            "total": 0, "kept": 0, "deduped": 0, "tier1": 0,
            "filtered_out": 0, "demoted_from_poisoned": 0,
            "per_scenario_capped": 0, "per_strategy_capped": 0,
        }

    with open(successes_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning successes", disable=not verbose):
            if not line.strip():
                continue
            total += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            scenario_id = str(d.get("scenario_id", "")).replace("bench_", "")
            if scenario_id not in defense_map:
                continue  # no defense text to embed

            strategy = str(d.get("strategy", "unknown"))
            attack = str(d.get("attack", ""))

            # Dedup (Constraint 2).
            dk = d.get("dedup_key") or _dedup_key(scenario_id, strategy, attack)
            if dk in seen_keys:
                deduped += 1
                continue
            seen_keys.add(dk)

            # Tier-1 filter (Constraint 3) — IGNORES poisoned `success` boolean.
            if _is_tier1(d):
                tier1 += 1
            else:
                filtered_out += 1
                if bool(d.get("success", False)):
                    demoted_from_poisoned += 1
                continue

            def_info = defense_map[scenario_id]
            defense_text = f"{def_info['opening_defense']}\n{def_info['closing_defense']}".strip()
            candidates.append({
                "scenario_id": scenario_id,
                "defense_text": defense_text,
                "defense_type": def_info["defense_type"],
                "access_code_type": def_info["access_code_type"],
                "strategy": strategy,
                "attack": attack,
                "victim_model": str(d.get("victim_model", "unknown")),
                "attempt_number": d.get("attempt_number", 1),
                "verified": bool(d.get("verification_success", False)),
                "access_granted": bool(d.get("access_granted", False)),
                "timestamp": str(d.get("timestamp", "")),
                "git_commit": str(d.get("git_commit", "unknown")),
                "dedup_key": dk,
            })

    # Diversity caps (Constraint 5): per-scenario K, per-strategy M (per defense_type).
    per_scenario_count: dict[str, int] = defaultdict(int)
    per_strategy_count: dict[str, int] = defaultdict(int)  # key = defense_type|strategy
    kept_records: list[dict] = []
    per_scenario_capped = 0
    per_strategy_capped = 0

    # Sort: most recent (by attempt_number descending, then timestamp) first so we
    # keep the strongest/most-recent exemplars when capping.
    candidates.sort(
        key=lambda r: (r["attempt_number"], r["timestamp"]),
        reverse=True,
    )

    for rec in candidates:
        sid = rec["scenario_id"]
        skey = f"{rec['defense_type']}|{rec['strategy']}"
        if per_scenario_count[sid] >= _MAX_PER_SCENARIO:
            per_scenario_capped += 1
            continue
        if per_strategy_count[skey] >= _MAX_PER_STRATEGY:
            per_strategy_capped += 1
            continue
        per_scenario_count[sid] += 1
        per_strategy_count[skey] += 1
        kept_records.append(rec)

    if not kept_records:
        if verbose:
            print("No Tier-1 successes found. Nothing to embed.")
        # Still write an empty index so the retriever's load guard stays consistent.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        faiss.write_index(index, str(Path(output_dir) / "success_defenses.index"))
        with open(Path(output_dir) / "success_metadata.json", "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return {
            "total": total, "kept": 0, "deduped": deduped, "tier1": tier1,
            "filtered_out": filtered_out, "demoted_from_poisoned": demoted_from_poisoned,
            "per_scenario_capped": 0, "per_strategy_capped": 0,
        }

    if verbose:
        print(f"Embedding {len(kept_records)} Tier-1 exemplars...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [r["defense_text"] for r in kept_records]
    embeddings = model.encode(texts, show_progress_bar=verbose, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    if verbose:
        print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP + L2-norm = cosine
    index.add(embeddings)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(Path(output_dir) / "success_defenses.index"))
    with open(Path(output_dir) / "success_metadata.json", "w", encoding="utf-8") as f:
        json.dump(kept_records, f, indent=2)

    if verbose:
        print(f"\nSaved RAG index to {output_dir}/")
        print(
            f"Summary: total={total} kept={len(kept_records)} deduped={deduped} "
            f"tier1={tier1} filtered_out={filtered_out} "
            f"demoted_from_poisoned={demoted_from_poisoned} "
            f"per_scenario_capped={per_scenario_capped} "
            f"per_strategy_capped={per_strategy_capped}"
        )

    return {
        "total": total,
        "kept": len(kept_records),
        "deduped": deduped,
        "tier1": tier1,
        "filtered_out": filtered_out,
        "demoted_from_poisoned": demoted_from_poisoned,
        "per_scenario_capped": per_scenario_capped,
        "per_strategy_capped": per_strategy_capped,
    }


def main() -> None:
    build_rag_index(
        successes_path="data/autored_successes_v1.jsonl",
        output_dir="data/rag",
        data_dir="data",
        verbose=True,
    )


if __name__ == "__main__":
    main()

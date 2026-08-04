#!/usr/bin/env python3
"""Build a filtered Tensor Trust dataset with access codes <= 30 characters.

This script:
1. Reads the full raw_dump_defenses.jsonl.bz2 (118K scenarios)
2. Filters to access codes <= 30 chars
3. Adds access_code_type classification (TOKEN/PHRASE/SENTENCE/MULTILINE)
4. Saves as both .jsonl.bz2 (for production) and .jsonl (for inspection)
"""

import pandas as pd
import json
import sys
import os

INPUT_PATH = "experiment/raw_dump_defenses.jsonl.bz2"
OUTPUT_BZ2 = "experiment/defenses_ac30.jsonl.bz2"
OUTPUT_JSONL = "experiment/defenses_ac30.jsonl"
MAX_AC_LEN = 30


def classify_access_code(ac: str) -> str:
    """Classify access code type based on structure."""
    ac = str(ac).strip()
    if "\n" in ac:
        return "MULTILINE"
    words = ac.split()
    wc = len(words)
    cc = len(ac)
    if wc == 1:
        return "TOKEN"
    elif wc <= 5 and cc <= 40:
        return "PHRASE"
    else:
        return "SENTENCE"


def main():
    print(f"Loading full dataset from {INPUT_PATH}...")
    df = pd.read_json(INPUT_PATH, lines=True, compression="bz2")
    print(f"  Total rows: {len(df)}")

    # Drop rows without access_code
    df = df.dropna(subset=["access_code"])
    print(f"  With access_code: {len(df)}")

    # Filter by access code length
    df["ac_len"] = df["access_code"].astype(str).str.len()
    filtered = df[df["ac_len"] <= MAX_AC_LEN].copy()
    print(f"  After filtering (ac_len <= {MAX_AC_LEN}): {len(filtered)}")

    # Classify access code types
    filtered["access_code_type"] = filtered["access_code"].apply(classify_access_code)

    # Drop helper column
    filtered = filtered.drop(columns=["ac_len"])

    # Keep only needed columns (preserve defense_id)
    keep_cols = ["defense_id", "opening_defense", "closing_defense", "access_code", "access_code_type"]
    extra = [c for c in ["llm_choice", "defender_id_anonymized"] if c in filtered.columns]
    filtered = filtered[keep_cols + extra]

    # Print summary
    print(f"\n{'='*60}")
    print(f"FILTERED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total scenarios:     {len(filtered)}")
    print(f"  Access code length:  <= {MAX_AC_LEN} chars")
    print(f"\n  By access_code_type:")
    for t, count in filtered["access_code_type"].value_counts().items():
        print(f"    {t:<12}: {count:>6} ({100*count/len(filtered):.1f}%)")

    ac_lens = filtered["access_code"].astype(str).str.len()
    print(f"\n  Access code length stats:")
    print(f"    min:    {ac_lens.min()}")
    print(f"    max:    {ac_lens.max()}")
    print(f"    mean:   {ac_lens.mean():.1f}")
    print(f"    median: {ac_lens.median():.1f}")
    print(f"{'='*60}")

    # Save compressed (for production / HPC)
    filtered.to_json(OUTPUT_BZ2, orient="records", lines=True, compression="bz2")
    size_mb = os.path.getsize(OUTPUT_BZ2) / (1024 * 1024)
    print(f"\n  Saved: {OUTPUT_BZ2} ({size_mb:.1f} MB)")

    # Save uncompressed (for quick inspection)
    filtered.to_json(OUTPUT_JSONL, orient="records", lines=True)
    size_mb2 = os.path.getsize(OUTPUT_JSONL) / (1024 * 1024)
    print(f"  Saved: {OUTPUT_JSONL} ({size_mb2:.1f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()

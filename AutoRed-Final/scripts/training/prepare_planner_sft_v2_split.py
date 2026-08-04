#!/usr/bin/env python3
"""
Prepare train/val splits for planner_sft_dataset_v2.

Input:
  data/planner_sft_dataset_v2.jsonl

Outputs:
  scripts/training/sft_data/planner_v2_train.jsonl
  scripts/training/sft_data/planner_v2_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "planner_sft_dataset_v2.jsonl"
DEFAULT_TRAIN = ROOT / "scripts" / "training" / "sft_data" / "planner_v2_train.jsonl"
DEFAULT_VAL = ROOT / "scripts" / "training" / "sft_data" / "planner_v2_val.jsonl"


def load_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_strategy(row: Dict) -> str:
    text = row["messages"][1]["content"]
    start = text.find("<strategy>")
    end = text.find("</strategy>")
    if start == -1 or end == -1:
        return "unknown"
    return text[start + len("<strategy>"):end].strip()


def stratified_split(rows: List[Dict], val_ratio: float, seed: int) -> tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    by_strategy: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_strategy[extract_strategy(row)].append(row)

    train: List[Dict] = []
    val: List[Dict] = []
    for strategy, items in by_strategy.items():
        rng.shuffle(items)
        val_count = max(1, round(len(items) * val_ratio)) if len(items) > 1 else 0
        val.extend(items[:val_count])
        train.extend(items[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_rows(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Planner SFT v2 train/val split")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-output", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    train, val = stratified_split(rows, args.val_ratio, args.seed)
    write_rows(args.train_output, train)
    write_rows(args.val_output, val)
    print(
        f"[SPLIT] input={len(rows)} "
        f"train={len(train)} val={len(val)} "
        f"train_out={args.train_output} val_out={args.val_output}"
    )


if __name__ == "__main__":
    main()

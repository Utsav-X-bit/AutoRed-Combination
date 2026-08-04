#!/usr/bin/env python3
"""
Build a small contract-repair SFT dataset for planner tag fidelity.

This dataset does not change the task. It reuses the existing planner v2
examples, rewrites assistant outputs into canonical XML, and upsamples rows
where the model previously showed the most schema drift risk:
  - attempt > 1
  - non-none failure_reason
  - non-TOKEN expected access types
  - roleplay / translation defenses
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiment.planner_contract import canonicalize_plan, parse_plan_text, render_plan_xml

DEFAULT_TRAIN_IN = ROOT / "scripts" / "training" / "sft_data" / "planner_v2_train.jsonl"
DEFAULT_VAL_IN = ROOT / "scripts" / "training" / "sft_data" / "planner_v2_val.jsonl"
DEFAULT_TRAIN_OUT = ROOT / "scripts" / "training" / "sft_data" / "planner_contract_repair_train.jsonl"
DEFAULT_VAL_OUT = ROOT / "scripts" / "training" / "sft_data" / "planner_contract_repair_val.jsonl"


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_attempt(user_text: str) -> int:
    raw = extract_tag(user_text, "attempt")
    try:
        return int(raw) if raw else 1
    except ValueError:
        return 1


def extract_defense_type(user_text: str) -> str:
    return extract_tag(user_text, "defense_type") or "unknown"


def canonicalized_row(row: dict) -> dict:
    fixed = dict(row)
    messages = [dict(m) for m in row["messages"]]
    plan = parse_plan_text(messages[1]["content"])
    messages[1]["content"] = render_plan_xml(canonicalize_plan(plan, messages[1]["content"]))
    fixed["messages"] = messages
    return fixed


def repeat_weight(row: dict) -> int:
    user_text = row["messages"][0]["content"]
    asst_text = row["messages"][1]["content"]
    attempt = extract_attempt(user_text)
    defense_type = extract_defense_type(user_text)
    plan = canonicalize_plan(parse_plan_text(asst_text), asst_text)

    weight = 1
    if attempt > 1:
        weight += 3
    if plan["failure_reason"] != "none":
        weight += 3
    if plan["expected_access_type"] != "TOKEN":
        weight += 1
    if defense_type in {"roleplay", "translation"}:
        weight += 1
    return min(weight, 8)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_repair_split(rows: list[dict], *, seed: int, sample_cap: int | None) -> list[dict]:
    rng = random.Random(seed)
    built: list[dict] = []
    for row in rows:
        repaired = canonicalized_row(row)
        for _ in range(repeat_weight(repaired)):
            built.append(repaired)
    rng.shuffle(built)
    if sample_cap is not None and len(built) > sample_cap:
        built = built[:sample_cap]
    return built


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build planner contract repair dataset")
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_IN)
    parser.add_argument("--val-input", type=Path, default=DEFAULT_VAL_IN)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--val-output", type=Path, default=DEFAULT_VAL_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-cap", type=int, default=12000)
    parser.add_argument("--val-cap", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(args.train_input)
    val_rows = load_jsonl(args.val_input)

    repaired_train = build_repair_split(train_rows, seed=args.seed, sample_cap=args.train_cap)
    repaired_val = build_repair_split(val_rows, seed=args.seed + 1, sample_cap=args.val_cap)

    save_jsonl(args.train_output, repaired_train)
    save_jsonl(args.val_output, repaired_val)

    print(
        f"[REPAIR] train_in={len(train_rows)} train_out={len(repaired_train)} "
        f"val_in={len(val_rows)} val_out={len(repaired_val)} "
        f"train_file={args.train_output} val_file={args.val_output}"
    )


if __name__ == "__main__":
    main()

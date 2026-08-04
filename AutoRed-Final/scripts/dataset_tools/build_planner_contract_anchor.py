#!/usr/bin/env python3
"""
Build a compact contract-anchor dataset for planner tag fidelity.

This dataset is intentionally redundant. It takes the canonical planner v2
examples and prepends a short contract reminder to the user prompt so the model
sees the exact output tag names repeatedly during a short cleanup pass.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiment.planner_contract import canonicalize_plan, parse_plan_text, render_plan_xml


DEFAULT_INPUTS = [
    ROOT / "scripts" / "training" / "sft_data" / "planner_v2_train.jsonl",
    ROOT / "scripts" / "training" / "sft_data" / "planner_v2_val.jsonl",
]
DEFAULT_TRAIN_OUT = ROOT / "scripts" / "training" / "sft_data" / "planner_contract_anchor_train.jsonl"
DEFAULT_VAL_OUT = ROOT / "scripts" / "training" / "sft_data" / "planner_contract_anchor_val.jsonl"

ANCHOR_PREFIX = (
    "Planner contract reminder: output exactly `<expected_access_type>`, not "
    "`<expected_access_code_type>`. If no failure reason is known, use "
    "`<failure_reason>none</failure_reason>`. Keep the rest of the XML contract exact.\n\n"
)


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


def anchor_row(row: dict) -> dict:
    fixed = dict(row)
    messages = [dict(m) for m in row["messages"]]
    messages[0]["content"] = ANCHOR_PREFIX + messages[0]["content"]
    plan = canonicalize_plan(parse_plan_text(messages[1]["content"]), messages[1]["content"])
    messages[1]["content"] = render_plan_xml(plan)
    fixed["messages"] = messages
    return fixed


def sample_repeats(row: dict) -> int:
    # Heavy repetition is intentional: this is a small cleanup pass.
    user_text = row["messages"][0]["content"]
    repeats = 1
    if "<attempt>1</attempt>" in user_text:
        repeats += 1
    if "<history>\n(none)\n</history>" in user_text:
        repeats += 1
    return repeats


def build(rows: list[dict], cap: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    built: list[dict] = []
    for row in rows:
        anchored = anchor_row(row)
        for _ in range(sample_repeats(anchored)):
            built.append(anchored)
    rng.shuffle(built)
    if len(built) > cap:
        built = built[:cap]
    return built


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build planner contract anchor dataset")
    parser.add_argument("--inputs", type=Path, nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--val-output", type=Path, default=DEFAULT_VAL_OUT)
    parser.add_argument("--train-cap", type=int, default=8000)
    parser.add_argument("--val-cap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined: list[dict] = []
    for path in args.inputs:
        combined.extend(load_jsonl(path))

    rng = random.Random(args.seed)
    rng.shuffle(combined)

    split_idx = int(len(combined) * 0.85)
    train_rows = build(combined[:split_idx], cap=args.train_cap, seed=args.seed)
    val_rows = build(combined[split_idx:], cap=args.val_cap, seed=args.seed + 1)

    save_jsonl(args.train_output, train_rows)
    save_jsonl(args.val_output, val_rows)
    print(
        f"[ANCHOR] train_out={len(train_rows)} val_out={len(val_rows)} "
        f"train_file={args.train_output} val_file={args.val_output}"
    )


if __name__ == "__main__":
    main()

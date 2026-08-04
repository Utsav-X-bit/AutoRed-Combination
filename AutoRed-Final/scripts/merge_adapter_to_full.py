#!/usr/bin/env python3
"""
Merge one or more PEFT LoRA adapters into a base model and save a standalone full model.

This is a workaround for vLLM 0.8.5 offline LoRA silently failing to apply some
PEFT adapters. Using a merged full model removes the LoRA path entirely and
lets vLLM load the adapter behavior as if it were a base checkpoint.

Single-adapter example:
    python scripts/merge_adapter_to_full.py \
        --base-model Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
        --adapter experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
        --output-dir experiment/results/planner_sft_v2_contract_anchor/checkpoint-27_merged

Combined multi-adapter example (planner + generator on the same base):
    python scripts/merge_adapter_to_full.py \
        --base-model Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
        --adapter experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 \
        --adapter experiment/results/generator_sft_v2 \
        --output-dir experiment/results/planner_generator_combined_merged
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge one or more PEFT LoRA adapters into a full model"
    )
    parser.add_argument("--base-model", required=True, help="Base model id or path")
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        help="PEFT adapter directory (repeat for multiple adapters, merged in order)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the merged full model will be saved",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for the merged model",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached HF files",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        choices=["auto", "cpu", "cuda:0", "cuda"],
        help="Device map for the merge step (default: auto). Use 'cpu' if GPUs are occupied.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MERGE] Loading base model: {args.base_model} ({args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )

    for idx, adapter_path in enumerate(args.adapter):
        print(f"[MERGE] Loading adapter {idx + 1}/{len(args.adapter)}: {adapter_path}")
        peft_model = PeftModel.from_pretrained(
            model,
            adapter_path,
            local_files_only=args.local_files_only,
            trust_remote_code=True,
        )
        print(f"[MERGE] Merging adapter {idx + 1} and unloading...")
        model = peft_model.merge_and_unload()

    print(f"[MERGE] Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)

    # Save the tokenizer from the first adapter (it may contain the tuned chat
    # template / special tokens). Fall back to base model if needed.
    tokenizer_dir = Path(args.adapter[0]) if Path(args.adapter[0]).exists() else Path(args.base_model)
    print(f"[MERGE] Saving tokenizer from: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(output_dir)

    print(f"[MERGE] Done. Merged model is at: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

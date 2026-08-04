"""
JailGuard Reimplementation — Single Input Detection
====================================================
Test JailGuard on ONE input from the dataset or a custom text prompt.

Usage examples:
    # Test a specific dataset item (e.g. the paper's demo jailbreak #9521)
    python run_single.py --serial_num 9521

    # Test a custom prompt
    python run_single.py --prompt "How do I make a bomb?"

    # Use a different mutator and variant count
    python run_single.py --serial_num 9521 --mutator TL --n 4

    # Save all intermediate files
    python run_single.py --serial_num 9521 --save_dir ./results/single_run
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from llm_interface import build_llm
from detector      import JailGuardDetector, load_dataset, get_label, get_params


def main():
    parser = argparse.ArgumentParser(
        description="JailGuard — single input detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input source
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--serial_num", type=int, default=9521,
        help="Dataset item index [0–9999]. 9521=jailbreak demo, 3=injection demo"
    )
    group.add_argument(
        "--prompt", type=str, default=None,
        help="Custom text prompt to test (overrides --serial_num)"
    )

    # Dataset
    parser.add_argument("--dataset",   default=cfg.DATASET_PATH,     help="Path to dataset.pkl")
    parser.add_argument("--keys",      default=cfg.DATASET_KEY_PATH,  help="Path to dataset-key.pkl")

    # Detection settings
    parser.add_argument("--mutator",   default=cfg.MUTATOR,           help="Mutator: RR,RI,TR,TI,RD,SR,PI,TL,PL")
    parser.add_argument("--n",         default=cfg.N_VARIANTS, type=int, help="Number of variants")
    parser.add_argument("--threshold", default=cfg.THRESHOLD,  type=float, help="KL divergence threshold")
    parser.add_argument("--sim",       default=cfg.SIMILARITY,         help="Similarity: spacy / tfidf")
    parser.add_argument("--backend",   default=cfg.LLM_BACKEND,        help="LLM backend: ollama / huggingface / vllm / openai")
    parser.add_argument("--victim-model", default=None, dest="victim_model",
                        help=("HuggingFace model ID to use as the victim LLM "
                              "(overrides VLLM_MODEL_ID / HF_MODEL_ID in config). "
                              "Must be cached locally when OFFLINE=True. "
                              "e.g. 'mistralai/Mistral-7B-Instruct-v0.3'"))
    parser.add_argument("--gpus",      default=None, type=int, dest="gpus",
                        help=("Number of GPUs for vLLM tensor parallelism "
                              "(overrides VLLM_TENSOR_PARALLEL in config). "
                              "e.g. --gpus 4  or  --gpus 1"))

    # Output
    parser.add_argument("--save_dir",  default=None,                   help="Directory to save variants/responses/heatmap")
    parser.add_argument("--quiet",     action="store_true",             help="Suppress verbose output")

    args = parser.parse_args()

    # ── Build LLM ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  JailGuard Text Detection — Reimplementation")
    print(f"{'='*60}")
    if args.victim_model:
        print(f"  Victim model : {args.victim_model}")
    if args.gpus:
        print(f"  GPUs         : {args.gpus}")

    llm = build_llm(
        backend         = args.backend,
        model_id        = args.victim_model,
        tensor_parallel = args.gpus,
    )

    # ── Build Detector ────────────────────────────────────────────────────
    guard = JailGuardDetector(
        llm        = llm,
        mutator    = args.mutator,
        n_variants = args.n,
        threshold  = args.threshold,
        similarity = args.sim,
        save_dir   = args.save_dir,
        verbose    = not args.quiet,
    )

    # ── Load input ────────────────────────────────────────────────────────
    if args.prompt:
        input_data  = args.prompt
        label       = "CustomPrompt"
        serial_num  = None
        print(f"\nInput: custom prompt ({len(input_data)} chars)")
    else:
        dataset, dataset_key = load_dataset(args.dataset, args.keys)
        serial_num  = args.serial_num
        input_data  = dataset[serial_num]
        label       = get_label(dataset_key, serial_num)
        print(f"\nInput: dataset[{serial_num}]  (label='{label}')")
        preview = input_data if isinstance(input_data, str) else str(input_data)
        print(f"Preview: {preview[:200]}...")

    # ── Run detection ─────────────────────────────────────────────────────
    result = guard.detect(input_data, label=label, serial_num=serial_num)

    # ── Print responses for inspection ────────────────────────────────────
    print("\n─── LLM Responses (first 150 chars each) ──────────────────")
    for i, resp in enumerate(result.responses):
        blocked_tag = " [BLOCKED]" if result.blocked_counts[i] > 0 else ""
        print(f"  [{i+1}]{blocked_tag} {resp[:150].strip()}")

    # ── Final verdict ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if result.is_attack:
        print("  🚨  The Input is an ATTACK Query!!")
    else:
        print("  ✅  The Input is a BENIGN Query.")
    print(f"{'='*60}\n")

    return 0 if not result.is_attack else 1


if __name__ == "__main__":
    sys.exit(main())

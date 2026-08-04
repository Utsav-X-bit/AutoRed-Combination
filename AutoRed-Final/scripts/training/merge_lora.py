#!/usr/bin/env python3
"""
Script to merge a LoRA adapter into the base model.
This creates a standalone model that can be loaded natively by vLLM for maximum generation speed.
"""

import argparse
import os
import shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, required=True, help="Path or name of the base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the LoRA adapter directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the merged model")
    parser.add_argument("--device", type=str, default="auto", help="Device map (default: auto)")
    
    args = parser.parse_args()
    
    print(f"Loading base model: {args.base_model}...")
    # Load base model in bfloat16 to match Llama-3.1 precision without 4-bit quantization,
    # as merging requires standard floating point weights.
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter if os.path.exists(os.path.join(args.adapter, "tokenizer_config.json")) else args.base_model,
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapter from: {args.adapter}...")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    
    print("Merging adapter into base model (this may take a few minutes)...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Merge complete! You can now pass this output directory directly as the generator path to use vLLM.")

if __name__ == "__main__":
    main()

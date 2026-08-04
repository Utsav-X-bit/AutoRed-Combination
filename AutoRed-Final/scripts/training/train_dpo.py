#!/usr/bin/env python3
import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import Dataset

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    set_seed,
)
from trl import DPOTrainer, DPOConfig

def patch_trainer_tokenizer_compat():
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters and "tokenizer" not in trainer_sig.parameters:
        class TokenizerKwargTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                if "tokenizer" in kwargs:
                    kwargs["processing_class"] = kwargs.pop("tokenizer")
                super().__init__(*args, **kwargs)
        import transformers
        transformers.Trainer = TokenizerKwargTrainer
        
def get_model_device_map(device_map_arg):
    if device_map_arg == "auto":
        return "auto"
    if torch.cuda.is_available():
        return {"": torch.cuda.current_device()}
    return "auto"

def disable_trainer_data_parallel(model):
    if hasattr(model, "is_parallelizable"):
        model.is_parallelizable = True
    if hasattr(model, "model_parallel"):
        model.model_parallel = True

def load_dataset_from_jsonl(train_path):
    train_data = []
    with open(train_path) as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line)
                if "prompt" not in entry or "chosen" not in entry or "rejected" not in entry:
                    continue
                train_data.append(entry)
            except json.JSONDecodeError as e:
                pass
    print(f"Loaded {len(train_data)} DPO training samples")
    return Dataset.from_list(train_data)

def main():
    parser = argparse.ArgumentParser(description="DPO training for AutoRed generator")
    parser.add_argument("--model_name", type=str,
                        default="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
                        help="Base model name or path")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to training JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=16,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max sequence length")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter")
    parser.add_argument("--device_map", choices=["single", "auto"], default="single",
                        help="Use one CUDA device by default; 'auto' may shard across GPUs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--run_name", type=str, default="autored_dpo",
                        help="Run name for logging")
    args = parser.parse_args()

    set_seed(args.seed)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    with open(output_path / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    train_dataset = load_dataset_from_jsonl(args.dataset)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    device_map = get_model_device_map(args.device_map)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # We also need a reference model for DPO, but since we use PEFT/LoRA, 
    # DPOTrainer can automatically use the un-adapted base model as reference
    # so we don't need to load a separate reference model.

    model = prepare_model_for_kbit_training(model)

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        fan_in_fan_out=False,
    )

    model = get_peft_model(model, lora_config)
    disable_trainer_data_parallel(model)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}<|{{ message['role'] }}|>\n{{ message['content'] }}</s>\n{% endfor %}"

    print(f"\nSetting up DPOTrainer...")
    patch_trainer_tokenizer_compat()
    
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=3,
        fp16=False,
        bf16=True,
        seed=args.seed,
        report_to="none",
        run_name=args.run_name,
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        beta=args.beta,
    )

    try:
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
    except Exception as e1:
        try:
            trainer = DPOTrainer(
                model=model,
                args=dpo_config,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
            )
        except Exception as e2:
            print(f"Failed to init DPOTrainer. Err1: {e1}, Err2: {e2}")
            return

    print("\nStarting training...")
    trainer.train()

    print("\nTraining complete! Saving model...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()

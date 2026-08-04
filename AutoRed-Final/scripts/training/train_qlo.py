#!/usr/bin/env python3
"""
QLoRA SFT training for AutoRed attack generator.

Trains Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 with 4-bit QLoRA
on verified/positive AutoRed success datasets.

Usage:
    python scripts/training/train_qlo.py \
        --dataset scripts/training/sft_data/variantc_verified_train.jsonl \
        --val_dataset scripts/training/sft_data/variantc_verified_val.jsonl \
        --output_dir experiment/results/qlo_verified_v1 \
        --epochs 10

    # Positive dataset (291 samples)
    python scripts/training/train_qlo.py \
        --dataset scripts/training/sft_data/variantc_positive_train.jsonl \
        --val_dataset scripts/training/sft_data/variantc_positive_val.jsonl \
        --output_dir experiment/results/qlo_positive_v1 \
        --epochs 6
"""

import argparse
import inspect
import json
import os
import sys
import types
from pathlib import Path

import torch
from datasets import Dataset

# PEFT on some transformers builds expects tensor_parallel helpers that may not
# exist. Patch them before importing peft to keep adapter save/load working.
try:
    import transformers
    import transformers.integrations

    if not hasattr(transformers.integrations, "tensor_parallel"):
        tp = types.ModuleType("transformers.integrations.tensor_parallel")
        sys.modules["transformers.integrations.tensor_parallel"] = tp
        transformers.integrations.tensor_parallel = tp
    else:
        tp = transformers.integrations.tensor_parallel

    if not hasattr(tp, "EmbeddingParallel"):
        class DummyEmbeddingParallel:
            pass
        tp.EmbeddingParallel = DummyEmbeddingParallel
except Exception:
    pass

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    set_seed,
)
from trl import SFTTrainer, SFTConfig


def patch_trainer_tokenizer_compat():
    """Handle TRL versions that pass tokenizer= to newer Transformers Trainer.

    Transformers v5 replaced Trainer(tokenizer=...) with
    Trainer(processing_class=...). Some TRL releases still call the old
    keyword internally, which raises:
      Trainer.__init__() got an unexpected keyword argument 'tokenizer'
    """
    trainer_sig = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_sig.parameters or "processing_class" not in trainer_sig.parameters:
        return
    if getattr(Trainer.__init__, "_autored_tokenizer_compat", False):
        return

    original_init = Trainer.__init__

    def compat_init(self, *args, tokenizer=None, processing_class=None, **kwargs):
        if processing_class is None and tokenizer is not None:
            processing_class = tokenizer
        return original_init(
            self,
            *args,
            processing_class=processing_class,
            **kwargs,
        )

    compat_init._autored_tokenizer_compat = True
    Trainer.__init__ = compat_init


def get_model_device_map(device_map_mode: str):
    """Return a device map compatible with QLoRA SFT training.

    For this project, one A100 is enough for 4-bit 8B QLoRA. Avoid
    device_map="auto" by default because it may shard across visible GPUs,
    after which Trainer can still attempt DataParallel from cuda:0.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0 and torch.cuda.is_available():
        return {"": local_rank}
    if device_map_mode == "auto":
        return "auto"
    if torch.cuda.is_available():
        return {"": torch.cuda.current_device()}
    return None


def configure_single_gpu_parallel_flags(model):
    """Prevent Trainer from wrapping quantized PEFT models in DataParallel."""
    for obj in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        if obj is not None:
            obj.is_parallelizable = True
            obj.model_parallel = True


def load_dataset_from_jsonl(train_path, val_path=None):
    """Load train/val datasets from JSONL files."""
    print(f"Loading training data from {train_path}...")

    # Load and verify format
    train_data = []
    with open(train_path) as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line)
                # Verify it has messages format
                if "messages" not in entry:
                    print(f"  Warning: line {i+1} missing 'messages', skipping")
                    continue
                train_data.append(entry)
            except json.JSONDecodeError as e:
                print(f"  Warning: line {i+1} JSON error: {e}")

    print(f"  Loaded {len(train_data)} training samples")

    train_dataset = Dataset.from_list(train_data)

    val_dataset = None
    if val_path and Path(val_path).exists():
        val_data = []
        with open(val_path) as f:
            for i, line in enumerate(f):
                try:
                    entry = json.loads(line)
                    if "messages" in entry:
                        val_data.append(entry)
                except json.JSONDecodeError:
                    pass

        print(f"  Loaded {len(val_data)} validation samples")
        val_dataset = Dataset.from_list(val_data)

    return train_dataset, val_dataset


def format_prompt(entry):
    """Format a conversation entry for the model."""
    messages = entry["messages"]
    # Simple chat template formatting
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append(f"<|user|>\n{content}</s>\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}</s>\n")
    return "".join(parts)


class CurriculumCallback(TrainerCallback):
    def __init__(self, trainer, easy_dataset, medium_dataset, hard_dataset, easy_epochs=2, medium_epochs=4):
        self.trainer = trainer
        self.easy_dataset = easy_dataset
        self.medium_dataset = medium_dataset
        self.hard_dataset = hard_dataset
        self.easy_epochs = easy_epochs
        self.medium_epochs = medium_epochs
        
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        if epoch < self.easy_epochs:
            current_phase = "easy"
            self.trainer.train_dataset = self.easy_dataset
        elif epoch < self.easy_epochs + self.medium_epochs:
            current_phase = "medium"
            self.trainer.train_dataset = self.medium_dataset
        else:
            current_phase = "hard"
            self.trainer.train_dataset = self.hard_dataset
        print(f"\n[CURRICULUM] Epoch {epoch} ended. Active dataset for next epoch: {current_phase} (size: {len(self.trainer.train_dataset)})")


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT training for AutoRed generator")
    parser.add_argument("--model_name", type=str,
                        default="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
                        help="Base model name or path")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to training JSONL file")
    parser.add_argument("--val_dataset", type=str, default=None,
                        help="Path to validation JSONL file")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Optional existing PEFT adapter to continue training from")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max sequence length")
    parser.add_argument("--device_map", choices=["single", "auto"], default="single",
                        help="Use one CUDA device by default; 'auto' may shard across GPUs")
    parser.add_argument("--packing", action="store_true",
                        help="Enable sequence packing in SFTTrainer for better throughput on short samples")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing for faster training if memory allows")
    parser.add_argument("--logging_steps", type=int, default=5,
                        help="Trainer logging interval")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Number of dataloader workers")
    parser.add_argument("--skip_best_model_reload", action="store_true",
                        help="Skip Trainer's best-model reload at the end of training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name (set to empty string to disable)")
    parser.add_argument("--run_name", type=str, default="autored_qlo",
                        help="Run name for logging")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning training mode")
    parser.add_argument("--easy_dataset", type=str, default="scripts/training/sft_data/curriculum_easy_v1.jsonl",
                        help="Path to easy curriculum dataset")
    parser.add_argument("--medium_dataset", type=str, default="scripts/training/sft_data/curriculum_medium_v1.jsonl",
                        help="Path to medium curriculum dataset")
    parser.add_argument("--hard_dataset", type=str, default="scripts/training/sft_data/curriculum_hard_v1.jsonl",
                        help="Path to hard curriculum dataset")
    parser.add_argument("--easy_epochs", type=int, default=2,
                        help="Number of epochs for easy curriculum phase")
    parser.add_argument("--medium_epochs", type=int, default=4,
                        help="Number of epochs for medium curriculum phase")
    args = parser.parse_args()

    if not args.curriculum and not args.dataset:
        parser.error("--dataset is required when not using --curriculum mode.")

    set_seed(args.seed)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_path / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {output_path / 'training_config.json'}")

    # Load dataset
    if args.curriculum:
        print("[CURRICULUM] Loading curriculum datasets...")
        easy_dataset, val_dataset = load_dataset_from_jsonl(args.easy_dataset, args.val_dataset)
        medium_dataset, _ = load_dataset_from_jsonl(args.medium_dataset, None)
        hard_dataset, _ = load_dataset_from_jsonl(args.hard_dataset, None)
        
        # Start training with the easy dataset
        train_dataset = easy_dataset
    else:
        train_dataset, val_dataset = load_dataset_from_jsonl(args.dataset, args.val_dataset)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"\nLoading model: {args.model_name}")
    device_map = get_model_device_map(args.device_map)
    if torch.cuda.is_available():
        print(f"  CUDA devices visible: {torch.cuda.device_count()}")
        print(f"  Current CUDA device: {torch.cuda.current_device()}")
    print(f"  Device map: {device_map}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    print(f"  Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
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

    if args.adapter_path:
        from peft import PeftModel
        print(f"Loading existing adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        configure_single_gpu_parallel_flags(model)
    model.print_trainable_parameters()

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Ensure tokenizer has chat template for SFTTrainer
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}<|{{ message['role'] }}|>\n{{ message['content'] }}</s>\n{% endfor %}"

    # SFT Trainer
    print(f"\nSetting up SFTTrainer...")
    patch_trainer_tokenizer_compat()
    load_best_model = bool(val_dataset) and not args.skip_best_model_reload

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=load_best_model,
        metric_for_best_model="eval_loss" if load_best_model else None,
        fp16=False,
        bf16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.run_name,
        max_seq_length=args.max_length,
        packing=args.packing,
        group_by_length=True,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False if world_size > 1 else None,
    )

    if args.wandb_project:
        sft_config.wandb_project = args.wandb_project

    # Compatibility: try different trl API signatures
    trainer = None
    errors = []

    # Attempt 1: SFTConfig + processing_class (trl >= 0.15)
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            args=sft_config,
        )
        print(f"  Using SFTConfig + processing_class API")
    except TypeError as e:
        errors.append(str(e))

    # Attempt 2: SFTConfig + tokenizer (trl ~0.12-0.14)
    if trainer is None:
        try:
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                args=sft_config,
            )
            print(f"  Using SFTConfig + tokenizer API")
        except TypeError as e:
            errors.append(str(e))

    # Attempt 3: TrainingArguments + tokenizer + max_seq_length (trl < 0.12)
    if trainer is None:
        try:
            from transformers import TrainingArguments
            training_args = TrainingArguments(
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
                eval_strategy="epoch" if val_dataset else "no",
                save_total_limit=3,
                load_best_model_at_end=True if val_dataset else False,
                metric_for_best_model="eval_loss" if val_dataset else None,
                fp16=False,
                bf16=True,
                dataloader_pin_memory=False,
                seed=args.seed,
                report_to="wandb" if args.wandb_project else "none",
                run_name=args.run_name,
            )
            if args.wandb_project:
                training_args.wandb_project = args.wandb_project

            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                args=training_args,
                max_seq_length=args.max_length,
            )
            print(f"  Using TrainingArguments + max_seq_length API")
        except TypeError as e:
            errors.append(str(e))

    if trainer is None:
        print(f"\n  All SFTTrainer attempts failed:")
        for e in errors:
            print(f"    - {e}")
        raise RuntimeError("Could not initialize SFTTrainer with any known API")

    if args.curriculum:
        print("[CURRICULUM] Registering CurriculumCallback...")
        trainer.add_callback(
            CurriculumCallback(
                trainer=trainer,
                easy_dataset=easy_dataset,
                medium_dataset=medium_dataset,
                hard_dataset=hard_dataset,
                easy_epochs=args.easy_epochs,
                medium_epochs=args.medium_epochs,
            )
        )

    # Training
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max length: {args.max_length}")
    print(f"{'='*60}\n")

    train_result = trainer.train()

    is_main_process = getattr(trainer, "is_world_process_zero", lambda: True)()
    if is_main_process:
        print(f"\nSaving model to {args.output_dir}...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        metrics = train_result.metrics
        metrics_dict = {k: round(float(v), 6) for k, v in metrics.items()}
        with open(output_path / "train_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print("Training metrics saved")
        print(f"  Train loss: {metrics_dict.get('train_loss', 'N/A')}")
        print(f"\nDone. Model saved to {args.output_dir}")

    if hasattr(trainer, "accelerator") and trainer.accelerator is not None:
        trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

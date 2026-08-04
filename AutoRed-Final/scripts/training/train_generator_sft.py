import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)

# Monkey-patch Trainer to fix trl vs transformers 5.x incompatibility
if not hasattr(Trainer, "_original_init"):
    Trainer._original_init = Trainer.__init__
    def patched_init(self, *args, **kwargs):
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        elif "tokenizer" in kwargs and "processing_class" in kwargs:
            del kwargs["tokenizer"]
        Trainer._original_init(self, *args, **kwargs)
    Trainer.__init__ = patched_init

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def train():
    # Paths and model
    model_name = "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
    dataset_path = "data/generator_sft_dataset.jsonl"
    output_dir = "experiment/generator_sft_adapter"

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    print(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Formatting function for SFTTrainer
    # The dataset already has "messages" with "user" and "assistant" roles.
    # We apply the chat template.
    def format_chat_template(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        return example
        
    print("Formatting dataset with chat template...")
    dataset = dataset.map(format_chat_template)

    # QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading model {model_name} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Disable cache to save memory
    model.config.use_cache = False
    
    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=50,
        optim="paged_adamw_8bit",
        fp16=False,
        bf16=True, # Llama 3 handles bfloat16 well
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none" # Disable wandb for local/HPC runs
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Training complete!")

if __name__ == "__main__":
    train()

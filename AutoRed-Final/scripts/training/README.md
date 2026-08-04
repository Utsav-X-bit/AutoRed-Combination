# AutoRed QLoRA Training

## Overview

Fine-tunes `Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2` using QLoRA on AutoRed success datasets.

Current status: verified-only Variant C training completed successfully on HPC and produced the first adapter at `experiment/results/qlo_verified_v1`.

## Prerequisites

Install training dependencies:
```bash
pip install -r ../requirements_qlo.txt
```

Or on HPC:
```bash
uv pip install -r ../requirements_qlo.txt
```

## Dataset Variants

| Variant | Input | Output | Best For |
|---------|-------|--------|----------|
| A | Strategy | Attack | Simple baseline |
| B | Defense + Strategy | Attack | Context-aware |
| C | Defense + Response + Strategy | Attack | **Recommended** — matches AutoRed workflow |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantization | NF4 4-bit |
| Batch size | 4 |
| Gradient accumulation | 8 (effective 32) |
| Learning rate | 2e-5 |
| Scheduler | Cosine with 5% warmup |
| Max length | 1024 tokens |
| Device map | `single` by default |

## Compatibility Notes

The training script includes two HPC compatibility fixes:

- TRL/Transformers API mismatch: older TRL can pass `tokenizer=` into newer `transformers.Trainer`, which expects `processing_class=`. `train_qlo.py` patches this at runtime when needed.
- Multi-GPU visibility: `device_map="auto"` can shard the model across visible GPUs and then fail when `Trainer` wraps the model with DataParallel. `train_qlo.py` defaults to `--device_map single`, which is the intended path for one 40GB A100 QLoRA training.

## Quick Start

### Local (GPU available)
```bash
# Verified dataset (138 samples)
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_qlo.py \
    --dataset scripts/training/sft_data/variantc_verified_train.jsonl \
    --val_dataset scripts/training/sft_data/variantc_verified_val.jsonl \
    --output_dir experiment/results/qlo_verified_v1 \
    --epochs 10 \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --learning_rate 2e-5 \
    --lora_r 64 \
    --run_name "qlo_verified_variantc_v1"

# Positive dataset (291 samples)
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_qlo.py \
    --dataset scripts/training/sft_data/variantc_positive_train.jsonl \
    --val_dataset scripts/training/sft_data/variantc_positive_val.jsonl \
    --output_dir experiment/results/qlo_positive_v1 \
    --epochs 6
```

### HPC (SLURM)
```bash
# Submit verified training
sbatch hpc/train_qlo_verified.slurm

# Submit positive training
sbatch hpc/train_qlo_positive.slurm
```

## Output

Training saves:
- `adapter_model.safetensors` — LoRA weights
- `adapter_config.json` — LoRA configuration
- `training_config.json` — Full training hyperparameters
- `train_metrics.json` — Training loss metrics

Verified-v1 completed run:

| Metric | Value |
|--------|-------|
| Dataset | `variantc_verified_train.jsonl` / `variantc_verified_val.jsonl` |
| Train samples | 112 |
| Validation samples | 26 |
| Epochs | 10 |
| Optimizer steps | 40 |
| Final train loss | 1.808561 |
| Final eval loss | 1.792 |
| Output dir | `experiment/results/qlo_verified_v1` |

## Merging Adapters (Optional)

After training, merge LoRA adapters with base model:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "experiment/results/qlo_verified_v1")
model = model.merge_and_unload()
model.save_pretrained("experiment/results/qlo_verified_merged")
```

## Evaluation

After training, benchmark the fine-tuned model against baseline:
```bash
mkdir -p results/benchmarks logs

# Baseline generator: formal 1000-scenario comparison point
CUDA_VISIBLE_DEVICES=0 python experiment/llama_3_8b_verbose.py \
    --mode benchmark \
    --rounds 1000 \
    --dataset-size 1000 \
    --generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
    --benchmark-output results/benchmarks/baseline_generator_v1_summary.json \
    2>&1 | tee logs/baseline_generator_v1.log

# Verified QLoRA adapter on the same deterministic scenario sample
CUDA_VISIBLE_DEVICES=0 python experiment/llama_3_8b_verbose.py \
    --mode benchmark \
    --rounds 1000 \
    --dataset-size 1000 \
    --generator-path experiment/results/qlo_verified_v1 \
    --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
    --benchmark-output results/benchmarks/qlo_verified_v1_summary.json \
    2>&1 | tee logs/qlo_verified_v1_benchmark.log
```

Compare against baseline metrics in `baseline_v1.md`.

For runtime estimation before a full 1000-scenario run, run a 50-scenario probe and multiply wall time by 20:

```bash
CUDA_VISIBLE_DEVICES=0 python experiment/llama_3_8b_verbose.py \
    --mode benchmark \
    --rounds 50 \
    --dataset-size 1000 \
    --generator-path experiment/results/qlo_verified_v1 \
    --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
    --benchmark-output results/benchmarks/qlo_verified_v1_probe50.json \
    2>&1 | tee logs/qlo_verified_v1_probe50.log
```

Track:

- Success rate
- Leak rate / ground-truth success
- Verifier success
- Mean attempts
- Hard-defense success

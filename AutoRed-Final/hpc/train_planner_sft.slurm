#!/bin/bash
# =============================================================================
# AutoRed Planner SFT Training — QLoRA on A100
# =============================================================================
# Usage:
#   sbatch hpc/train_planner_sft.slurm [VARIANT] [EPOCHS] [OUTPUT_DIR]
#
#   VARIANT:    "planner" or "generator" (default: planner)
#   EPOCHS:     Number of training epochs (default: 5)
#   OUTPUT_DIR: Where to save the model (default: experiment/results/planner_sft_v4)
#
# Examples:
#   sbatch hpc/train_planner_sft.slurm planner 5
#   sbatch hpc/train_planner_sft.slurm generator 3 experiment/results/gen_sft_v4
# =============================================================================

#SBATCH --job-name=AutoRed_PlannerSFT
#SBATCH --output=logs/planner_sft_%j.out
#SBATCH --error=logs/planner_sft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --partition=airawatp

# ── Setup ──
mkdir -p logs
source /nlsasfs/home/isea/isea11/slurmJobs/AutoRed/.venv/bin/activate
cd /nlsasfs/home/isea/isea38/AutoRed

# Offline mode (models cached on HPC)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
# Ensure only 1 GPU is used if running interactively (bypassing sbatch)
if [ -z "$CUDA_VISIBLE_DEVICES" ] && [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "Interactive mode detected: Setting CUDA_VISIBLE_DEVICES=0"
fi


# Parse arguments
VARIANT=${1:-planner}
EPOCHS=${2:-5}
OUTPUT_DIR=${3:-experiment/results/${VARIANT}_sft_v4}

# Select dataset paths
if [ "$VARIANT" == "planner" ]; then
    TRAIN_DATA="scripts/training/sft_data/planner_v4_train.jsonl"
    VAL_DATA="scripts/training/sft_data/planner_v4_val.jsonl"
    RUN_NAME="planner_sft_v4_e${EPOCHS}"
    MAX_LENGTH=1536
    BATCH_SIZE=2
    GRAD_ACCUM=16
elif [ "$VARIANT" == "generator" ]; then
    TRAIN_DATA="scripts/training/sft_data/generator_v4_train.jsonl"
    VAL_DATA="scripts/training/sft_data/generator_v4_val.jsonl"
    RUN_NAME="generator_sft_v4_e${EPOCHS}"
    MAX_LENGTH=1024
    BATCH_SIZE=4
    GRAD_ACCUM=8
else
    echo "ERROR: Unknown variant '$VARIANT'. Use 'planner' or 'generator'."
    exit 1
fi

# Verify datasets exist
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found: $TRAIN_DATA"
    echo "Run: python scripts/dataset_tools/build_oracle_sft_dataset.py first"
    exit 1
fi

echo "============================================="
echo "AutoRed Planner SFT Training"
echo "============================================="
echo "Variant     : $VARIANT"
echo "Train Data  : $TRAIN_DATA"
echo "Val Data    : $VAL_DATA"
echo "Output      : $OUTPUT_DIR"
echo "Epochs      : $EPOCHS"
echo "Batch Size  : $BATCH_SIZE × $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "Max Length   : $MAX_LENGTH"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================="

# Count training examples
TRAIN_COUNT=$(wc -l < "$TRAIN_DATA")
VAL_COUNT=$(wc -l < "$VAL_DATA" 2>/dev/null || echo 0)
echo "Training examples : $TRAIN_COUNT"
echo "Validation examples: $VAL_COUNT"
echo ""

# ── Run Training ──
python scripts/training/train_qlo.py \
    --model_name "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2" \
    --dataset "$TRAIN_DATA" \
    --val_dataset "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation "$GRAD_ACCUM" \
    --learning_rate 2e-5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --max_length "$MAX_LENGTH" \
    --device_map single \
    --seed 42 \
    --run_name "$RUN_NAME"

EXIT_CODE=$?

echo ""
echo "============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "Model saved to: $OUTPUT_DIR"
    
    # Show training metrics
    if [ -f "$OUTPUT_DIR/train_metrics.json" ]; then
        echo ""
        echo "Training Metrics:"
        cat "$OUTPUT_DIR/train_metrics.json"
    fi
else
    echo "❌ Training failed with exit code $EXIT_CODE"
fi
echo "============================================="

exit $EXIT_CODE

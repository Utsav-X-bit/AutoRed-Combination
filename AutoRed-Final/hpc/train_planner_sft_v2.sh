#!/bin/bash
# =============================================================================
# AutoRed Planner SFT v2 Training — AC30 subset
# =============================================================================
# Usage:
#   sbatch hpc/train_planner_sft_v2.slurm [EPOCHS] [OUTPUT_DIR]
#
# Examples:
#   sbatch hpc/train_planner_sft_v2.slurm
#   sbatch hpc/train_planner_sft_v2.slurm 5 experiment/results/planner_sft_v2
# =============================================================================

#SBATCH --job-name=AutoRed_PlannerSFTv2
#SBATCH --output=logs/planner_sft_v2_%j.out
#SBATCH --error=logs/planner_sft_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --partition=airawatp

mkdir -p logs
source /nlsasfs/home/isea/isea11/slurmJobs/AutoRed/.venv/bin/activate
cd /nlsasfs/home/isea/isea38/AutoRed

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

if [ -z "$CUDA_VISIBLE_DEVICES" ] && [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "Interactive mode detected: setting CUDA_VISIBLE_DEVICES=0"
fi

EPOCHS=${1:-5}
OUTPUT_DIR=${2:-experiment/results/planner_sft_v2}
TRAIN_DATA="scripts/training/sft_data/planner_v2_train.jsonl"
VAL_DATA="scripts/training/sft_data/planner_v2_val.jsonl"
RUN_NAME="planner_sft_v2_ac30_e${EPOCHS}"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found: $TRAIN_DATA"
    echo "Run:"
    echo "  python scripts/dataset_tools/build_planner_sft_v2.py"
    echo "  python scripts/training/prepare_planner_sft_v2_split.py"
    exit 1
fi

echo "============================================="
echo "AutoRed Planner SFT v2 Training"
echo "============================================="
echo "Train Data   : $TRAIN_DATA"
echo "Val Data     : $VAL_DATA"
echo "Output       : $OUTPUT_DIR"
echo "Epochs       : $EPOCHS"
echo "Batch Size   : 4 x 8 = 32"
echo "Max Length   : 2048"
echo "LoRA         : r=32 alpha=64"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================="

TRAIN_COUNT=$(wc -l < "$TRAIN_DATA")
VAL_COUNT=$(wc -l < "$VAL_DATA" 2>/dev/null || echo 0)
echo "Training examples  : $TRAIN_COUNT"
echo "Validation examples: $VAL_COUNT"
echo ""

python scripts/training/train_qlo.py \
    --model_name "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2" \
    --dataset "$TRAIN_DATA" \
    --val_dataset "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --learning_rate 2e-5 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --max_length 2048 \
    --device_map single \
    --seed 42 \
    --run_name "$RUN_NAME"

EXIT_CODE=$?

echo ""
echo "============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully."
    echo "Model saved to: $OUTPUT_DIR"
    if [ -f "$OUTPUT_DIR/train_metrics.json" ]; then
        echo ""
        echo "Training Metrics:"
        cat "$OUTPUT_DIR/train_metrics.json"
    fi
else
    echo "Training failed with exit code $EXIT_CODE"
fi
echo "============================================="

exit $EXIT_CODE

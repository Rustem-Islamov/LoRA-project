#!/bin/bash
#SBATCH --job-name=lora-code
#SBATCH --output=logs/lora-code_%x_%A_%a.out
#SBATCH --error=logs/lora-code_%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=lucchi-h200
#SBATCH --qos=lucchi
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00

# export MAX_STEPS=2

# echo MAX STEPS = $MAX_STEPS

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./code_job_common.sh "$@"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-LLAMA-2-7B}"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/logs/wandb"
export WANDB_ENTITY="jim-zhao-university-of-basel"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p "$WANDB_DIR"

echo $WANDB_MODE
echo $WANDB_PROJECT
echo $WANDB_DIR

NPROC=2
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))

python - <<'PY'
import torch
import flash_attn

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
print("PyTorch:", torch.__version__)
print("FlashAttention:", flash_attn.__version__)
PY

# The repository's pickle cache has no internal locking.
# Prepare it under a shared lock before either training rank reads it.
mkdir -p data_cache
(
    flock -x 9

    if [[ ! -f data_cache/load_codefeedback.pkl ]]; then
        test -d ./data/CodeFeedback-Filtered-Instruction
    fi

    python - <<'PY'
import peta

dataset = peta.tasks.load_codefeedback()
assert len(dataset["train"]) == 100000
print("CodeFeedback training examples:", len(dataset["train"]))
PY
) 9>data_cache/load_codefeedback.lock

srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    minimal_lora_llama2_code_transformers.py \
    --lora "$METHOD" \
    --seed "$SEED" \
    --lr "$LR" \
    --rank "$LORA_R" \
    --alpha "$LORA_ALPHA" \
    --global-batch-size 32 \
    --per-device-batch-size 16 \
    --epochs 1 \
    --max-steps "${MAX_STEPS:--1}" \
    --output-dir "$RUN_DIR"

test -f "${RUN_DIR}/TRAINING_COMPLETE"
echo "Training finished: $(date)"
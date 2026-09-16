#!/bin/bash
#SBATCH --job-name=lora-code
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a100-80g
#SBATCH --qos=a100-6hours
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=03:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./code_job_common.sh "$@"

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
    --per-device-batch-size 1 \
    --epochs 1 \
    --max-steps "${MAX_STEPS:--1}" \
    --output-dir "$RUN_DIR"

test -f "${RUN_DIR}/TRAINING_COMPLETE"
echo "Training finished: $(date)"
#!/bin/bash
#SBATCH --job-name=fft-code
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a100-80g
#SBATCH --qos=a100-6hours
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line $LINENO: $BASH_COMMAND (exit $rc)" >&2; exit "$rc"' ERR
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

FFT_MAX_STEPS="${2:?Pass max steps as the second argument (-1 for a full run)}"
source ./code_job_common.sh "$1"
if [[ "$METHOD" != "full-ft" ]]; then
    echo "ERROR: this launcher requires method full-ft." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
NPROC=2
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))
echo "FFT max_steps=$FFT_MAX_STEPS"

python - <<'PY'
import torch
import flash_attn
import deepspeed

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
for index in range(2):
    gib = torch.cuda.get_device_properties(index).total_memory / 2**30
    assert gib >= 75, f"FFT requires 80 GB GPUs; GPU {index} has {gib:.1f} GiB"
print("PyTorch:", torch.__version__)
print("FlashAttention:", flash_attn.__version__)
print("DeepSpeed:", deepspeed.__version__)
PY

mkdir -p data_cache
(
    echo "Waiting for CodeFeedback cache lock..."
    flock -x 9
    echo "CodeFeedback cache lock acquired."
    if [[ ! -f data_cache/load_codefeedback.pkl &&
          ! -d data/CodeFeedback-Filtered-Instruction ]]; then
        echo "ERROR: missing data/CodeFeedback-Filtered-Instruction" >&2
        exit 1
    fi
    python - <<'PY'
import peta
dataset = peta.tasks.load_codefeedback()
n = len(dataset["train"])
assert n == 100000, f"Expected 100000 training examples, got {n}"
print("CodeFeedback training examples:", n)
PY
) 9>data_cache/load_codefeedback.lock

srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    minimal_fft_llama2_code_transformers.py \
    --seed "$SEED" \
    --lr "$LR" \
    --output-dir "$RUN_DIR" \
    --deepspeed-config ./config/deepspeed_zero3_fullft_2gpu.json \
    --global-batch-size 32 \
    --per-device-batch-size 1 \
    --epochs 1 \
    --max-steps "$FFT_MAX_STEPS"

test -f "$RUN_DIR/TRAINING_COMPLETE"
echo "FFT finished: $(date)"
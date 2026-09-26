#!/bin/bash
# One Slurm script for all four methods' GSM8K evaluation. It rebuilds the
# checkpoint path with the exact same run-naming code the train_*.py scripts
# used to save it (peta.utils.build_output_dir/build_run_name), so it can
# never drift out of sync with where a checkpoint actually landed, then hands
# off to the single shared evaluator, evaluation/eval_gsm8k.py.
#SBATCH --job-name=eval-gsm8k
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a100,a100-80g
#SBATCH --qos=a100-6hours
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate lorapro39

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN

# Override any of these at submission time, for example:
# sbatch --export=ALL,METHOD=lora-pro-rs-scale,LR=1e-4,SEED=1 eval_gsm8k.sh
#
# METHOD one of:
#   lora                      (LoRA / rsLoRA / DoRA adapters; matches --lora)
#   lora-pro-rs-scale          (LoRA-Pro, paper scaling)
#   lora-pro-standard-scale    (LoRA-Pro, standard scaling)
#   full-ft                    (full fine-tuning)
#   metric-lorapro-mx-rs-scale (Metric LoRA-Pro)
METHOD="${METHOD:-lora}"
MODEL="${MODEL:-qwen3-1.7b-base}"
SEED="${SEED:-0}"
LR="${LR:-1e-4}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DATASET_PATH="${DATASET_PATH:-./data/gsm8k/main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
NPROC="${NPROC:-1}"
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))

# Metric LoRA-Pro's extra hyperparameters; ignored by every other method.
M_X_AVERAGING="${M_X_AVERAGING:-0.90}"
M_X_DAMPING="${M_X_DAMPING:-0.01}"
M_X_SCALE_CLIP="${M_X_SCALE_CLIP:-5.0}"

CHECKPOINT="$(python - "$METHOD" "$MODEL" "$SEED" "$LR" "$RANK" "$ALPHA" \
    "$M_X_AVERAGING" "$M_X_DAMPING" "$M_X_SCALE_CLIP" "$OUTPUT_ROOT" <<'PY'
import sys

from peta.utils import build_output_dir, build_run_name

method, model, seed, lr, rank, alpha, mxavg, damp, clip, output_root = sys.argv[1:]
extra = None
if method.startswith("metric-lorapro"):
    extra = {"mxavg": float(mxavg), "damp": float(damp), "clip": float(clip)}
is_full_ft = method == "full-ft"
run_name = build_run_name(
    seed=int(seed),
    lr=float(lr),
    rank=None if is_full_ft else int(rank),
    alpha=None if is_full_ft else int(alpha),
    extra_hparams=extra,
)
print(build_output_dir(method=method, run_name=run_name, model=model, output_root=output_root))
PY
)"

mkdir -p ./logs/eval

echo "========================================================================"
echo "Job started:          $(date)"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Host:                 $(hostname)"
echo "Working directory:    $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Method:               ${METHOD}"
echo "Model:                ${MODEL}"
echo "Learning rate:        ${LR}"
echo "Seed:                 ${SEED}"
echo "Checkpoint:           ${CHECKPOINT}"
echo "GSM8K:                ${DATASET_PATH}"
echo "========================================================================"

if [[ ! -f "${CHECKPOINT}/config.json" && ! -f "${CHECKPOINT}/adapter_config.json" ]]; then
    echo "ERROR: no config.json or adapter_config.json found in:" >&2
    echo "  ${CHECKPOINT}" >&2
    echo "Nearby saved runs:" >&2
    find "$(dirname "${CHECKPOINT}")" -maxdepth 1 -mindepth 1 -print 2>/dev/null || true
    exit 1
fi
test -d "${DATASET_PATH}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())
assert torch.cuda.is_available()
PY

echo "Starting GSM8K evaluation..."

srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    ./evaluation/eval_gsm8k.py \
    --checkpoint "${CHECKPOINT}" \
    --base-model "${MODEL}" \
    --method "${METHOD}" \
    --dataset-path "${DATASET_PATH}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --batch-size "${BATCH_SIZE}" \
    --max-input-length 768 \
    --max-new-tokens 512

echo "========================================================================"
echo "Evaluation completed: $(date)"
echo "========================================================================"

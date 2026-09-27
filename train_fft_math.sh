#!/bin/bash
#SBATCH --job-name=fullft-math
#SBATCH --output=logs/fullft-math_%j.out
#SBATCH --error=logs/fullft-math_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a100-80g
#SBATCH --qos=a100-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00

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
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-Qwen3-1.7B-Math}"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/logs/wandb"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Override any of these at submission time, for example:
# sbatch --export=ALL,MODEL=qwen3-1.7b-base,SEED=1,LR=2e-5 train_fft_math.sh
MODEL="${MODEL:-qwen3-1.7b-base}"
SEED="${SEED:-0}"
LR="${LR:-2e-5}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
NPROC="${NPROC:-1}"
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))

mkdir -p "$WANDB_DIR"

echo "========================================================================"
echo "Job started:          $(date)"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Host:                 $(hostname)"
echo "Working directory:    $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Model:                ${MODEL}"
echo "Seed:                 ${SEED}"
echo "Learning rate:        ${LR}"
echo "========================================================================"

python - "${MODEL}" <<'PY'
import sys
from pathlib import Path
from peta.utils import resolve_model_path

model_path = Path(resolve_model_path(sys.argv[1]))
assert (model_path / "config.json").is_file(), f"Base model not found at {model_path}"
PY
test -d ./data/MetaMathQA
test -f ./config/deepspeed_zero3_fullft_2gpu.json
test -f ./train_fft_math.py

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available()
PY

# Build the repository's raw-data cache before the distributed launch. Its
# pickle writer is not safe for two ranks creating the same file concurrently.
if [[ ! -f ./data_cache/load_meta_math.pkl ]]; then
    echo "Preparing MetaMathQA raw-data cache..."
    python - <<'PY'
import peta

dataset = peta.tasks.load_meta_math()
print("Training samples:", len(dataset["train"]))
print("Evaluation samples:", len(dataset["eval"]))
assert len(dataset["train"]) == 100000
PY
fi

echo "Starting full-parameter fine-tuning..."

srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    ./train_fft_math.py \
    --model "${MODEL}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --deepspeed-config ./config/deepspeed_zero3_fullft_2gpu.json \
    --seed "$SEED" \
    --lr "$LR" \
    --max-length 1024 \
    --global-batch-size 32 \
    --per-device-batch-size 2 \
    --epochs 1

echo "Training completed: $(date)"

#!/bin/bash
#SBATCH --job-name=lora-math
#SBATCH --output=logs/lora-math_%j.out
#SBATCH --error=logs/lora-math_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a100-80g,a100
#SBATCH --qos=a100-6hours
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00

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
export NCCL_DEBUG=WARN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$WANDB_DIR"

# Override any of these at submission time, for example:
# sbatch --export=ALL,MODEL=qwen3-1.7b-base,SEED=1,LR=1e-4 train_lora_math.sh
MODEL="${MODEL:-qwen3-1.7b-base}"
LORA="${LORA:-lora}"
SEED="${SEED:-0}"
LR="${LR:-1e-4}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
NPROC="${NPROC:-1}"
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "Model: ${MODEL}, method: ${LORA}, seed: ${SEED}, lr: ${LR}, rank: ${RANK}, alpha: ${ALPHA}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - "${MODEL}" <<'PY'
import sys
from pathlib import Path
from peta.utils import resolve_model_path

model_path = Path(resolve_model_path(sys.argv[1]))
assert (model_path / "config.json").is_file(), f"Base model not found at {model_path}"
PY
test -d ./data/MetaMathQA

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available()
PY

if [[ ! -f data_cache/load_meta_math.pkl ]]; then
    echo "Preparing MetaMathQA cache..."

    python - <<'PY'
import peta

dataset = peta.tasks.load_meta_math()
print("Training samples:", len(dataset["train"]))
print("Evaluation samples:", len(dataset["eval"]))

assert len(dataset["train"]) == 100000
PY
fi

echo "Starting LoRA training..."

srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    train_lora_math.py \
    --model "${MODEL}" \
    --lora "${LORA}" \
    --seed "${SEED}" \
    --lr "${LR}" \
    --rank "${RANK}" \
    --alpha "${ALPHA}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}"

echo "Training completed: $(date)"

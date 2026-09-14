#!/bin/bash
#SBATCH --job-name=dora-math
#SBATCH --output=logs/dora-math_%j.out
#SBATCH --error=logs/dora-math_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a100-80g
#SBATCH --qos=a100-6hours
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
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
export WANDB_MODE=online
export WANDB_DIR="${SLURM_SUBMIT_DIR}/logs/wandb"
export NCCL_DEBUG=WARN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$WANDB_DIR"

NPROC=2
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

test -f ./models/llama-2-7b/config.json
test -d ./data/MetaMathQA

python - <<'PY'
import inspect
import flash_attn
import peft
import torch
from peft import LoraConfig

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("FlashAttention:", flash_attn.__version__)
print("PEFT:", peft.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
assert "use_dora" in inspect.signature(LoraConfig).parameters
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

echo "Starting DoRA training..."

srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    minimal_dora_llama2_chat_transformers.py \
    --lora dora \
    --seed 2 \
    --lr 0.000005

echo "Training completed: $(date)"
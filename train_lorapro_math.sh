#!/bin/bash
#SBATCH --job-name=lorapro-paper
#SBATCH --output=logs/lorapro-paper_%j.out
#SBATCH --error=logs/lorapro-paper_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a100-80g,a100
#SBATCH --qos=a100-1day
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
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

SEED="${SEED:-2}"
LR="${LR:-128e-5}"
RS_SCALING="${RS_SCALING:-true}"
NPROC=2

# PYTHONHASHSEED has to exist before torchrun creates the Python processes.
export PYTHONHASHSEED="${SEED}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-LLAMA-2-7B-Math}"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/logs/wandb"
export NCCL_DEBUG=WARN
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))

mkdir -p "$WANDB_DIR"

echo "========================================================================"
echo "Job started:          $(date)"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Host:                 $(hostname)"
echo "Working directory:    $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Seed:                 ${SEED}"
echo "PYTHONHASHSEED:        ${PYTHONHASHSEED}"
echo "Learning rate:        ${LR}"
echo "rs_scaling:           ${RS_SCALING}"
echo "========================================================================"

test -f ./models/llama-2-7b/config.json
test -d ./data/MetaMathQA
test -f ./config/deepspeed_zero2_lorapro_paper.json
test -f ./minimal_lorapro_paper_llama2_math_transformers.py

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python - <<'PY'
import flash_attn
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("FlashAttention:", flash_attn.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
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

echo "Starting paper-consistent LoRA-Pro training..."

srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    minimal_lorapro_paper_llama2_math_transformers.py \
    --lora lora-pro \
    --rs-scaling "${RS_SCALING}" \
    --seed "${SEED}" \
    --lr "${LR}" \
    --global-batch-size 32 \
    --per-device-train-batch-size 2 \
    --deepspeed-config ./config/deepspeed_zero2_lorapro_paper.json

echo "========================================================================"
echo "Training completed: $(date)"
echo "========================================================================"

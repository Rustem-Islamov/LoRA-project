#!/bin/bash
#SBATCH --job-name=lorapro-math
#SBATCH --output=logs/lorapro-math_%j.out
#SBATCH --error=logs/lorapro-math_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a100,a100-80g
#SBATCH --qos=a100-6hours
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

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online
export NCCL_DEBUG=WARN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

NPROC=2
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

# Confirm that all required local files exist.
test -f ./models/llama-2-7b/config.json
test -d ./data/MetaMathQA
test -f ./config/deepspeed_zero2.json

python - <<'PY'
import deepspeed
import flash_attn
import torch

print("PyTorch:", torch.__version__)
print("FlashAttention:", flash_attn.__version__)
print("DeepSpeed:", deepspeed.__version__)
print("DeepSpeed source:", deepspeed.__file__)

assert torch.cuda.device_count() == 2
assert "DeepSpeed-0.15.1" in deepspeed.__file__
PY

# Generate the dataset cache once before multiple ranks start.
if [[ ! -f data_cache/load_meta_math.pkl ]]; then
    echo "Preparing MetaMathQA cache..."
    python - <<'PY'
import peta

dataset = peta.tasks.load_meta_math()
print("Training samples:", len(dataset["train"]))
print("Evaluation samples:", len(dataset["eval"]))
PY
fi

echo "Starting LoRA-Pro training..."

srun --ntasks=1 \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    minimal_lorapro_llama2_math_transformers_old.py \
    --lora rslora-pro \
    --seed 0 \
    --lr 0.00128

echo "Training completed: $(date)"
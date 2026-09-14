#!/bin/bash
#SBATCH --job-name=lorapro-install
#SBATCH --output=lorapro-install_%j.out
#SBATCH --error=lorapro-install_%j.err
#SBATCH --partition=a100,a100-80g
#SBATCH --qos=a100-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
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

# Critical memory-control settings.
export MAX_JOBS=2
export NVCC_THREADS=2
export FLASH_ATTENTION_FORCE_BUILD=TRUE

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Python: $(command -v python)"
echo "CUDA_HOME: ${CUDA_HOME}"
echo "MAX_JOBS: ${MAX_JOBS}"
echo "NVCC_THREADS: ${NVCC_THREADS}"

nvcc --version

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

assert torch.__version__.startswith("2.4.0")
assert torch.version.cuda == "12.1"
assert torch.cuda.is_available()
PY

echo "Building FlashAttention..."

python -m pip install \
    --no-build-isolation \
    --no-cache-dir \
    "flash-attn==2.6.3"

echo "Installing LoRA-Pro's bundled DeepSpeed..."

python -m pip uninstall -y deepspeed || true

python -m pip install \
    --no-build-isolation \
    --no-cache-dir \
    -e ./DeepSpeed-0.15.1

echo "Verifying the environment..."

python - <<'PY'
import torch
import flash_attn
import deepspeed
import transformers
import accelerate
import peft

print("torch:", torch.__version__)
print("flash-attn:", flash_attn.__version__)
print("deepspeed:", deepspeed.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("peft:", peft.__version__)

from flash_attn import flash_attn_func

q = torch.randn(
    1, 128, 8, 64,
    device="cuda",
    dtype=torch.float16,
    requires_grad=True,
)

result = flash_attn_func(q, q, q)
result.sum().backward()

print("FlashAttention GPU test passed:", tuple(result.shape))
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Installation completed: $(date)"
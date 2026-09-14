#!/bin/bash
#SBATCH --job-name=lorapro-install
#SBATCH --output=lorapro-install-%j.out
#SBATCH --error=lorapro-install-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=a100,a100-80g
#SBATCH --qos=a100-6hours
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "Job started: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

# SciCORE provides the CUDA compiler through its software modules.
module load CUDA

echo "Loaded modules:"
module list

nvidia-smi
gcc --version
nvcc --version

# Resolve CUDA_HOME from the loaded nvcc executable.
NVCC_PATH="$(readlink -f "$(command -v nvcc)")"
export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME: $CUDA_HOME"

# ---------------------------------------------------------------------------
# Conda environment
# ---------------------------------------------------------------------------

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "lorapro39"; then
    echo "Reusing Conda environment: lorapro39"
else
    echo "Creating Conda environment: lorapro39"
    conda create -n lorapro39 python=3.9 -y
fi

conda activate lorapro39

echo "Python: $(command -v python)"
python --version

python -m pip install \
    pip==24.2 \
    setuptools==69.5.1 \
    wheel==0.44.0 \
    packaging==24.1 \
    ninja

# ---------------------------------------------------------------------------
# PyTorch
#
# Do not specify download.pytorch.org. Pip is already configured to use
# SciCORE's internal Nexus PyPI proxy.
# ---------------------------------------------------------------------------

python -m pip install \
    torch==2.4.0 \
    torchvision==0.19.0

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())

assert torch.__version__.split("+")[0] == "2.4.0"
assert torch.cuda.is_available(), "Installed PyTorch cannot access the GPU"

print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
PY

# ---------------------------------------------------------------------------
# LoRA-Pro dependencies
# ---------------------------------------------------------------------------

# Requirements leaves PEFT unpinned.
python -m pip install peft==0.12.0

python -m pip install -r requirements.txt

# ---------------------------------------------------------------------------
# FlashAttention
#
# Force a local build using SciCORE's CUDA module. Restrict compilation to
# A100's compute capability to make the build considerably faster.
# ---------------------------------------------------------------------------

export CUDA_HOME
export TORCH_CUDA_ARCH_LIST="8.0"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=8

python -m pip install \
    flash-attn==2.6.3 \
    --no-build-isolation

# ---------------------------------------------------------------------------
# Repository-specific modified DeepSpeed
# ---------------------------------------------------------------------------

DS_BUILD_OPS=0 \
python -m pip install \
    --no-build-isolation \
    -e ./DeepSpeed-0.15.1

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

python - <<'PY'
from pathlib import Path

import torch
import torchvision
import transformers
import peft
import flash_attn
import deepspeed

deepspeed_path = Path(deepspeed.__file__).resolve()

print()
print("============================================================")
print("LoRA-Pro installation verification")
print("============================================================")
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)
print("FlashAttention:", flash_attn.__version__)
print("DeepSpeed:", deepspeed_path)
print("============================================================")

assert torch.__version__.split("+")[0] == "2.4.0"
assert torchvision.__version__.split("+")[0] == "0.19.0"
assert transformers.__version__ == "4.44.0"
assert peft.__version__ == "0.12.0"
assert torch.cuda.is_available()
assert "DeepSpeed-0.15.1" in str(deepspeed_path), (
    "The repository's modified DeepSpeed was not loaded"
)

print("Installation completed successfully.")
PY

python -m pip check

echo "============================================================"
echo "Job finished successfully: $(date)"
echo "============================================================"
#!/bin/bash
#SBATCH --job-name=metric-lorapro
#SBATCH --output=logs/metric-lorapro_%j.out
#SBATCH --error=logs/metric-lorapro_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a100-80g
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
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LORAPRO_REQUIRE_MX=1

mkdir -p "$WANDB_DIR"

# Override any of these at submission time, for example:
# sbatch --export=ALL,SEED=1,LR=2e-5,M_X_AVERAGING=0.95 train_metric_lorapro_math.sh
MODEL="${MODEL:-qwen3-1.7b-base}"
SEED="${SEED:-0}"
LR="${LR:-2e-5}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16}"
M_X_AVERAGING="${M_X_AVERAGING:-0.90}"
M_X_DAMPING="${M_X_DAMPING:-0.01}"
M_X_SCALE_CLIP="${M_X_SCALE_CLIP:-5.0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
NPROC="${NPROC:-1}"
export PYTHONHASHSEED="${SEED}"
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))

echo "========================================================================"
echo "Job started:          $(date)"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Host:                 $(hostname)"
echo "Working directory:    $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Model:                 ${MODEL}"
echo "Seed:                  ${SEED}"
echo "PYTHONHASHSEED:         ${PYTHONHASHSEED}"
echo "Learning rate:         ${LR}"
echo "Rank / alpha:          ${RANK} / ${ALPHA}"
echo "m_x_averaging:         ${M_X_AVERAGING}"
echo "M_x damping:           ${M_X_DAMPING}"
echo "M_x scale clip:        ${M_X_SCALE_CLIP}"
echo "========================================================================"

python - "${MODEL}" <<'PY'
import sys
from pathlib import Path
from peta.utils import resolve_model_path

model_path = Path(resolve_model_path(sys.argv[1]))
assert (model_path / "config.json").is_file(), f"Base model not found at {model_path}"
PY
test -d ./data/MetaMathQA
test -f ./config/deepspeed_zero2_rslorapro_mx.json
test -f ./train_metric_lorapro_math.py

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python - <<'PY'
from pathlib import Path
import deepspeed
import torch

ds_source = Path(deepspeed.__file__).resolve()
patched_file = ds_source.parent / "runtime" / "zero" / "stage_1_and_2.py"

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("DeepSpeed:", deepspeed.__version__)
print("DeepSpeed source:", ds_source)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available()
assert "DeepSpeed-0.15.1" in str(ds_source), (
    "Install the repository copy once with: "
    "python -m pip install -e ./DeepSpeed-0.15.1 --no-build-isolation"
)
source = patched_file.read_text()
for marker in (
    "_lorapro_mx_state",
    "LORAPRO_REQUIRE_MX",
    "effective_A = A * px.unsqueeze(0)",
    "grad_A = grad_A * px_inverse.unsqueeze(0)",
):
    assert marker in source, f"Missing {marker!r} in imported {patched_file}"
PY

# Fast test of mask handling, EMA averaging, damping/clipping, and detachment.
python train_metric_lorapro_math.py --metric-unit-test-only

# Build the local raw-data cache once, before any distributed ranks start.
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

echo "Starting Metric LoRA-Pro training..."

srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    train_metric_lorapro_math.py \
    --model "${MODEL}" \
    --seed "${SEED}" \
    --lr "${LR}" \
    --rank "${RANK}" \
    --alpha "${ALPHA}" \
    --m-x-averaging "${M_X_AVERAGING}" \
    --m-x-damping "${M_X_DAMPING}" \
    --m-x-scale-clip "${M_X_SCALE_CLIP}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --global-batch-size 32 \
    --per-device-train-batch-size 2 \
    --deepspeed-config ./config/deepspeed_zero2_rslorapro_mx.json

echo "========================================================================"
echo "Training completed: $(date)"
echo "========================================================================"

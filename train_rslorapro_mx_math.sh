#!/bin/bash
#SBATCH --job-name=metric-lorapro-mx
#SBATCH --output=logs/metric-lorapro-mx_%j.out
#SBATCH --error=logs/metric-lorapro-mx_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a100-80g
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
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-LLAMA-2-7B}"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/logs/wandb"
export NCCL_DEBUG=WARN
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LORAPRO_REQUIRE_MX=1

mkdir -p "$WANDB_DIR"

NPROC=2
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))

# Override any of these at submission time, for example:
# sbatch --export=ALL,SEED=1,LR=2e-5,M_X_AVERAGING=0.95 train_rslorapro_mx_math.slurm
SEED="${SEED:-2}"
LR="${LR:-16e-5}"
M_X_AVERAGING="${M_X_AVERAGING:-0.90}"
M_X_DAMPING="${M_X_DAMPING:-0.01}"
M_X_SCALE_CLIP="${M_X_SCALE_CLIP:-5.0}"
export PYTHONHASHSEED="${SEED}"

read -r LR_DIR METRIC_TAG < <(
    python - "${LR}" "${M_X_AVERAGING}" "${M_X_DAMPING}" \
        "${M_X_SCALE_CLIP}" <<'PY'
import sys

values = [float(value) for value in sys.argv[1:]]
fmt = lambda value: format(value, ".12g")
print(
    fmt(values[0]),
    f"mxavg_{fmt(values[1])}_damp_{fmt(values[2])}_clip_{fmt(values[3])}",
)
PY
)
METHOD="metric-lorapro-mx-rs-scale"
OUTPUT_DIR="./logs/transformers/llama-2-7b/math/${LR_DIR}/${METHOD}/${METRIC_TAG}/${SEED}"

echo "========================================================================"
echo "Job started:          $(date)"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Host:                 $(hostname)"
echo "Working directory:    $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Seed:                  ${SEED}"
echo "PYTHONHASHSEED:         ${PYTHONHASHSEED}"
echo "Learning rate:         ${LR}"
echo "m_x_averaging:         ${M_X_AVERAGING}"
echo "M_x damping:           ${M_X_DAMPING}"
echo "M_x scale clip:        ${M_X_SCALE_CLIP}"
echo "Output directory:      ${OUTPUT_DIR}"
echo "========================================================================"

test -f ./models/llama-2-7b/config.json
test -d ./data/MetaMathQA
test -f ./config/deepspeed_zero2_rslorapro_mx.json
test -f ./minimal_rslorapro_mx_llama2_math_transformers.py

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python - <<'PY'
from pathlib import Path
import deepspeed
import flash_attn
import torch

ds_source = Path(deepspeed.__file__).resolve()
patched_file = ds_source.parent / "runtime" / "zero" / "stage_1_and_2.py"

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("FlashAttention:", flash_attn.__version__)
print("DeepSpeed:", deepspeed.__version__)
print("DeepSpeed source:", ds_source)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
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
python minimal_rslorapro_mx_llama2_math_transformers.py \
    --metric-unit-test-only

# Build the local raw-data cache once, before the two distributed ranks start.
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

echo "Starting M_x Metric-LoRA-Pro training..."

srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    minimal_rslorapro_mx_llama2_math_transformers.py \
    --lora metric-lorapro-mx-rs-scale \
    --seed "${SEED}" \
    --lr "${LR}" \
    --m-x-averaging "${M_X_AVERAGING}" \
    --m-x-damping "${M_X_DAMPING}" \
    --m-x-scale-clip "${M_X_SCALE_CLIP}" \
    --global-batch-size 32 \
    --per-device-train-batch-size 2 \
    --deepspeed-config ./config/deepspeed_zero2_rslorapro_mx.json

echo "========================================================================"
echo "Training completed: $(date)"
echo "========================================================================"

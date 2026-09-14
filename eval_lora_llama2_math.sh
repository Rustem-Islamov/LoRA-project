#!/bin/bash
#SBATCH --job-name=eval-lora-gsm8k
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a100,a100-80g
#SBATCH --qos=a100-6hours
#SBATCH --gres=gpu:2
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

NPROC=2
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))

# Override at submission time with, for example:
# sbatch --export=ALL,SEED=1 evaluation/eval_lora_1e-05_gsm8k.slurm
SEED="${SEED:-0}"

BASE_MODEL="./models/llama-2-7b"
DATASET_PATH="./data/gsm8k/main"
ADAPTER_ROOT="./logs/transformers/llama-2-7b/math/0.00128/lora"
EVALUATOR="./evaluation/eval_llama2_math_gsm8k.py"
OUTPUT_JSON="./logs/eval/gsm8k_lora_lr0.00128_seed${SEED}.json"

# Training normally saves each seed in its own subdirectory. Also accept an
# adapter saved directly in ADAPTER_ROOT.
if [[ -f "${ADAPTER_ROOT}/${SEED}/adapter_config.json" ]]; then
    ADAPTER_PATH="${ADAPTER_ROOT}/${SEED}"
elif [[ -f "${ADAPTER_ROOT}/adapter_config.json" ]]; then
    ADAPTER_PATH="${ADAPTER_ROOT}"
else
    echo "ERROR: adapter_config.json was not found in:" >&2
    echo "  ${ADAPTER_ROOT}/${SEED}" >&2
    echo "  ${ADAPTER_ROOT}" >&2
    echo "Available files:" >&2
    find "${ADAPTER_ROOT}" -maxdepth 2 -type f -print 2>/dev/null || true
    exit 1
fi

# This also creates the result directory. The same directory must already
# exist before sbatch is called so Slurm can open its .out and .err files.
mkdir -p ./logs/eval

echo "========================================================================"
echo "Job started:          $(date)"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Host:                 $(hostname)"
echo "Working directory:    $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "OMP_NUM_THREADS:      ${OMP_NUM_THREADS}"
echo "Base model:           ${BASE_MODEL}"
echo "Adapter:              ${ADAPTER_PATH}"
echo "GSM8K:                ${DATASET_PATH}"
echo "Output JSON:          ${OUTPUT_JSON}"
echo "========================================================================"

test -f "${BASE_MODEL}/config.json"
test -d "${DATASET_PATH}"
test -f "${ADAPTER_PATH}/adapter_config.json"
test -f "${EVALUATOR}"

if [[ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" \
   && ! -f "${ADAPTER_PATH}/adapter_model.bin" ]]; then
    echo "ERROR: adapter weights were not found in ${ADAPTER_PATH}" >&2
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total \
    --format=csv,noheader

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

echo "Starting GSM8K evaluation..."

srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    "${EVALUATOR}" \
    --base-model "${BASE_MODEL}" \
    --dataset-path "${DATASET_PATH}" \
    --adapter-path "${ADAPTER_PATH}" \
    --method lora \
    --seed "${SEED}" \
    --learning-rate-directory 0.00128 \
    --output-json "${OUTPUT_JSON}" \
    --batch-size 8 \
    --max-input-length 768 \
    --max-new-tokens 512

echo "========================================================================"
echo "Evaluation completed: $(date)"
echo "Results:              ${OUTPUT_JSON}"
echo "========================================================================"

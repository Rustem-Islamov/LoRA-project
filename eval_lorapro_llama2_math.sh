#!/bin/bash
#SBATCH --job-name=eval-lorapro-gsm8k
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

# Override any of these with sbatch --export. The defaults reproduce the
# Table 2 LoRA-Pro setup at learning rate 2e-05 and seed 0.
METHOD="${METHOD:-rslora-pro}"
#LR="${LR:-8e-05}"
LR="${LR:-0.00128}"
SEED="${SEED:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"

BASE_MODEL="./models/llama-2-7b"
DATASET_PATH="./data/gsm8k/main"
ADAPTER_ROOT="./logs/transformers/llama-2-7b/math/${LR}/${METHOD}"
ADAPTER_PATH="${ADAPTER_ROOT}/${SEED}"
EVALUATOR="./evaluation/eval_llama2_math_gsm8k.py"
OUTPUT_JSON="./logs/eval/gsm8k_${METHOD}_lr${LR}_seed${SEED}.json"

# Normally each seed is saved in its own directory. Also accept a checkpoint
# saved directly in ADAPTER_ROOT for compatibility with older runs.
if [[ ! -f "${ADAPTER_PATH}/adapter_config.json" ]]; then
    if [[ -f "${ADAPTER_ROOT}/adapter_config.json" ]]; then
        ADAPTER_PATH="${ADAPTER_ROOT}"
    else
        echo "ERROR: adapter_config.json was not found in:" >&2
        echo "  ${ADAPTER_ROOT}/${SEED}" >&2
        echo "  ${ADAPTER_ROOT}" >&2
        echo "Available files under the learning-rate directory:" >&2
        find "./logs/transformers/llama-2-7b/math/${LR}" \
            -maxdepth 3 -type f -print 2>/dev/null || true
        exit 1
    fi
fi

# logs/eval must already exist when sbatch is invoked because Slurm opens the
# .out and .err files before this script starts. This mkdir handles JSON output.
mkdir -p ./logs/eval

echo "========================================================================"
echo "Job started:          $(date)"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Host:                 $(hostname)"
echo "Working directory:    $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "OMP_NUM_THREADS:      ${OMP_NUM_THREADS}"
echo "Method:               ${METHOD}"
echo "Learning rate:        ${LR}"
echo "Seed:                 ${SEED}"
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

echo "Starting LoRA-Pro GSM8K evaluation..."

srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    "${EVALUATOR}" \
    --base-model "${BASE_MODEL}" \
    --dataset-path "${DATASET_PATH}" \
    --adapter-path "${ADAPTER_PATH}" \
    --method "${METHOD}" \
    --seed "${SEED}" \
    --learning-rate-directory "${LR}" \
    --output-json "${OUTPUT_JSON}" \
    --batch-size "${BATCH_SIZE}" \
    --max-input-length 768 \
    --max-new-tokens 512

echo "========================================================================"
echo "Evaluation completed: $(date)"
echo "Results:              ${OUTPUT_JSON}"
echo "========================================================================"

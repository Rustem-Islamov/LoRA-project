#!/bin/bash
#SBATCH --job-name=eval-fullft-gsm8k
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

# Override at submission time, for example:
# sbatch --export=ALL,LR=2e-5,SEED=1 eval_fullft_llama2_math.sh
LR="${LR:-2e-5}"
SEED="${SEED:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
export PYTHONHASHSEED="${SEED}"

MODEL_ROOT="./logs/transformers/llama-2-7b/math/${LR}/full-ft"
MODEL_PATH="${MODEL_ROOT}/${SEED}"
DATASET_PATH="./data/gsm8k/main"
EVALUATOR="./evaluation/eval_llama2_fft_math_gsm8k.py"
OUTPUT_JSON="./logs/eval/gsm8k_full-ft_lr${LR}_seed${SEED}.json"

# Accept an older full checkpoint saved directly in MODEL_ROOT.
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    if [[ -f "${MODEL_ROOT}/config.json" ]]; then
        MODEL_PATH="${MODEL_ROOT}"
    else
        echo "ERROR: full-model config.json was not found in:" >&2
        echo "  ${MODEL_ROOT}/${SEED}" >&2
        echo "  ${MODEL_ROOT}" >&2
        echo "Available files under the learning-rate directory:" >&2
        find "./logs/transformers/llama-2-7b/math/${LR}" \
            -maxdepth 4 -type f -print 2>/dev/null || true
        exit 1
    fi
fi

mkdir -p ./logs/eval

echo "========================================================================"
echo "Job started:          $(date)"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Host:                 $(hostname)"
echo "Working directory:    $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "OMP_NUM_THREADS:      ${OMP_NUM_THREADS}"
echo "Method:               full-ft"
echo "Learning rate:        ${LR}"
echo "Training seed:        ${SEED}"
echo "Full model:           ${MODEL_PATH}"
echo "GSM8K:                ${DATASET_PATH}"
echo "Output JSON:          ${OUTPUT_JSON}"
echo "========================================================================"

test -f "${MODEL_PATH}/config.json"
test -d "${DATASET_PATH}"
test -f "${EVALUATOR}"

if [[ -f "${MODEL_PATH}/adapter_config.json" ]]; then
    echo "ERROR: ${MODEL_PATH} is an adapter checkpoint, not a full model." >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}/model.safetensors" \
   && ! -f "${MODEL_PATH}/model.safetensors.index.json" \
   && ! -f "${MODEL_PATH}/pytorch_model.bin" \
   && ! -f "${MODEL_PATH}/pytorch_model.bin.index.json" ]]; then
    echo "ERROR: full-model weights were not found in ${MODEL_PATH}" >&2
    find "${MODEL_PATH}" -maxdepth 1 -type f -print >&2 || true
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

echo "Starting full-FT GSM8K evaluation..."

srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    "${EVALUATOR}" \
    --model-path "${MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --seed "${SEED}" \
    --learning-rate "${LR}" \
    --output-json "${OUTPUT_JSON}" \
    --batch-size "${BATCH_SIZE}" \
    --max-input-length 768 \
    --max-new-tokens 512

echo "========================================================================"
echo "Evaluation completed: $(date)"
echo "Results:              ${OUTPUT_JSON}"
echo "========================================================================"

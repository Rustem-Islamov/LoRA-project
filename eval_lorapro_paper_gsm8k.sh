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

SEED="${SEED:-1}"
LR="${LR:-1e-05}"
RS_SCALING="${RS_SCALING:-true}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NPROC=2

case "${RS_SCALING,,}" in
    true|1|yes|y|on)
        RS_SCALING=true
        DEFAULT_METHOD="lora-pro-rs-scale"
        ;;
    false|0|no|n|off)
        RS_SCALING=false
        DEFAULT_METHOD="lora-pro-standard-scale"
        ;;
    *)
        echo "ERROR: RS_SCALING must be true or false, got '${RS_SCALING}'." >&2
        exit 2
        ;;
esac

METHOD="${METHOD:-${DEFAULT_METHOD}}"
# Training parses LR as a Python float, so 2e-5 is saved under 2e-05.
LR_DIR="${LR_DIR:-$(python -c 'import sys; print(float(sys.argv[1]))' "${LR}")}"

export PYTHONHASHSEED="${SEED}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))

BASE_MODEL="./models/llama-2-7b"
DATASET_PATH="./data/gsm8k/main"
ADAPTER_PATH="./logs/transformers/llama-2-7b/math/${LR_DIR}/${METHOD}/${SEED}"
EVALUATOR="./evaluation/eval_lorapro_paper_gsm8k.py"
OUTPUT_JSON="./logs/eval/gsm8k_${METHOD}_lr${LR_DIR}_seed${SEED}.json"

# This directory must also exist before sbatch is called, because Slurm opens
# its .out and .err files before this script begins executing.
mkdir -p ./logs/eval

echo "========================================================================"
echo "Job started:          $(date)"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Host:                 $(hostname)"
echo "Working directory:    $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Method:               ${METHOD}"
echo "Learning rate input:  ${LR}"
echo "Learning-rate folder: ${LR_DIR}"
echo "Training seed:        ${SEED}"
echo "Expected rs scaling:  ${RS_SCALING}"
echo "Base model:           ${BASE_MODEL}"
echo "Adapter:              ${ADAPTER_PATH}"
echo "GSM8K:                ${DATASET_PATH}"
echo "Output JSON:          ${OUTPUT_JSON}"
echo "========================================================================"

test -f "${BASE_MODEL}/config.json"
test -d "${DATASET_PATH}"
test -f "${EVALUATOR}"

if [[ ! -f "${ADAPTER_PATH}/adapter_config.json" ]]; then
    echo "ERROR: adapter_config.json not found in ${ADAPTER_PATH}" >&2
    echo "Nearby saved files:" >&2
    find "./logs/transformers/llama-2-7b/math/${LR_DIR}" \
        -maxdepth 4 -type f -print 2>/dev/null || true
    exit 1
fi

if [[ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" \
   && ! -f "${ADAPTER_PATH}/adapter_model.bin" ]]; then
    echo "ERROR: adapter weights not found in ${ADAPTER_PATH}" >&2
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python - <<'PY'
import flash_attn
import peft
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("PEFT:", peft.__version__)
print("FlashAttention:", flash_attn.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
PY

echo "Starting corrected LoRA-Pro GSM8K evaluation..."

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
    --learning-rate-directory "${LR_DIR}" \
    --expected-rs-scaling "${RS_SCALING}" \
    --output-json "${OUTPUT_JSON}" \
    --batch-size "${BATCH_SIZE}" \
    --max-input-length 768 \
    --max-new-tokens 512

echo "========================================================================"
echo "Evaluation completed: $(date)"
echo "Results:              ${OUTPUT_JSON}"
echo "========================================================================"

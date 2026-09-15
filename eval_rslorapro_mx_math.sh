#!/bin/bash
#SBATCH --job-name=eval-metric-lorapro-mx
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a100,a100-80g
#SBATCH --qos=a100-30min
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load CUDA/12.1.1

eval "$(conda shell.bash hook)"
conda activate lorapro39

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

SEED="${SEED:-0}"
LR="${LR:-8e-5}"
M_X_AVERAGING="${M_X_AVERAGING:-0.90}"
M_X_DAMPING="${M_X_DAMPING:-0.01}"
M_X_SCALE_CLIP="${M_X_SCALE_CLIP:-5.0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NPROC=2

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
BASE_MODEL="./models/llama-2-7b"
DATASET_PATH="./data/gsm8k/main"
ADAPTER_PATH="./logs/transformers/llama-2-7b/math/${LR_DIR}/${METHOD}/${METRIC_TAG}/${SEED}"
EVALUATOR="./evaluation/eval_lorapro_paper_gsm8k.py"
OUTPUT_JSON="./logs/eval/gsm8k_${METHOD}_${METRIC_TAG}_lr${LR_DIR}_seed${SEED}.json"

export PYTHONHASHSEED="${SEED}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))

# This directory must already exist when sbatch is invoked so Slurm can open
# stdout/stderr. The mkdir remains useful for the JSON result itself.
mkdir -p ./logs/eval

echo "========================================================================"
echo "Job started:          $(date)"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Host:                 $(hostname)"
echo "Working directory:    $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Method:               ${METHOD}"
echo "Learning-rate folder: ${LR_DIR}"
echo "Training seed:        ${SEED}"
echo "Metric tag:           ${METRIC_TAG}"
echo "Base model:           ${BASE_MODEL}"
echo "Adapter:              ${ADAPTER_PATH}"
echo "GSM8K:                ${DATASET_PATH}"
echo "Output JSON:          ${OUTPUT_JSON}"
echo "========================================================================"

test -f "${BASE_MODEL}/config.json"
test -d "${DATASET_PATH}"
test -f "${EVALUATOR}"
test -f "${ADAPTER_PATH}/adapter_config.json"
test -f "${ADAPTER_PATH}/mx_training_config.json"
test -f "${ADAPTER_PATH}/mx_metric_state.pt"

if [[ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" \
   && ! -f "${ADAPTER_PATH}/adapter_model.bin" ]]; then
    echo "ERROR: adapter weights not found in ${ADAPTER_PATH}" >&2
    exit 1
fi

# Refuse to evaluate a native/unfolded metric checkpoint with ordinary PEFT.
python - "${ADAPTER_PATH}" "${SEED}" "${LR}" "${M_X_AVERAGING}" \
    "${M_X_DAMPING}" "${M_X_SCALE_CLIP}" <<'PY'
import json
import math
import sys
from pathlib import Path

adapter = Path(sys.argv[1])
expected = {
    "seed": int(sys.argv[2]),
    "learning_rate": float(sys.argv[3]),
    "m_x_averaging": float(sys.argv[4]),
    "m_x_damping": float(sys.argv[5]),
    "m_x_scale_clip": float(sys.argv[6]),
}
with (adapter / "mx_training_config.json").open() as handle:
    config = json.load(handle)

assert config["method"] == "metric-lorapro-mx-rs-scale", config["method"]
assert config["metric_folded_into_lora_A"] is True
assert config["metric_requires_grad"] is False
assert config["use_rslora"] is True
assert int(config["rank"]) == 8
assert float(config["lora_alpha"]) == 16.0
assert int(config["optimizer_steps"]) == 3125, config["optimizer_steps"]
assert config["metric_updates_per_layer"] == [3125], config[
    "metric_updates_per_layer"
]
for key, value in expected.items():
    actual = config[key]
    if isinstance(value, float):
        assert math.isclose(float(actual), value, rel_tol=1e-12, abs_tol=1e-12), (
            key,
            actual,
            value,
        )
    else:
        assert actual == value, (key, actual, value)
print("Metric checkpoint metadata verified:", config)
PY

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

echo "Starting M_x Metric-LoRA-Pro GSM8K evaluation..."

srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    "${EVALUATOR}" \
    --base-model "${BASE_MODEL}" \
    --dataset-path "${DATASET_PATH}" \
    --adapter-path "${ADAPTER_PATH}" \
    --method "${METHOD}-${METRIC_TAG}" \
    --seed "${SEED}" \
    --learning-rate-directory "${LR_DIR}" \
    --expected-rs-scaling true \
    --output-json "${OUTPUT_JSON}" \
    --batch-size "${BATCH_SIZE}" \
    --max-input-length 768 \
    --max-new-tokens 512

echo "========================================================================"
echo "Evaluation completed: $(date)"
echo "Results:              ${OUTPUT_JSON}"
echo "========================================================================"

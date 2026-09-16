#!/bin/bash
# Sourced by the train/eval jobs.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

MANIFEST="${1:?Pass the absolute path to sweep.tsv}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script as an array}"

mapfile -t ROWS < "$MANIFEST"
if (( TASK_ID < 0 || TASK_ID >= ${#ROWS[@]} )); then
    echo "Array index is outside the sweep table." >&2
    exit 1
fi

read -r METHOD SEED LR LORA_R LORA_ALPHA <<< "${ROWS[$TASK_ID]}"

SWEEP_DIR="$(dirname "$MANIFEST")"
RUN_DIR="${SWEEP_DIR}/runs/${METHOD}/r${LORA_R}_a${LORA_ALPHA}/lr${LR}/seed${SEED}"
RESULT_DIR="${RUN_DIR}/humaneval"

module purge
module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate lorapro39

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export PYTHONHASHSEED="$SEED"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN

# This sweep uses ordinary Pro / RS-Pro, without the M_x extension.
unset LORAPRO_REQUIRE_MX

test -f ./models/llama-2-7b/config.json

echo "Method=$METHOD Seed=$SEED LR=$LR Rank=$LORA_R Alpha=$LORA_ALPHA"
echo "Run directory: $RUN_DIR"
#!/bin/bash
#SBATCH --job-name=fullft-math
#SBATCH --output=logs/fullft-math_%j.out
#SBATCH --error=logs/fullft-math_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a100-80g
#SBATCH --qos=a100-6hours
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
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
export WANDB_MODE=online
export WANDB_PROJECT=LLAMA-2-7B
export WANDB_DIR="${SLURM_SUBMIT_DIR}/logs/wandb"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC=2
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))

SEED="${SEED:-1}"
LR="${LR:-5e-6}"
OUTPUT_DIR="./logs/transformers/llama-2-7b/math/${LR}/full-ft/${SEED}"

mkdir -p "$WANDB_DIR" "$OUTPUT_DIR"

echo "========================================================================"
echo "Job started:          $(date)"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Host:                 $(hostname)"
echo "Working directory:    $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Output directory:     ${OUTPUT_DIR}"
echo "Seed:                 ${SEED}"
echo "Learning rate:        ${LR}"
echo "========================================================================"

test -f ./models/llama-2-7b/config.json
test -d ./data/MetaMathQA
test -f ./config/deepspeed_zero3_fullft_2gpu.json
test -f ./minimal_fft_llama2_math_transformers.py

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<'PY'
import deepspeed
import flash_attn
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("FlashAttention:", flash_attn.__version__)
print("DeepSpeed:", deepspeed.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
for index in range(2):
    total_gib = torch.cuda.get_device_properties(index).total_memory / 2**30
    assert total_gib >= 75, f"GPU {index} has only {total_gib:.1f} GiB"
PY

# Build the repository's raw-data cache before the distributed launch. Its
# pickle writer is not safe for two ranks creating the same file concurrently.
if [[ ! -f ./data_cache/load_meta_math.pkl ]]; then
    echo "Preparing MetaMathQA raw-data cache..."
    python - <<'PY'
import peta

dataset = peta.tasks.load_meta_math()
print("Training samples:", len(dataset["train"]))
print("Evaluation samples:", len(dataset["eval"]))
assert len(dataset["train"]) == 100000
PY
fi

echo "Starting full-parameter fine-tuning..."

srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    ./minimal_fft_llama2_math_transformers.py \
    --model-path ./models/llama-2-7b \
    --deepspeed-config ./config/deepspeed_zero3_fullft_2gpu.json \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --lr "$LR" \
    --max-length 1024 \
    --global-batch-size 32 \
    --per-device-batch-size 2 \
    --epochs 1

echo "Training completed: $(date)"

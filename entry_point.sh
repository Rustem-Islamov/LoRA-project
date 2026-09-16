
set -euo pipefail

# These directories must exist before sbatch opens its log files.
mkdir -p logs/slurm logs/sweeps

# Each submission gets its own manifest and output directories.
SWEEP_DIR="$(mktemp -d "$PWD/logs/sweeps/code.XXXXXX")"
MANIFEST="${SWEEP_DIR}/sweep.tsv"

MAX_STEPS=2

python - "$MANIFEST" <<'PY'
import itertools
import sys

# Edit these values before submitting.
seeds = [0] #, 1
learning_rates = [2e-5]  # Examples; replace with your range. , 8e-5, 3.2e-4
methods = ["lora"]
rank = 8
alpha = 16

with open(sys.argv[1], "w") as handle:
    for method, lr, seed in itertools.product(
        methods, learning_rates, seeds
    ):
        handle.write(f"{method}\t{seed}\t{lr}\t{rank}\t{alpha}\n")
PY

N_RUNS="$(wc -l < "$MANIFEST")"
LAST_TASK=$((N_RUNS - 1))

# At most two training experiments run concurrently: four training GPUs.
TRAIN_JOB="$(sbatch --parsable \
    --array="0-${LAST_TASK}%2" \
    train_lora_code.sh "$MANIFEST")"
TRAIN_JOB="${TRAIN_JOB%%;*}"

EVAL_JOB="$(sbatch --parsable \
    --array="0-${LAST_TASK}%2" \
    --dependency="aftercorr:${TRAIN_JOB}" \
    --kill-on-invalid-dep=yes \
    eval_lora_llama2_code.sh "$MANIFEST")"

echo "Sweep: $SWEEP_DIR"
echo "Training array: $TRAIN_JOB"
echo "Evaluation array: $EVAL_JOB"
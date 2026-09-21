#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export LORAPRO_ENV="${LORAPRO_ENV:-$PWD/.venv}"
FFT_MAX_STEPS="${MAX_STEPS:-2}"

mkdir -p logs/slurm logs/sweeps
SWEEP_DIR="$(mktemp -d "$PWD/logs/sweeps/fft-code.XXXXXX")"
MANIFEST="$SWEEP_DIR/sweep.tsv"

"$LORAPRO_ENV/bin/python" - "$MANIFEST" "$FFT_MAX_STEPS" <<'PY'
import itertools
import sys

steps = int(sys.argv[2])
if steps == 0 or steps < -1:
    raise ValueError("MAX_STEPS must be -1 or a positive integer.")

seeds = [0]
learning_rates = [5e-6]  # Edit for the FFT sweep.

with open(sys.argv[1], "w") as handle:
    for lr, seed in itertools.product(learning_rates, seeds):
        # Five-column manifest; FFT ignores rank and alpha.
        handle.write(f"full-ft\t{seed}\t{lr}\t0\t0\n")
PY

N_RUNS="$(wc -l < "$MANIFEST")"
LAST_TASK=$((N_RUNS - 1))

TRAIN_JOB="$(sbatch --parsable --export=ALL \
    --array="0-${LAST_TASK}%1" \
    train_fft_code.sh "$MANIFEST" "$FFT_MAX_STEPS")"
TRAIN_JOB="${TRAIN_JOB%%;*}"

EVAL_JOB="$(sbatch --parsable --export=ALL \
    --array="0-${LAST_TASK}%1" \
    --dependency="aftercorr:${TRAIN_JOB}" \
    --kill-on-invalid-dep=yes \
    eval_lora_llama2_code.sh "$MANIFEST")"
EVAL_JOB="${EVAL_JOB%%;*}"

echo "Sweep: $SWEEP_DIR"
echo "FFT max steps: $FFT_MAX_STEPS"
echo "Training array: $TRAIN_JOB"
echo "HumanEval array: $EVAL_JOB"
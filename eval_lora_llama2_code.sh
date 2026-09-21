#!/bin/bash
#SBATCH --job-name=eval-lora-code
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=lucchi-h200
#SBATCH --qos=lucchi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./code_job_common.sh "$@"

export OMP_NUM_THREADS=4

# test -f "${RUN_DIR}/TRAINING_COMPLETE"
# test -f "${RUN_DIR}/adapter_config.json"
# mkdir -p "$RESULT_DIR"

# srun --ntasks=1 --kill-on-bad-exit=1 \
#     python evaluation/eval_llama2_code.py \
#     --base-model /scicore/home/lucchi0001/zhao0005/models/llama-2-7b \
#     --adapter-path "$RUN_DIR" \
#     --output-file "${RESULT_DIR}/samples.jsonl" \
#     --batch-size 16 \
#     --max-new-tokens 512

export PYTHONUNBUFFERED=1
if [[ ! -f "$RUN_DIR/TRAINING_COMPLETE" ]]; then
    echo "ERROR: training did not complete: $RUN_DIR" >&2
    exit 1
fi

if [[ "$METHOD" == "full-ft" ]]; then
    MODEL_ARGS=(--model-path "$RUN_DIR")
else
    MODEL_ARGS=(--base-model ./models/llama-2-7b --adapter-path "$RUN_DIR")
fi

mkdir -p "$RESULT_DIR"
echo "Generating HumanEval predictions for $METHOD..."
srun --ntasks=1 --kill-on-bad-exit=1 \
    python evaluation/eval_llama2_code.py \
    "${MODEL_ARGS[@]}" \
    --output-file "$RESULT_DIR/samples.jsonl" \
    --batch-size 4 \
    --max-new-tokens 512

echo "Generation finished; scoring HumanEval..."

# Run this scoring process within the cluster's supported code sandbox.
srun --ntasks=1 --kill-on-bad-exit=1 \
    python - "$RESULT_DIR" "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path
from human_eval.evaluation import evaluate_functional_correctness

result_dir = Path(sys.argv[1])
run_dir = Path(sys.argv[2])

metrics = evaluate_functional_correctness(
    sample_file=str(result_dir / "samples.jsonl"),
    k=[1],
    n_workers=4,
    timeout=3.0,
)

training = json.loads((run_dir / "training_config.json").read_text())
generation = json.loads(
    (result_dir / "generation_config.json").read_text()
)

result = {
    "method": training.get("method", training.get("lora")),
    "rank": training.get("rank"),
    "alpha": training.get("alpha"),
    "seed": training["seed"],
    "learning_rate": training["lr"],
    "generation": generation,
    "metrics": {key: float(value) for key, value in metrics.items()},
}
(result_dir / "metrics.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
PY

echo "Evaluation finished: ${RESULT_DIR}/metrics.json"
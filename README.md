# LoRA-Pro: Are Low-Rank Adapters Properly Optimized?

This repo contains the pre-release version of LoRA-Pro, proposed by [LoRA-Pro: Are Low-Rank Adapters Properly Optimized?](https://openreview.net/forum?id=gTwRMU3lJ5).

In LoRA-Pro, we uncover a fundamental connection between the optimization processes of LoRA and full fine-tuning: using LoRA for optimization is mathematically equivalent to full fine-tuning using a low-rank gradient for parameter updates. And this low-rank gradient can be expressed in terms of the gradients of the two low-rank matrices in LoRA. Leveraging this insight, we introduce LoRA-Pro, a method that enhances LoRA's performance by strategically adjusting the gradients of these low-rank matrices. This adjustment allows the low-rank gradient to more accurately approximate the full fine-tuning gradient, thereby narrowing the performance gap between LoRA and full fine-tuning. Furthermore, we theoretically derive the optimal solutions for adjusting the gradients of the low-rank matrices, applying them during fine-tuning in LoRA-Pro.



## Requirements

### Installation

Create a conda environment and install dependencies:

```shell
git clone https://github.com/mrflogs/LoRA-Pro.git
cd LoRA-Pro

conda activate -n lorapro python=3.9
conda activate lorapro

# install required package
pip install flash-attn --no-build-isolation
pip install -r requirements.txt

# install modified deepspeed
pip install -e DeepSpeed-0.15.1
```

### Datasets and Models

Install a base model from huggingface and link it to `./models`. The scripts in
this repo default to `Qwen3-1.7B-Base` at `./models/Qwen3-1.7B-Base`; Llama-2-7B
at `./models/llama-2-7b` is also supported (see `MODEL_PATHS` in
`peta/utils/experiment.py`, or pass an arbitrary path via `--model`/`MODEL=`).

Install datasets (MetaMathQA, GSM8K, WizardLM, and CodeFeedback-Filtered-Instruction,
etc.) from huggingface and link them to `./data`.

## Compare LoRA, LoRA-Pro, full fine-tuning, and Metric LoRA-Pro on Qwen3-1.7B-Base

Four methods share the same MetaMathQA-100k -> GSM8K pipeline:

| Method            | Training script              | Slurm wrapper                    |
|-------------------|-------------------------------|-----------------------------------|
| LoRA (/ rsLoRA / DoRA) | `train_lora_math.py`     | `train_lora_math.sh`              |
| LoRA-Pro          | `train_lorapro_math.py`       | `train_lorapro_math.sh`           |
| Full fine-tuning  | `train_fft_math.py`           | `train_fft_math.sh`               |
| Metric LoRA-Pro    | `train_metric_lorapro_math.py`| `train_metric_lorapro_math.sh`    |

All four save their checkpoint under a single, hyperparameter-qualified
directory built by `peta/utils/experiment.py`:

```
./checkpoints/<model-slug>/math/<method>/<hyperparameters>/
```

For example, LoRA at rank 8, alpha 16, learning rate 1e-4, seed 0 on
Qwen3-1.7B-Base lands at
`./checkpoints/qwen3-1.7b-base/math/lora/r8_a16_lr0.0001_seed0/`. Every run
also writes a `metadata.json` recording every hyperparameter and the base
model it started from, so two runs can never silently collide, and
evaluation never needs to guess which base model an adapter was trained on.

Every script (training and evaluation, Python and Slurm) accepts `--model`
(or the `MODEL` environment variable in the `.sh` wrappers) to pick the base
model, e.g.:

```shell
python train_lora_math.py --model qwen3-1.7b-base --lora lora --rank 8 --alpha 16 --lr 1e-4 --seed 0
# or, on Slurm:
sbatch --export=ALL,MODEL=qwen3-1.7b-base,LR=1e-4,SEED=0 train_lora_math.sh
```

All four methods are evaluated on GSM8K by the same script,
`evaluation/eval_gsm8k.py` (Slurm wrapper: `eval_gsm8k.sh`). It auto-detects
whether a checkpoint is a PEFT adapter (LoRA, LoRA-Pro, Metric LoRA-Pro) or a
full model (full fine-tuning), loads the right base model automatically from
`metadata.json`, and applies the same prompting/generation/answer-extraction
protocol regardless of method:

```shell
torchrun --standalone --nproc_per_node=1 evaluation/eval_gsm8k.py \
    --checkpoint ./checkpoints/qwen3-1.7b-base/math/lora/r8_a16_lr0.0001_seed0
# or, on Slurm:
sbatch --export=ALL,METHOD=lora,MODEL=qwen3-1.7b-base,LR=1e-4,SEED=0 eval_gsm8k.sh
```

## Get Started

### Example

In LoRA-Pro, to ensure compatibility with DeepSpeed, we've integrated the Adam optimization process directly into DeepSpeed (in `DeepSpeed-0.15.1/deepspeed/runtime/zero/stage_1_and_2.py`). Therefore, in the TrainingArguments, **you need to set the optimizer to "sgd" to prevent parameters from being updated twice.**

#### 0. Training 

```python
# Define your LoRA


# Define TrainingArguments, keep optim as "sgd" here.
train_args = TrainingArguments(
	...,
    optim="sgd",
	...,
)

# LoRA-Pro Trainer
trainer = Trainer(
    model=model,
    train_dataset=datasets["train"],
    eval_dataset=datasets["eval"],
    tokenizer=tokenizer,
    args=train_args,
    data_collator=default_data_collator,
)

# Train
trainer.train()
```

### Reproduce Llama-2-7b results

#### 0. Training

The training scripts  can be found in `./scripts/llama-2-7b_transformers.sh`

For math task (now generalized to any local causal LM, defaulting to
Qwen3-1.7B-Base; pass `--model llama-2-7b` to reproduce the original setup),

```shell
torchrun --nproc_per_node=8 train_lora_math.py --model llama-2-7b --lora rslora-pro --seed 0
```

For code task,

```shell
torchrun --nproc_per_node=8 minimal_lora_llama2_code_transformers.py --lora rslora-pro --seed 0
```

For chat task,

```shell
torchrun --nproc_per_node=8 minimal_lora_llama2_chat_transformers.py --lora rslora-pro --seed 0
```

#### 1. Evaluation

For math task,

```shell
torchrun --nproc_per_node=8 evaluation/eval_llama-2_math_multi_gpus.py
```

For code task, we generate results with script below and evaluate its PASS@1 using [HumanEval](https://github.com/openai/human-eval), 

```shell
torchrun --nproc_per_node=8 evaluation/eval_llama-2_code_multi_gpus.py
```

For chat task, we use [FastChat](https://github.com/lm-sys/FastChat/tree/main) to generation and evaluate with GPT-4, please read their instruction.



## Citation
```latex
@inproceedings{wang2024lorapro,
  title={LoRA-Pro: Are Low-Rank Adapters Properly Optimized?},
  author={Wang, Zhengbo and Liang, Jian and He, Ran and Wang, Zilei and Tan, Tieniu},
  booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## Contact

If you have any question, feel free to contact 📫zhengbowang@mail.ustc.edu.cn.










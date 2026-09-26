"""LoRA-Pro fine-tuning on MetaMathQA100k with explicit scaling choice.

Works with any local causal-LM checkpoint (e.g. Qwen3-1.7B-Base or
Llama-2-7B) via ``transformers``' ``Auto*`` classes.

With --rs-scaling true (the paper setup), scale = alpha / sqrt(rank).
With --rs-scaling false, scale = alpha / rank. The selected value is used in
both PEFT's forward pass and the repository's custom DeepSpeed LoRA-Pro
gradient transformation. The requested seed controls adapter initialization,
Trainer RNG state, and distributed data sampling.
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import transformers
from peft import LoraConfig
import peft
from transformers import Trainer, TrainingArguments, default_data_collator

import peta
from peta.utils import (
    DEFAULT_MODEL,
    build_output_dir,
    build_run_name,
    resolve_model_path,
    save_run_metadata,
)
from peta.utils import TitledLog
import wandb


log = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_float32_matmul_precision("medium")

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model key (e.g. qwen3-1.7b-base, llama-2-7b) or a local path.",
    )
    parser.add_argument("--lora", default="lora-pro", type=str)
    parser.add_argument("--rank", default=8, type=int)
    parser.add_argument("--alpha", default=16, type=int)
    parser.add_argument(
        "--rs-scaling",
        "--rs_scaling",
        default=True,
        type=parse_bool,
        help="true: alpha/sqrt(rank); false: alpha/rank",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="e.g. sdpa (portable) or flash_attention_2 (requires flash-attn).",
    )
    parser.add_argument("--global-batch-size", default=32, type=int)
    parser.add_argument("--per-device-train-batch-size", default=2, type=int)
    parser.add_argument("--output-root", default="./checkpoints")
    parser.add_argument(
        "--deepspeed-config",
        default="./config/deepspeed_zero2_lorapro_paper.json",
    )
    args = parser.parse_args()

    if args.global_batch_size <= 0:
        parser.error("--global-batch-size must be positive")
    if args.per_device_train_batch_size <= 0:
        parser.error("--per-device-train-batch-size must be positive")
    return args


def seed_before_model_initialization(seed: int) -> None:
    """Seed all RNGs used before Trainer is constructed."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def verify_python_hash_seed(seed: int, global_rank: int) -> None:
    """PYTHONHASHSEED only takes effect if exported before Python starts."""

    configured = os.getenv("PYTHONHASHSEED")
    if configured != str(seed):
        raise RuntimeError(
            f"PYTHONHASHSEED={configured!r}; export PYTHONHASHSEED={seed} "
            "before torchrun (the supplied Slurm file does this)."
        )
    if global_rank == 0:
        print(f"PYTHONHASHSEED: {configured}", flush=True)


def verify_patched_deepspeed() -> Path:
    """Reject stock or stale DeepSpeed before allocating the model."""

    import deepspeed

    package = Path(deepspeed.__file__).resolve().parent
    stage2 = package / "runtime" / "zero" / "stage_1_and_2.py"
    if not stage2.is_file():
        raise RuntimeError(f"Cannot locate DeepSpeed ZeRO-2 source at {stage2}")

    source = stage2.read_text()
    required_markers = (
        "def lorapro_full_adjustment",
        'getattr(A, "_lorapro_scaling"',
        "is_first_lorapro_step = self.global_step == 1",
        "step = self.global_step",
    )
    missing = [marker for marker in required_markers if marker not in source]
    if missing:
        raise RuntimeError(
            "The imported DeepSpeed does not contain the complete LoRA-Pro "
            f"scaling/step patch. Source: {stage2}; missing: {missing}"
        )
    return stage2


def attach_and_verify_lorapro_scaling(
    model: torch.nn.Module,
    rank: int,
    alpha: int,
    rs_scaling: bool,
) -> Tuple[int, List[float]]:
    """Pass PEFT's real scaling to each corresponding LoRA-Pro A factor."""

    expected = (alpha / math.sqrt(rank)) if rs_scaling else (alpha / rank)
    values: List[float] = []
    count = 0

    for module in model.modules():
        if not all(hasattr(module, name) for name in ("lora_A", "lora_B", "scaling")):
            continue
        if "default" not in module.lora_A or "default" not in module.lora_B:
            continue

        actual = float(module.scaling["default"])
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(
                f"PEFT scaling is {actual}, but the selected formula gives {expected}"
            )
        module.lora_A["default"].weight._lorapro_scaling = actual
        values.append(actual)
        count += 1

    if count == 0:
        raise RuntimeError("No PEFT LoRA layers were found")
    return count, sorted(set(values))


def main() -> None:
    args = get_arguments()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    verify_python_hash_seed(args.seed, global_rank)
    seed_before_model_initialization(args.seed)
    deepspeed_source = verify_patched_deepspeed()

    micro_global_batch = args.per_device_train_batch_size * world_size
    if args.global_batch_size % micro_global_batch != 0:
        raise ValueError(
            "global batch size must be divisible by "
            "per-device batch size times world size"
        )
    gradient_accumulation_steps = args.global_batch_size // micro_global_batch
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least one")

    selected_scaling = (
        (args.alpha / math.sqrt(args.rank)) if args.rs_scaling else (args.alpha / args.rank)
    )
    scaling_formula = "alpha/sqrt(rank)" if args.rs_scaling else "alpha/rank"
    scaling_tag = "rs-scale" if args.rs_scaling else "standard-scale"
    method = f"{args.lora}-{scaling_tag}"

    run_name = build_run_name(seed=args.seed, lr=args.lr, rank=args.rank, alpha=args.alpha)
    output_dir = build_output_dir(
        method=method,
        run_name=run_name,
        model=args.model,
        output_root=args.output_root,
    )
    wandb_run_name = f"{args.model}_math_rank{args.rank}_{method}_lr{args.lr}_seed{args.seed}"

    wandb_enabled = os.getenv("WANDB_MODE", "online").lower() != "disabled"
    wandb_run = None
    if global_rank == 0:
        print(f"DeepSpeed source: {deepspeed_source}", flush=True)
        print(f"LoRA-Pro scaling formula: {scaling_formula}", flush=True)
        print(f"LoRA-Pro scaling factor: {selected_scaling}", flush=True)
        print(f"Requested seed: {args.seed}", flush=True)
        print(f"World size: {world_size}", flush=True)
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}", flush=True)
        if wandb_enabled:
            wandb_run = wandb.init(
                entity=os.getenv("WANDB_ENTITY") or None,
                project=os.getenv("WANDB_PROJECT", "Qwen3-1.7B-Math"),
                name=wandb_run_name,
                group="Transformers-Math",
                config={
                    "model": args.model,
                    "method": method,
                    "lora_type": args.lora,
                    "learning_rate": args.lr,
                    "seed": args.seed,
                    "data_seed": args.seed,
                    "lora_r": args.rank,
                    "lora_alpha": args.alpha,
                    "rs_scaling": args.rs_scaling,
                    "use_rslora": args.rs_scaling,
                    "scaling_formula": scaling_formula,
                    "scaling_factor": selected_scaling,
                    "global_batch_size": args.global_batch_size,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "epochs": 1,
                },
            )

    model_path = resolve_model_path(args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokens_added = 0
    if tokenizer.eos_token is None:
        tokens_added += tokenizer.add_special_tokens({"eos_token": "</s>"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )
    if tokens_added:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        use_rslora=args.rs_scaling,
    )
    model = peft.get_peft_model(model, lora_config)
    if model.peft_config["default"].use_rslora != args.rs_scaling:
        raise RuntimeError("PEFT's use_rslora value does not match --rs-scaling")

    # The repository's LoRA-Pro matrix operations are performed in float32.
    # Cast before attaching attributes, since Module.to() may replace Parameter
    # objects in some PyTorch configurations.
    for name, module in model.named_modules():
        if "lora_" in name:
            module.to(torch.float32)

    layer_count, scaling_values = attach_and_verify_lorapro_scaling(
        model, args.rank, args.alpha, args.rs_scaling
    )
    if global_rank == 0:
        print(f"LoRA layers: {layer_count}", flush=True)
        print(f"Scaling values passed to DeepSpeed: {scaling_values}", flush=True)
        model.print_trainable_parameters()

    with TitledLog("load datasets and dataloaders", log_fn=log.info):
        datasets = peta.tasks.load_meta_math()
        if len(datasets["train"]) != 100000:
            raise RuntimeError(
                f"Expected 100000 MetaMathQA training examples, got {len(datasets['train'])}"
            )
        preprocessor = peta.tasks.MetaMathQA100k_Preprocessor(
            tokenizer=tokenizer,
            tokenizer_kwargs={
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt",
                "max_length": 1024,
            },
        )
        datasets = datasets.map(
            preprocessor,
            batched=True,
            batch_size=1000,
            num_proc=1,
            desc="Running tokenizer on dataset",
        )

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir="./logs/transformers_logs",
        run_name=wandb_run_name,
        do_train=True,
        do_eval=False,
        num_train_epochs=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # The patched DeepSpeed code constructs LoRA-Pro's Adam-equivalent
        # update; its underlying optimizer must remain SGD.
        optim="sgd",
        bf16=True,
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_steps=1,
        report_to=["wandb"] if wandb_enabled else [],
        label_names=["labels"],
        ddp_find_unused_parameters=False,
        eval_strategy="no",
        save_strategy="no",
        seed=args.seed,
        data_seed=args.seed,
        # LoRA-Pro's gradient adjustment is implemented inside the patched
        # DeepSpeed ZeRO-2 optimizer (see verify_patched_deepspeed above), so
        # DeepSpeed must run even on a single GPU -- without it, training
        # silently degrades to plain SGD with no LoRA-Pro correction.
        deepspeed=args.deepspeed_config,
    )

    trainer = Trainer(
        model=model,
        train_dataset=datasets["train"],
        tokenizer=tokenizer,
        args=train_args,
        data_collator=default_data_collator,
    )
    trainer.train()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if global_rank == 0:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        save_run_metadata(
            output_dir,
            {
                "method": method,
                "model": args.model,
                "base_model_path": model_path,
                "lora_type": args.lora,
                "learning_rate": args.lr,
                "seed": args.seed,
                "data_seed": args.seed,
                "rank": args.rank,
                "alpha": args.alpha,
                "rs_scaling": args.rs_scaling,
                "use_rslora": args.rs_scaling,
                "scaling_formula": scaling_formula,
                "scaling_factor": selected_scaling,
                "global_batch_size": args.global_batch_size,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "epochs": 1,
                "deepspeed_source": str(deepspeed_source),
            },
        )
        print(f"Saved LoRA-Pro adapter at: {output_dir}", flush=True)
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()

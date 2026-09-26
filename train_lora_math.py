#!/usr/bin/env python
"""LoRA (and rsLoRA/DoRA) fine-tuning on MetaMathQA100k.

Works with any local causal-LM checkpoint (e.g. Qwen3-1.7B-Base or
Llama-2-7B) via ``transformers``' ``Auto*`` classes. The checkpoint and its
hyperparameters (rank, alpha, learning rate, seed) are saved under a
directory built by ``peta.utils.experiment``, and can be evaluated with
``evaluation/eval_gsm8k.py``.
"""
import argparse
import logging
import math
import os

import lightning as L
import torch
import transformers
from transformers import Trainer, TrainingArguments, default_data_collator

import peft
from peft import LoraConfig

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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model key (e.g. qwen3-1.7b-base, llama-2-7b) or a local path.",
    )
    parser.add_argument(
        "--lora",
        default="lora",
        type=str,
        help="Adapter method tag, e.g. lora, rslora, or dora.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--rank", default=8, type=int)
    parser.add_argument("--alpha", default=16, type=int)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="e.g. sdpa (portable) or flash_attention_2 (requires flash-attn).",
    )
    parser.add_argument("--global-batch-size", default=32, type=int)
    parser.add_argument("--per-device-train-batch-size", default=2, type=int)
    parser.add_argument("--output-root", default="./checkpoints")
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()

    log.info(f"set seed to {args.seed}")
    L.seed_everything(args.seed)
    set_seed(args.seed)

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    use_rslora = "rs" in args.lora
    scaling_factor = (
        (args.alpha / math.sqrt(args.rank)) if use_rslora else (args.alpha / args.rank)
    )

    run_name = build_run_name(seed=args.seed, lr=args.lr, rank=args.rank, alpha=args.alpha)
    output_dir = build_output_dir(
        method=args.lora,
        run_name=run_name,
        model=args.model,
        output_root=args.output_root,
    )

    wandb_enabled = os.getenv("WANDB_MODE", "online").lower() != "disabled"
    wandb_run = None
    if local_rank == 0 and wandb_enabled:
        wandb_run = wandb.init(
            entity=os.getenv("WANDB_ENTITY") or None,
            project=os.getenv("WANDB_PROJECT", "Qwen3-1.7B-Math"),
            name=f"{args.model}_math_rank{args.rank}_{args.lora}_lr{args.lr}_seed{args.seed}",
            group="Transformers-Math",
            config={
                "model": args.model,
                "lora_r": args.rank,
                "lora_alpha": args.alpha,
                "lora_type": args.lora,
                "learning_rate": args.lr,
                "seed": args.seed,
                "method": args.lora,
                "use_rslora": use_rslora,
                "scaling_factor": scaling_factor,
            },
        )
        print(f"W&B run: {wandb_run.url}", flush=True)

    # Step 0: load model and tokenizer
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

    # Step 1: Peft model
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        use_rslora=use_rslora,
    )

    model = peft.get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.lora not in ["lora", "rslora"]:
        for name, module in model.named_modules():
            if "lora_" in name:
                module.to(torch.float32)

    # Step 2: load dataset
    with TitledLog("load datasets and dataloaders", log_fn=log.info):
        datasets = peta.tasks.load_meta_math()

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

    gradient_accumulation_steps = args.global_batch_size // (
        world_size * args.per_device_train_batch_size
    )
    assert (
        args.per_device_train_batch_size * world_size * gradient_accumulation_steps
        == args.global_batch_size
    )

    # Step 3: Training Args
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir="./logs/transformers_logs",
        run_name=run_name,
        do_train=True,
        num_train_epochs=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        logging_steps=1,
        bf16=True,
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to=["wandb"] if wandb_enabled else [],
        label_names=["labels"],
        ddp_find_unused_parameters=False,
        do_eval=False,
        eval_strategy="no",
        save_strategy="no",
        seed=args.seed,
        data_seed=args.seed,
        deepspeed=None,
    )

    # Step 4: Trainer
    trainer = Trainer(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        tokenizer=tokenizer,
        args=train_args,
        data_collator=default_data_collator,
    )

    # Step 5: Train
    trainer.train()

    # Step 6: Save model
    if local_rank == 0:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        save_run_metadata(
            output_dir,
            {
                "method": args.lora,
                "model": args.model,
                "base_model_path": model_path,
                "learning_rate": args.lr,
                "seed": args.seed,
                "rank": args.rank,
                "alpha": args.alpha,
                "use_rslora": use_rslora,
                "scaling_factor": scaling_factor,
                "global_batch_size": args.global_batch_size,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "epochs": 1,
            },
        )
        print(f"Saving at path: {output_dir}", flush=True)

        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()

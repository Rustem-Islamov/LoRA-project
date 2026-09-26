#!/usr/bin/env python
"""Full-parameter fine-tuning on the LoRA-Pro MetaMathQA split.

Works with any local causal-LM checkpoint (e.g. Qwen3-1.7B-Base or
Llama-2-7B) via ``transformers``' ``Auto*`` classes.
"""

import argparse
import os
from pathlib import Path

import torch
import transformers
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

import peta
from peta.utils import (
    DEFAULT_MODEL,
    build_output_dir,
    build_run_name,
    resolve_model_path,
    save_run_metadata,
)


class WandbMetadataCallback(TrainerCallback):
    """Add experiment fields that are not native TrainingArguments."""

    def __init__(self, metadata):
        self.metadata = metadata

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero and "wandb" in args.report_to:
            import wandb

            if wandb.run is not None:
                wandb.config.update(self.metadata, allow_val_change=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model key (e.g. qwen3-1.7b-base, llama-2-7b) or a local path.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="e.g. sdpa (portable) or flash_attention_2 (requires flash-attn).",
    )
    parser.add_argument(
        "--deepspeed-config",
        default="./config/deepspeed_zero3_fullft_2gpu.json",
    )
    parser.add_argument("--output-root", default="./checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1.0)
    return parser.parse_args()


def main():
    cli = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    denominator = world_size * cli.per_device_batch_size
    if cli.global_batch_size % denominator != 0:
        raise ValueError(
            "global_batch_size must be divisible by "
            "WORLD_SIZE * per_device_batch_size"
        )
    gradient_accumulation_steps = cli.global_batch_size // denominator

    model_path = resolve_model_path(cli.model)
    if not (Path(model_path) / "config.json").is_file():
        raise FileNotFoundError(f"Base model not found at {model_path}")
    run_name = build_run_name(seed=cli.seed, lr=cli.lr)
    output_dir = build_output_dir(
        method="full-ft",
        run_name=run_name,
        model=cli.model,
        output_root=cli.output_root,
    )
    ds_path = cli.deepspeed_config
    if world_size > 1 and not Path(ds_path).is_file():
        raise FileNotFoundError(f"DeepSpeed config not found at {ds_path}")

    set_seed(cli.seed)
    wandb_run_name = f"{cli.model}_math_full-ft_lr{cli.lr:g}_seed{cli.seed}"
    use_wandb = os.environ.get("WANDB_MODE", "online").lower() != "disabled"

    # Construct TrainingArguments before from_pretrained. This lets the
    # Transformers/DeepSpeed integration activate ZeRO-3 model initialization.
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=wandb_run_name,
        do_train=True,
        do_eval=False,
        num_train_epochs=cli.epochs,
        per_device_train_batch_size=cli.per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        learning_rate=cli.lr,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        logging_dir="./logs/transformers_logs/full-ft",
        report_to=["wandb"] if use_wandb else [],
        save_strategy="no",
        deepspeed=ds_path if world_size > 1 else None,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        seed=cli.seed,
        data_seed=cli.seed,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    if tokenizer.eos_token is None:
        raise ValueError("The local tokenizer has no EOS token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Do not pass device_map with DeepSpeed. Every base-model parameter remains
    # trainable; there is deliberately no LoraConfig/get_peft_model call here.
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=cli.attn_implementation,
    )
    model.config.use_cache = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable != total:
        raise RuntimeError(f"Only {trainable:,}/{total:,} parameters are trainable")
    if local_rank == 0:
        print(f"Trainable parameters: {trainable:,}/{total:,}", flush=True)
        print(f"World size: {world_size}", flush=True)
        print(f"Per-device batch: {cli.per_device_batch_size}", flush=True)
        print(f"Gradient accumulation: {gradient_accumulation_steps}", flush=True)
        print(f"Global batch: {cli.global_batch_size}", flush=True)

    datasets = peta.tasks.load_meta_math()
    if len(datasets["train"]) != 100_000:
        raise RuntimeError(
            f"Expected 100000 MetaMathQA training examples, got {len(datasets['train'])}"
        )

    def preprocess(batch):
        combined = [
            prompt + " " + answer + tokenizer.eos_token
            for prompt, answer in zip(batch["x"], batch["y"])
        ]
        encoded = tokenizer(
            combined,
            padding="max_length",
            truncation=True,
            max_length=cli.max_length,
        )
        prompt_ids = tokenizer(
            batch["x"],
            truncation=True,
            max_length=cli.max_length,
        )["input_ids"]

        labels = []
        for ids, mask, prompt in zip(
            encoded["input_ids"], encoded["attention_mask"], prompt_ids
        ):
            item = list(ids)
            prompt_length = min(len(prompt), cli.max_length)
            item[:prompt_length] = [-100] * prompt_length
            item = [token if keep else -100 for token, keep in zip(item, mask)]
            labels.append(item)
        encoded["labels"] = labels
        return encoded

    # Only rank 0 creates the Arrow cache; the other rank waits and reuses it.
    with train_args.main_process_first(desc="tokenizing MetaMathQA"):
        datasets = datasets.map(
            preprocess,
            batched=True,
            batch_size=1000,
            remove_columns=datasets["train"].column_names,
            desc="Tokenizing MetaMathQA",
        )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=datasets["train"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[
            WandbMetadataCallback(
                {
                    "model": cli.model,
                    "method": "full-ft",
                    "lr": cli.lr,
                    "seed": cli.seed,
                    "global_batch_size": cli.global_batch_size,
                    "sequence_length": cli.max_length,
                }
            )
        ],
    )

    trainer.train()

    # All ranks must enter save_model because ZeRO-3 gathering is collective.
    model.config.use_cache = True
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
        save_run_metadata(
            output_dir,
            {
                "method": "full-ft",
                "model": cli.model,
                "base_model_path": model_path,
                "learning_rate": cli.lr,
                "seed": cli.seed,
                "data_seed": cli.seed,
                "global_batch_size": cli.global_batch_size,
                "per_device_train_batch_size": cli.per_device_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "sequence_length": cli.max_length,
                "epochs": cli.epochs,
            },
        )
        print(f"Saved full model to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

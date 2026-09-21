import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
import peft
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

import peta
import wandb

os.environ.setdefault("WANDB_SILENT", "true")

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj", "gate_proj",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora",
        choices=["lora", "lora-pro", "rslora-pro"],
        default="lora",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    if (
        not math.isfinite(args.lr) or args.lr <= 0
        or args.rank <= 0 or args.alpha <= 0
        or args.global_batch_size <= 0
        or args.per_device_batch_size <= 0
        or args.epochs <= 0
    ):
        parser.error("LR, rank, alpha, batch sizes, and epochs must be positive.")
    return args


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)

    is_pro = args.lora != "lora"
    use_rs = args.lora == "rslora-pro"

    # This extension uses the repository's existing two-GPU Pro setup.
    if is_pro and world_size != 2:
        raise ValueError("Use two GPUs for this repository's LoRA-Pro setup.")

    deepspeed_source = None
    if is_pro:
        from minimal_lorapro_paper_llama2_math_transformers import (
            verify_patched_deepspeed,
        )
        deepspeed_source = str(verify_patched_deepspeed())

    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise RuntimeError("Export PYTHONHASHSEED before starting torchrun.")

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    micro_global_batch = args.per_device_batch_size * world_size
    if args.global_batch_size % micro_global_batch:
        raise ValueError(
            "Global batch size must be divisible by GPUs × per-device batch."
        )
    accumulation = args.global_batch_size // micro_global_batch

    output = Path(args.output_dir)
    if (output / "adapter_config.json").exists():
        raise FileExistsError(f"An adapter already exists in {output}")

    wandb_run = None
    wandb_enabled = os.getenv("WANDB_MODE", "online").lower() != "disabled"
    if global_rank == 0 and wandb_enabled:
        wandb_run = wandb.init(
            entity=os.getenv("WANDB_ENTITY") or None,
            project=os.getenv("WANDB_PROJECT", "LLAMA-2-7B"),
            name=run_name,
            group="Transformers-Math",
            config={
                "method": args.lora,
                "learning_rate": args.lr,
                "seed": args.seed,
                "data_seed": args.seed,
                "rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "use_rslora": True,
                "scaling_formula": "alpha/sqrt(rank)",
                "scaling_factor": LORA_ALPHA / math.sqrt(LORA_RANK),
                "m_x_averaging": args.m_x_averaging,
                "m_x_damping": args.m_x_damping,
                "m_x_scale_clip": args.m_x_scale_clip,
                "metric_tag": run_metric_tag,
                "metric_definition": "diagonal EMA of mean squared adapter inputs",
                "metric_update_timing": "once per optimizer step",
                "metric_requires_grad": False,
                "global_batch_size": args.global_batch_size,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "epochs": 1,
            },
        )

    base_model = "./models/llama-2-7b"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.eos_token_id is None:
        raise ValueError("The local Llama tokenizer must define an EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank},
    )
    model.config.use_cache = False

    model = peft.get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
            use_rslora=use_rs,
        ),
    )

    # Use float32 adapter parameters consistently across methods.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    expected_scale = (
        args.alpha / math.sqrt(args.rank)
        if use_rs else args.alpha / args.rank
    )
    layer_count = 0
    for module in model.modules():
        if not hasattr(module, "lora_A"):
            continue
        if "default" not in module.lora_A:
            continue
        actual_scale = float(module.scaling["default"])
        if not math.isclose(actual_scale, expected_scale):
            raise RuntimeError("Unexpected PEFT adapter scaling.")
        if is_pro:
            # Read by the repository's patched DeepSpeed optimizer.
            module.lora_A["default"].weight._lorapro_scaling = actual_scale
        layer_count += 1
    if layer_count == 0:
        raise RuntimeError("No LoRA layers were created.")

    # Explicitly keep all Pro factors in one optimizer parameter group.
    optimizer = None
    if is_pro:
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            momentum=0.0,
            weight_decay=0.0,
        )

    ds_config = None
    if is_pro:
        ds_config = {
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 50_000_000,
            },
            "bf16": {"enabled": "auto"},
            "zero_allow_untested_optimizer": True,
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
        }

    training_args = TrainingArguments(
        output_dir=str(output),
        run_name=output.name,
        do_train=True,
        do_eval=False,
        evaluation_strategy="no",
        save_strategy="no",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=accumulation,
        learning_rate=args.lr,
        optim="sgd" if is_pro else "adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to=["wandb"] if wandb_enabled else [],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=10,
        report_to=[],
        label_names=["labels"],
        deepspeed=ds_config,
    )

    datasets = peta.tasks.load_codefeedback()
    raw_train = datasets["train"]
    if len(raw_train) != 100_000:
        raise RuntimeError(
            f"Expected 100000 training examples, got {len(raw_train)}."
        )

    preprocessor = peta.tasks.CodeFeedback100k_Preprocessor(
        tokenizer=tokenizer,
        tokenizer_kwargs={
            "padding": "max_length",
            "truncation": True,
            "max_length": 1024,
        },
    )
    train_dataset = raw_train.map(
        preprocessor,
        batched=True,
        batch_size=1000,
        num_proc=1,
        remove_columns=raw_train.column_names,
        desc="Tokenizing CodeFeedback",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        # Handles differing lengths across preprocessing batches.
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
        ),
        optimizers=(optimizer, None),
    )

    if trainer.is_world_process_zero():
        model.print_trainable_parameters()
        print({
            "method": args.lora,
            "seed": args.seed,
            "lr": args.lr,
            "rank": args.rank,
            "alpha": args.alpha,
            "scaling": expected_scale,
            "optimizer": training_args.optim,
            "global_batch_size": args.global_batch_size,
            "gradient_accumulation_steps": accumulation,
            "deepspeed_source": deepspeed_source,
        }, flush=True)

    trainer.train()

    if dist.is_initialized():
        dist.barrier()

    if trainer.is_world_process_zero():
        output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        metadata = {
            **vars(args),
            "base_model": base_model,
            "use_rslora": use_rs,
            "scaling": expected_scale,
            "world_size": world_size,
            "gradient_accumulation_steps": accumulation,
            "deepspeed_source": deepspeed_source,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
        }
        (output / "training_config.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        (output / "training_arguments.json").write_text(
            training_args.to_json_string()
        )
        (output / "TRAINING_COMPLETE").touch()

        if wandb_run is not None:
            wandb_run.summary["optimizer_steps"] = int(trainer.state.global_step)
            wandb_run.summary["metric_updates_per_layer"] = int(
                trainer.state.global_step
            )
            wandb.finish()

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
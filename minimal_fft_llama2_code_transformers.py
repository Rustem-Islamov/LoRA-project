import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments, set_seed,
)
import peta


def barrier():
    if dist.is_initialized():
        dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--deepspeed-config",
        default="./config/deepspeed_zero3_fullft_2gpu.json",
    )
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()
    if (
        not math.isfinite(args.lr) or args.lr <= 0
        or args.global_batch_size <= 0 or args.per_device_batch_size <= 0
        or not math.isfinite(args.epochs) or args.epochs <= 0
        or args.max_steps == 0 or args.max_steps < -1
    ):
        parser.error("Use positive settings; max-steps must be -1 or positive.")
    return args


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise RuntimeError("Export PYTHONHASHSEED before starting torchrun.")

    micro_batch = world_size * args.per_device_batch_size
    if args.global_batch_size % micro_batch:
        raise ValueError("Global batch must be divisible by GPUs × per-device batch.")
    accumulation = args.global_batch_size // micro_batch

    base_model = "./models/llama-2-7b"
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")

    ds_path = Path(args.deepspeed_config)
    zero = json.loads(ds_path.read_text()).get("zero_optimization", {})
    if zero.get("stage") != 3:
        raise ValueError("This FFT trainer expects ZeRO-3.")
    if not zero.get("stage3_gather_16bit_weights_on_model_save"):
        raise ValueError("Enable ZeRO-3 16-bit weight gathering on save.")

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    # Construct before loading the model to enable ZeRO-3 initialization.
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
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=1,
        report_to=[],
        label_names=["labels"],
        deepspeed=str(ds_path),
        save_safetensors=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # DeepSpeed controls placement; do not set device_map.
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    if any(not p.requires_grad for p in model.parameters()):
        raise RuntimeError("FFT requires every model parameter to be trainable.")

    with training_args.main_process_first(desc="Preparing CodeFeedback"):
        raw_train = peta.tasks.load_codefeedback()["train"]
        if len(raw_train) != 100_000:
            raise RuntimeError(f"Expected 100000 examples, got {len(raw_train)}.")
        preprocessor = peta.tasks.CodeFeedback100k_Preprocessor(
            tokenizer=tokenizer,
            tokenizer_kwargs={
                "padding": "max_length", "truncation": True, "max_length": 1024,
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
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100,
        ),
    )
    if trainer.is_world_process_zero():
        total = sum(getattr(p, "ds_numel", p.numel()) for p in model.parameters())
        print(f"FFT trainable parameters: {total:,}", flush=True)
        print(
            f"GPUs={world_size}, global batch={args.global_batch_size}, "
            f"accumulation={accumulation}, max_steps={args.max_steps}",
            flush=True,
        )

    trainer.train()
    barrier()
    model.config.use_cache = True

    # All ranks participate in ZeRO-3 weight gathering.
    trainer.save_model(str(output))
    barrier()

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output)
        weight_files = (
            "model.safetensors", "model.safetensors.index.json",
            "pytorch_model.bin", "pytorch_model.bin.index.json",
        )
        if not (output / "config.json").is_file() or not any(
            (output / name).is_file() for name in weight_files
        ):
            raise RuntimeError("A reloadable full checkpoint was not saved.")

        metadata = {
            **vars(args),
            "method": "full-ft",
            "checkpoint_type": "full",
            "base_model": base_model,
            "rank": None,
            "alpha": None,
            "world_size": world_size,
            "gradient_accumulation_steps": accumulation,
            "completed_steps": trainer.state.global_step,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }
        (output / "training_config.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        (output / "training_arguments.json").write_text(
            training_args.to_json_string()
        )
        (output / "TRAINING_COMPLETE").touch()
        print(f"Saved FFT checkpoint: {output}", flush=True)
    barrier()


if __name__ == "__main__":
    main()
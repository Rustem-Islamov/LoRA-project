#!/usr/bin/env python3
"""Evaluate a saved Llama-2 PEFT adapter on GSM8K.

This preserves the evaluation protocol used by the LoRA-Pro repository for
its Table 2 math experiment while fixing its ``batch_decode(**outputs)``
compatibility bug and making the checkpoint/seed configurable.

Launch this file with torchrun. Each rank loads one complete model replica on
one GPU and evaluates a non-overlapping, contiguous part of the GSM8K test set.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM


# Keep the whitespace identical to the released LoRA-Pro evaluator.
PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA, DoRA, or LoRA-Pro adapter on GSM8K."
    )
    parser.add_argument(
        "--base-model",
        default="./models/llama-2-7b",
        help="Local Llama-2-7B base-model directory.",
    )
    parser.add_argument(
        "--dataset-path",
        default="./data/gsm8k/main",
        help="Local GSM8K dataset-script directory.",
    )
    parser.add_argument(
        "--method",
        default=os.environ.get("EVAL_METHOD", "lora-pro-full"),
        help="Adapter method directory, e.g. lora, dora, or lora-pro-full.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("EVAL_SEED", "0")),
    )
    parser.add_argument(
        "--learning-rate-directory",
        default=os.environ.get("EVAL_LR", "2e-05"),
        help="Directory name used when training, normally 2e-05 for Table 2.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Explicit adapter directory. Overrides method/LR/seed path construction.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Result path. Defaults to logs/eval/gsm8k_<method>_lr<lr>_seed<seed>.json.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def extract_gsm_num(text: str) -> int:
    """Apply the exact answer rule used by the released evaluator.

    It selects the first sequence of digits following ``####`` and assigns 0
    when no valid marker is found. Keeping this behavior is important when
    comparing the resulting accuracy with Table 2.
    """
    match = re.search(r"####\s*(\d+)", text)
    result = match.group(1) if match else ""

    try:
        return int(result.replace(",", ""))
    except (TypeError, ValueError):
        return 0


def split_bounds(total_size: int, rank: int, world_size: int) -> tuple[int, int]:
    per_process_size = math.ceil(total_size / world_size)
    start_index = rank * per_process_size
    end_index = min(start_index + per_process_size, total_size)
    return start_index, end_index


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.adapter_path is None:
        adapter_path = Path(
            "./logs/transformers/llama-2-7b/math"
        ) / args.learning_rate_directory / args.method / str(args.seed)
    else:
        adapter_path = Path(args.adapter_path)

    if args.output_json is None:
        output_path = Path(
            f"./logs/eval/gsm8k_{args.method}_"
            f"lr{args.learning_rate_directory}_seed{args.seed}.json"
        )
    else:
        output_path = Path(args.output_json)

    return adapter_path, output_path


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    adapter_path, output_path = resolve_paths(args)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this inside a GPU Slurm job.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")

    base_model_path = Path(args.base_model)
    dataset_path = Path(args.dataset_path)

    if not (base_model_path / "config.json").is_file():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"GSM8K dataset not found: {dataset_path}")
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter configuration not found: {adapter_path}")
    if not any(
        (adapter_path / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError(f"Adapter weights not found: {adapter_path}")

    if rank == 0:
        print("=" * 72, flush=True)
        print(f"Base model:  {base_model_path}", flush=True)
        print(f"Adapter:     {adapter_path}", flush=True)
        print(f"Dataset:     {dataset_path}", flush=True)
        print(f"Method:      {args.method}", flush=True)
        print(f"Seed:        {args.seed}", flush=True)
        print(f"World size:  {world_size}", flush=True)
        print(f"Output:      {output_path}", flush=True)
        print("=" * 72, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        padding_side="left",
        local_files_only=True,
    )

    model = LlamaForCausalLM.from_pretrained(
        str(base_model_path),
        max_length=1024,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        local_files_only=True,
    )

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig.from_pretrained(
        str(adapter_path),
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        config=lora_config,
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()

    dataset = load_dataset(str(dataset_path), split="test")
    if len(dataset) != 1319:
        raise RuntimeError(
            f"Expected 1,319 GSM8K test examples but found {len(dataset):,}."
        )

    start_index, end_index = split_bounds(len(dataset), rank, world_size)
    batch_starts = range(start_index, end_index, args.batch_size)

    if rank == 0:
        batch_starts = tqdm(
            batch_starts,
            total=math.ceil((end_index - start_index) / args.batch_size),
            desc="Evaluating GSM8K",
        )

    local_results: List[Dict[str, Any]] = []

    for batch_start in batch_starts:
        batch_end = min(batch_start + args.batch_size, end_index)
        examples = [dataset[index] for index in range(batch_start, batch_end)]
        prompts = [
            PROMPT_TEMPLATE.format(instruction=example["question"]) + " "
            for example in examples
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            max_length=args.max_input_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

        # The released file used batch_decode(**outputs), which is incompatible
        # with current Transformers. Decoding outputs.sequences is the intended
        # operation and retains the full prompt-plus-completion text.
        decoded_predictions = tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
        )

        for offset, (example, decoded_prediction) in enumerate(
            zip(examples, decoded_predictions)
        ):
            prediction = extract_gsm_num(decoded_prediction)
            reference = extract_gsm_num(example["answer"])
            local_results.append(
                {
                    "index": batch_start + offset,
                    "question": example["question"],
                    "reference_text": example["answer"],
                    "generated_text": decoded_prediction,
                    "prediction": prediction,
                    "reference": reference,
                    "correct": prediction == reference,
                }
            )

    gathered_results: List[Any] = [None] * world_size
    dist.all_gather_object(gathered_results, local_results)

    if rank == 0:
        results = [
            item
            for process_results in gathered_results
            for item in process_results
        ]
        results.sort(key=lambda item: item["index"])

        correct = sum(item["correct"] for item in results)
        accuracy = correct / len(results)
        accuracy_percent = 100.0 * accuracy

        payload = {
            "protocol": "LoRA-Pro Table 2 released GSM8K evaluator",
            "method": args.method,
            "seed": args.seed,
            "learning_rate_directory": args.learning_rate_directory,
            "base_model": str(base_model_path),
            "adapter_path": str(adapter_path),
            "dataset_path": str(dataset_path),
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "num_examples": len(results),
            "num_correct": correct,
            "accuracy": accuracy_percent,
            "predictions": results,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, ensure_ascii=False)

        print("=" * 72, flush=True)
        print(f"Evaluate seed {args.seed} results:", flush=True)
        print(f"Test samples: {len(results)}", flush=True)
        print(f"Correct:      {correct}", flush=True)
        print(f"Final Accuracy: {accuracy_percent:.6f}", flush=True)
        print(f"Saved results: {output_path}", flush=True)
        print("=" * 72, flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

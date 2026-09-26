#!/usr/bin/env python3
"""Evaluate any of the four fine-tuning methods on GSM8K.

One evaluator is shared by LoRA, LoRA-Pro, full fine-tuning, and Metric
LoRA-Pro: it auto-detects whether ``--checkpoint`` is a PEFT adapter (has
``adapter_config.json``) or a full model, loads the matching base model with
``transformers``' ``Auto*`` classes (so it works for Qwen3-1.7B-Base,
Llama-2-7B, or any other local causal LM), and applies the same prompting,
generation, and answer-extraction protocol to every method.

Launch this file with torchrun. Each rank loads one complete model replica on
one GPU and evaluates a disjoint, evenly-strided slice of the GSM8K test set.
"""

import argparse
import json
import os
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from peta.utils import (
    DEFAULT_MODEL,
    default_eval_output_path,
    load_run_metadata,
    resolve_model_path,
)


PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    r"(?:\s*/\s*[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+))?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA, LoRA-Pro, full fine-tuning, or Metric "
        "LoRA-Pro checkpoint on GSM8K."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Adapter directory (has adapter_config.json) or full-model "
        "directory saved by one of the train_*.py scripts.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model key or path for an adapter checkpoint. Defaults to "
        "the base_model_path recorded in the checkpoint's metadata.json, "
        f"falling back to {DEFAULT_MODEL!r}. Ignored for a full model.",
    )
    parser.add_argument("--dataset-path", default="./data/gsm8k/main")
    parser.add_argument(
        "--method",
        default=None,
        help="Label recorded in the output JSON. Defaults to the method in "
        "metadata.json, falling back to the checkpoint directory name.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="e.g. sdpa (portable) or flash_attention_2 (requires flash-attn).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Result path. Defaults to a path mirroring the checkpoint's "
        "directory under ./logs/eval.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_input_length <= 0 or args.max_new_tokens <= 0:
        parser.error("generation lengths must be positive")
    return args


def normalize_number(candidate: str) -> Optional[str]:
    cleaned = candidate.strip().replace(",", "").replace("$", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    try:
        if "/" in cleaned:
            value = Fraction(cleaned)
            if value.denominator == 1:
                return str(value.numerator)
            decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
        else:
            decimal_value = Decimal(cleaned)
        if not decimal_value.is_finite():
            return None
        if decimal_value == decimal_value.to_integral_value():
            return str(int(decimal_value))
        normalized = format(decimal_value.normalize(), "f")
        return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def last_number(text: str) -> Optional[str]:
    matches = NUMBER_PATTERN.findall(text)
    for candidate in reversed(matches):
        normalized = normalize_number(candidate)
        if normalized is not None:
            return normalized
    return None


def extract_reference(answer: str) -> Optional[str]:
    marker = re.search(r"####\s*(.+?)\s*$", answer, flags=re.DOTALL)
    return last_number(marker.group(1) if marker else answer)


def extract_prediction(completion: str) -> Optional[str]:
    # Prefer the answer conventions used by GSM8K and MetaMathQA, then fall
    # back to the final number anywhere in the generated completion.
    patterns = (
        r"####\s*(.+?)(?:\n|$)",
        r"(?i)the\s+answer\s+is\s*:?\s*(.+?)(?:\n|$)",
        r"(?i)final\s+answer\s*:?\s*(.+?)(?:\n|$)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, completion)
        if matches:
            answer = last_number(matches[-1])
            if answer is not None:
                return answer
    return last_number(completion)


def local_indices(dataset_size: int, rank: int, world_size: int) -> List[int]:
    # Strided partitioning has no padding or duplicated examples.
    return list(range(rank, dataset_size, world_size))


def is_adapter_checkpoint(checkpoint: Path) -> bool:
    return (checkpoint / "adapter_config.json").is_file()


def resolve_base_model(checkpoint: Path, cli_base_model: Optional[str], rank: int) -> str:
    if cli_base_model is not None:
        return resolve_model_path(cli_base_model)

    metadata = load_run_metadata(checkpoint)
    if metadata is not None and metadata.get("base_model_path"):
        return metadata["base_model_path"]

    if rank == 0:
        print(
            f"WARNING: no --base-model given and no metadata.json found in "
            f"{checkpoint}; falling back to {DEFAULT_MODEL!r}.",
            flush=True,
        )
    return resolve_model_path(DEFAULT_MODEL)


def resolve_method_label(checkpoint: Path, cli_method: Optional[str]) -> str:
    if cli_method is not None:
        return cli_method
    metadata = load_run_metadata(checkpoint)
    if metadata is not None and metadata.get("method"):
        return metadata["method"]
    return checkpoint.name


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this inside a GPU job.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"GSM8K dataset not found: {dataset_path}")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    adapter_mode = is_adapter_checkpoint(checkpoint)
    method = resolve_method_label(checkpoint, args.method)

    # For an adapter, the weights to load are the base model's; for a full
    # fine-tuned checkpoint, the checkpoint itself already contains the
    # trained weights and must be loaded directly, never the original base
    # model it started from.
    if adapter_mode:
        base_model_path = resolve_base_model(checkpoint, args.base_model, rank)
        if not (Path(base_model_path) / "config.json").is_file():
            raise FileNotFoundError(f"Base model not found: {base_model_path}")
        model_source = base_model_path
    else:
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(
                f"Full-model config.json not found in: {checkpoint}"
            )
        base_model_path = None
        model_source = str(checkpoint)

    output_path = (
        Path(args.output_json)
        if args.output_json is not None
        else default_eval_output_path(checkpoint)
    )

    if rank == 0:
        print("=" * 72, flush=True)
        print(f"Checkpoint:  {checkpoint}", flush=True)
        print(f"Kind:        {'PEFT adapter' if adapter_mode else 'full model'}", flush=True)
        if adapter_mode:
            print(f"Base model:  {base_model_path}", flush=True)
        print(f"Method:      {method}", flush=True)
        print(f"Dataset:     {dataset_path}", flush=True)
        print(f"World size:  {world_size}", flush=True)
        print(f"Output:      {output_path}", flush=True)
        print("=" * 72, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        padding_side="left",
        local_files_only=True,
    )
    if tokenizer.eos_token is None:
        raise RuntimeError("The tokenizer has no EOS token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        local_files_only=True,
    )
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    if adapter_mode:
        if not any(
            (checkpoint / filename).is_file()
            for filename in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            raise FileNotFoundError(f"Adapter weights not found: {checkpoint}")
        model = PeftModel.from_pretrained(
            model,
            str(checkpoint),
            is_trainable=False,
            local_files_only=True,
        )

    model.config.use_cache = True
    # Checkpoints may carry sampling defaults (temperature, top_p). Clear them
    # because evaluation is greedy and deterministic for every method.
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.eval()

    dataset = load_dataset(str(dataset_path), split="test")
    if len(dataset) != 1319:
        raise RuntimeError(
            f"Expected 1,319 GSM8K test examples but found {len(dataset):,}."
        )

    indices = local_indices(len(dataset), rank, world_size)
    starts = range(0, len(indices), args.batch_size)
    iterator = (
        tqdm(starts, desc="Evaluating GSM8K", total=len(starts))
        if rank == 0
        else starts
    )

    local_results: List[Dict[str, Any]] = []

    for start in iterator:
        batch_indices = indices[start : start + args.batch_size]
        questions = [dataset[index]["question"] for index in batch_indices]
        raw_references = [dataset[index]["answer"] for index in batch_indices]
        prompts = [
            PROMPT_TEMPLATE.format(instruction=question) + " " for question in questions
        ]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
            return_token_type_ids=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

        # Left-padded inputs share a common width, so every completion starts
        # at that width regardless of its own prompt's length.
        completion_ids = generated[:, input_ids.shape[1] :]
        completions = tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        for index, question, raw_reference, completion in zip(
            batch_indices, questions, raw_references, completions
        ):
            reference = extract_reference(raw_reference)
            prediction = extract_prediction(completion)
            if reference is None:
                raise RuntimeError(f"Could not parse GSM8K reference at index {index}")
            local_results.append(
                {
                    "index": index,
                    "question": question,
                    "reference": reference,
                    "prediction": prediction,
                    "correct": prediction == reference,
                    "completion": completion,
                }
            )

    gathered: List[Optional[List[Dict[str, Any]]]] = [None] * world_size
    dist.all_gather_object(gathered, local_results)

    if rank == 0:
        results = [item for rank_results in gathered for item in rank_results]
        results.sort(key=lambda item: item["index"])
        result_indices = [item["index"] for item in results]
        if result_indices != list(range(len(dataset))):
            raise RuntimeError("Distributed evaluation produced missing or duplicate examples")

        correct = sum(item["correct"] for item in results)
        accuracy = correct / len(results)

        payload = {
            "checkpoint": str(checkpoint.resolve()),
            "kind": "adapter" if adapter_mode else "full_model",
            "method": method,
            "base_model": str(Path(base_model_path).resolve()) if base_model_path else None,
            "dataset_path": str(dataset_path.resolve()),
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "generation": {"do_sample": False, "num_beams": 1},
            "num_examples": len(results),
            "num_correct": correct,
            "accuracy": accuracy,
            "accuracy_percent": 100.0 * accuracy,
            "results": results,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        temporary_path.replace(output_path)

        print("=" * 72, flush=True)
        print(f"Examples:       {len(results)}", flush=True)
        print(f"Correct:        {correct}", flush=True)
        print(f"GSM8K accuracy: {100.0 * accuracy:.4f}%", flush=True)
        print(f"Results written to: {output_path}", flush=True)
        print("=" * 72, flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

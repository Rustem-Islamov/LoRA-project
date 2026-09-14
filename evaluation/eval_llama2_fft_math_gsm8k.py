#!/usr/bin/env python
"""Deterministic two-GPU GSM8K evaluation for a full Llama-2 checkpoint."""

import argparse
import json
import os
import random
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import transformers
from datasets import load_dataset
from tqdm.auto import tqdm


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", default="./data/gsm8k/main")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", default="unknown")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_input_length <= 0 or args.max_new_tokens <= 0:
        parser.error("generation lengths must be positive")
    return args


def seed_process(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_full_model_checkpoint(model_path: Path) -> None:
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing model config: {model_path / 'config.json'}")
    if (model_path / "adapter_config.json").is_file():
        raise RuntimeError(
            f"{model_path} contains adapter_config.json; expected a full-model checkpoint"
        )

    weight_candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if not any((model_path / name).is_file() for name in weight_candidates):
        available = sorted(path.name for path in model_path.iterdir())
        raise FileNotFoundError(
            f"No full-model weights found in {model_path}. Files: {available}"
        )


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
    # back to the final number in the generated completion only.
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


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    seed_process(args.seed)

    model_path = Path(args.model_path)
    require_full_model_checkpoint(model_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        local_files_only=True,
    )
    if tokenizer.eos_token is None:
        raise RuntimeError("The saved tokenizer has no EOS token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank},
        local_files_only=True,
    )
    model.config.use_cache = True
    # Some Llama checkpoints carry sampling defaults such as temperature=0.6
    # and top_p=0.9. Clear them because Table-2 evaluation uses one greedy,
    # deterministic completion per question.
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.eval()

    dataset = load_dataset(args.dataset_path, split="test")
    if len(dataset) != 1319:
        raise RuntimeError(f"Expected 1319 GSM8K test examples, got {len(dataset)}")

    indices = local_indices(len(dataset), rank, world_size)
    local_results: List[Dict] = []
    starts = range(0, len(indices), args.batch_size)
    iterator = tqdm(starts, desc="Evaluating GSM8K") if rank == 0 else starts

    for start in iterator:
        batch_indices = indices[start : start + args.batch_size]
        questions = [dataset[index]["question"] for index in batch_indices]
        raw_references = [dataset[index]["answer"] for index in batch_indices]
        prompts = [PROMPT_TEMPLATE.format(instruction=question) + " " for question in questions]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
            return_token_type_ids=False,
        )
        input_ids = encoded["input_ids"].to(local_rank)
        attention_mask = encoded["attention_mask"].to(local_rank)

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

        # Decoder-only generation returns prompt + completion. Since inputs are
        # left-padded to a common width, every completion starts at this width.
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

    gathered: List[Optional[List[Dict]]] = [None] * world_size
    dist.all_gather_object(gathered, local_results)

    if rank == 0:
        results = [item for rank_results in gathered for item in rank_results]
        results.sort(key=lambda item: item["index"])
        result_indices = [item["index"] for item in results]
        expected_indices = list(range(len(dataset)))
        if result_indices != expected_indices:
            raise RuntimeError("Distributed evaluation produced missing or duplicate examples")

        correct = sum(item["correct"] for item in results)
        accuracy = correct / len(results)
        output = {
            "method": "full-ft",
            "model_path": str(model_path.resolve()),
            "dataset_path": str(Path(args.dataset_path).resolve()),
            "seed": args.seed,
            "learning_rate": args.learning_rate,
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

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_path.open("w") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)
        temporary_path.replace(output_path)

        print(f"Examples: {len(results)}", flush=True)
        print(f"Correct: {correct}", flush=True)
        print(f"GSM8K accuracy: {100.0 * accuracy:.4f}%", flush=True)
        print(f"Results written to: {output_path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    evaluate(args)


if __name__ == "__main__":
    main()

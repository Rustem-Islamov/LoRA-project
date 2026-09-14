#!/usr/bin/env python3
"""Evaluate a corrected LoRA-Pro Llama-2 adapter on GSM8K.

The prompt, fixed-length input padding, greedy generation, and ``####`` answer
extraction match the released LoRA-Pro Table 2 evaluator. Each torchrun rank
loads one model replica and evaluates a disjoint part of the 1,319 examples.
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
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM


PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="./models/llama-2-7b")
    parser.add_argument("--dataset-path", default="./data/gsm8k/main")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--method", default="lora-pro-rs-scale")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate-directory", default="2e-05")
    parser.add_argument("--expected-rs-scaling", type=parse_bool, default=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_input_length < 1 or args.max_new_tokens < 1:
        parser.error("generation lengths must be positive")
    return args


def extract_gsm_num(text: str) -> int:
    """Use the released evaluator's answer extraction rule."""

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


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this inside a GPU job")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    # Evaluation is greedy, but seed every rank to prevent accidental
    # nondeterminism if generation settings are changed later.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    base_model_path = Path(args.base_model)
    adapter_path = Path(args.adapter_path)
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_json)

    require_file(base_model_path / "config.json", "base-model config")
    require_file(adapter_path / "adapter_config.json", "adapter config")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Missing GSM8K directory: {dataset_path}")
    if not any(
        (adapter_path / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError(f"Missing adapter weights in {adapter_path}")

    adapter_config = LoraConfig.from_pretrained(
        str(adapter_path),
        local_files_only=True,
    )
    actual_rs_scaling = bool(getattr(adapter_config, "use_rslora", False))
    if adapter_config.r != 8:
        raise RuntimeError(f"Expected LoRA rank 8, found {adapter_config.r}")
    if float(adapter_config.lora_alpha) != 16.0:
        raise RuntimeError(
            f"Expected LoRA alpha 16, found {adapter_config.lora_alpha}"
        )
    if actual_rs_scaling != args.expected_rs_scaling:
        raise RuntimeError(
            "Adapter scaling mismatch: "
            f"use_rslora={actual_rs_scaling}, "
            f"expected {args.expected_rs_scaling}"
        )

    if rank == 0:
        print("=" * 72, flush=True)
        print(f"Base model:       {base_model_path}", flush=True)
        print(f"Adapter:          {adapter_path}", flush=True)
        print(f"Dataset:          {dataset_path}", flush=True)
        print(f"Method:           {args.method}", flush=True)
        print(f"Training seed:    {args.seed}", flush=True)
        print(f"LoRA rank:        {adapter_config.r}", flush=True)
        print(f"LoRA alpha:       {adapter_config.lora_alpha}", flush=True)
        print(f"use_rslora:       {actual_rs_scaling}", flush=True)
        print(f"World size:       {world_size}", flush=True)
        print(f"Output:           {output_path}", flush=True)
        print("=" * 72, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        padding_side="left",
        local_files_only=True,
    )
    if tokenizer.eos_token is None:
        raise RuntimeError("The base tokenizer has no EOS token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        str(base_model_path),
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        config=adapter_config,
        is_trainable=False,
        local_files_only=True,
    )
    model.config.use_cache = True
    # The base checkpoint may contain sampling defaults. They are deliberately
    # disabled because the released evaluator was effectively greedy.
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.eval()

    dataset = load_dataset(str(dataset_path), split="test")
    if len(dataset) != 1319:
        raise RuntimeError(f"Expected 1,319 GSM8K examples, found {len(dataset):,}")

    start_index, end_index = split_bounds(len(dataset), rank, world_size)
    batch_starts = range(start_index, end_index, args.batch_size)
    iterator = (
        tqdm(
            batch_starts,
            total=math.ceil((end_index - start_index) / args.batch_size),
            desc="Evaluating GSM8K",
        )
        if rank == 0
        else batch_starts
    )
    local_results: List[Dict[str, Any]] = []

    for batch_start in iterator:
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
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        # Keep the full prompt-plus-completion text, matching the released
        # evaluator. Decoding outputs.sequences fixes its batch_decode bug.
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
        expected_indices = list(range(len(dataset)))
        if [item["index"] for item in results] != expected_indices:
            raise RuntimeError("Distributed evaluation lost or duplicated examples")

        correct = sum(item["correct"] for item in results)
        accuracy_percent = 100.0 * correct / len(results)
        payload = {
            "protocol": "LoRA-Pro Table 2 released GSM8K evaluator",
            "method": args.method,
            "seed": args.seed,
            "learning_rate_directory": args.learning_rate_directory,
            "base_model": str(base_model_path.resolve()),
            "adapter_path": str(adapter_path.resolve()),
            "dataset_path": str(dataset_path.resolve()),
            "adapter_config": {
                "r": adapter_config.r,
                "lora_alpha": adapter_config.lora_alpha,
                "use_rslora": actual_rs_scaling,
            },
            "generation": {"do_sample": False, "num_beams": 1},
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
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, ensure_ascii=False)
        temporary_path.replace(output_path)

        print("=" * 72, flush=True)
        print(f"Test samples:   {len(results)}", flush=True)
        print(f"Correct:        {correct}", flush=True)
        print(f"Final Accuracy: {accuracy_percent:.6f}", flush=True)
        print(f"Saved results:  {output_path}", flush=True)
        print("=" * 72, flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

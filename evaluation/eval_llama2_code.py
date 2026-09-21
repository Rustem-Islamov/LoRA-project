import argparse
import json
import re
from pathlib import Path

import torch
from human_eval.data import read_problems, write_jsonl
# from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def clean_completion(text):
    # Stop at a closing Markdown fence or a new top-level definition.
    # Preserve the generated function body's indentation.
    text = text.split("```", 1)[0]
    match = re.search(
        r"\n(?:def |class |async def |@|if __name__)",
        text,
    )
    if match:
        text = text[:match.start()]
    return text.rstrip() + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="./models/llama-2-7b")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter-path")
    source.add_argument("--model-path", help="Full fine-tuned checkpoint directory")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Batch size and generation length must be positive.")

    # Fixed across all training seeds. Generation itself is greedy.
    set_seed(0)
    torch.cuda.set_device(0)

    checkpoint = Path(args.adapter_path or args.model_path)
    if not (checkpoint / "TRAINING_COMPLETE").is_file():
        raise RuntimeError(f"Training did not finish successfully: {checkpoint}")

    if args.adapter_path:
        if not (checkpoint / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing adapter_config.json: {checkpoint}")
    else:
        if (checkpoint / "adapter_config.json").exists():
            raise ValueError("Use --adapter-path for a PEFT adapter checkpoint.")
        weight_files = (
            "model.safetensors", "model.safetensors.index.json",
            "pytorch_model.bin", "pytorch_model.bin.index.json",
        )
        if not (checkpoint / "config.json").is_file() or not any(
            (checkpoint / name).is_file() for name in weight_files
        ):
            raise FileNotFoundError(f"Missing full-model checkpoint: {checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, padding_side="left", local_files_only=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token.")
    tokenizer.pad_token = tokenizer.eos_token

    model_source = args.base_model if args.adapter_path else str(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
        local_files_only=True,
    )
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)

    model.eval()
    model.config.use_cache = True

    problems = read_problems()
    task_ids = sorted(
        problems,
        key=lambda task_id: int(task_id.rsplit("/", 1)[1]),
    )
    samples = []
    raw_samples = []

    for start in range(0, len(task_ids), args.batch_size):
        batch_ids = task_ids[start:start + args.batch_size]
        prompts = [problems[task_id]["prompt"] for task_id in batch_ids]

        # Never silently truncate a benchmark prompt.
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to("cuda:0")
        prompt_width = encoded["input_ids"].shape[1]

        if (
            prompt_width + args.max_new_tokens
            > model.config.max_position_embeddings
        ):
            raise ValueError("Prompt and completion exceed the context window.")

        output = model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decode_kwargs = {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        decoded_prompts = tokenizer.batch_decode(
            encoded["input_ids"], **decode_kwargs
        )
        decoded_full = tokenizer.batch_decode(output, **decode_kwargs)

        continuations = []
        for prefix, full in zip(decoded_prompts, decoded_full):
            if not full.startswith(prefix):
                raise RuntimeError("Decoded prompt changed at generation boundary.")
            continuations.append(full[len(prefix):])

        for task_id, text in zip(batch_ids, continuations):
            raw_samples.append({
                "task_id": task_id,
                "completion": text,
            })
            samples.append({
                "task_id": task_id,
                "completion": clean_completion(text),
            })

    if (
        len(samples) != len(problems)
        or len({sample["task_id"] for sample in samples}) != len(problems)
    ):
        raise RuntimeError("Missing or duplicate HumanEval predictions.")

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output_file), samples)
    write_jsonl(str(output_file.with_name("raw_samples.jsonl")), raw_samples)

    metadata = {
        "protocol": "raw-humaneval-prompt-greedy-v1",
        "do_sample": False,
        "num_beams": 1,
        "samples_per_task": 1,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "num_problems": len(problems),
        "checkpoint_type": "adapter" if args.adapter_path else "full",
        "checkpoint_path": str(checkpoint),
        "adapter_path": str(checkpoint) if args.adapter_path else None,
        "model_path": str(model_source),
        "postprocessing": "stop at fence or new top-level definition",
    }
    (output_file.parent / "generation_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
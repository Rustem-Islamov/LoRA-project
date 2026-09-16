import argparse
import json
import re
from pathlib import Path

import torch
from human_eval.data import read_problems, write_jsonl
from peft import PeftModel
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
    parser.add_argument("--adapter-path", required=True)
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

    adapter = Path(args.adapter_path)
    if not (adapter / "TRAINING_COMPLETE").is_file():
        raise RuntimeError(f"Training did not finish successfully: {adapter}")

    tokenizer = AutoTokenizer.from_pretrained(
        adapter,
        padding_side="left",
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token.")
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, adapter)
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
        continuations = tokenizer.batch_decode(
            output[:, prompt_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

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
        "adapter_path": str(adapter),
        "postprocessing": "stop at fence or new top-level definition",
    }
    (output_file.parent / "generation_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
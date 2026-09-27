"""Fine-tune with rsLoRA-Pro + diagonal M_x ("Metric LoRA-Pro").

Works with any local causal-LM checkpoint (e.g. Qwen3-1.7B-Base or
Llama-2-7B) via ``transformers``' ``Auto*`` classes.

M_x follows the Metric-LoRA notebook supplied with this experiment: it is an
EMA of per-input-feature squared activations.  P_x is the clipped, damped
inverse square root of the mean-normalized diagonal M_x.  Metric statistics
are detached, synchronized across data-parallel ranks, and committed once per
optimizer step.  P_x is therefore frozen throughout each forward/backward.

The accompanying DeepSpeed patch interprets C = A P_x as LoRA-Pro's effective
right factor.  At the end of training P_x is folded into A, so the saved PEFT
adapter can be evaluated with an unmodified PEFT model.
"""

import argparse
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)
import transformers

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
os.environ.setdefault("WANDB_SILENT", "true")
torch.set_float32_matmul_precision("medium")

METHOD_NAME = "metric-lorapro-mx-rs-scale"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]


class MxBatchContext:
    """Holds the current batch mask only for the duration of model.forward."""

    token_mask: Optional[torch.Tensor] = None

    @classmethod
    def set_mask(cls, attention_mask: Optional[torch.Tensor]) -> None:
        cls.token_mask = attention_mask

    @classmethod
    def clear(cls) -> None:
        cls.token_mask = None


@torch.no_grad()
def masked_square_sum_and_count(
    tensor: torch.Tensor,
    token_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return detached float32 per-feature square sums and token count."""

    values = tensor.detach().float().reshape(-1, tensor.shape[-1])
    if token_mask is not None and token_mask.numel() == values.shape[0]:
        weights = token_mask.detach().float().reshape(-1, 1)
        return (values.square() * weights).sum(dim=0), weights.sum().clamp_min(1.0)

    count = torch.tensor(
        float(values.shape[0]), device=values.device, dtype=torch.float32
    )
    return values.square().sum(dim=0), count


class DiagonalMxState(nn.Module):
    """Non-trainable diagonal M_x state for one LoRA-injected linear layer."""

    def __init__(
        self,
        in_features: int,
        device: torch.device,
        m_x_averaging: float,
        damping: float,
        scale_clip: float,
    ) -> None:
        super().__init__()
        self.m_x_averaging = float(m_x_averaging)
        self.damping = float(damping)
        self.scale_clip = float(scale_clip)
        self.stats_enabled = True

        self.register_buffer(
            "mx_ema", torch.ones(in_features, device=device, dtype=torch.float32)
        )
        self.register_buffer(
            "mx_updates", torch.zeros((), device=device, dtype=torch.long)
        )
        self.register_buffer(
            "mx_accum_sum",
            torch.zeros(in_features, device=device, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "mx_accum_count",
            torch.zeros((), device=device, dtype=torch.float32),
            persistent=False,
        )

    @torch.no_grad()
    def accumulate(self, x: torch.Tensor, token_mask: Optional[torch.Tensor]) -> None:
        if not self.stats_enabled:
            return
        square_sum, count = masked_square_sum_and_count(x, token_mask)
        self.mx_accum_sum.add_(square_sum.to(self.mx_accum_sum))
        self.mx_accum_count.add_(count.to(self.mx_accum_count))

    @torch.no_grad()
    def commit(self, global_sum: torch.Tensor, global_count: torch.Tensor) -> None:
        if global_count.item() > 0:
            estimate = global_sum / global_count.clamp_min(1.0)
            if self.mx_updates.item() == 0:
                self.mx_ema.copy_(estimate)
            else:
                self.mx_ema.mul_(self.m_x_averaging).add_(
                    estimate, alpha=1.0 - self.m_x_averaging
                )
            self.mx_updates.add_(1)
        self.mx_accum_sum.zero_()
        self.mx_accum_count.zero_()

    @torch.no_grad()
    def metric_scale(self) -> torch.Tensor:
        normalized = self.mx_ema / self.mx_ema.mean().clamp_min(1e-12)
        scale = torch.rsqrt(normalized + self.damping)
        return scale.clamp(min=1.0 / self.scale_clip, max=self.scale_clip).detach()


@dataclass
class MxBinding:
    name: str
    owner: nn.Module
    lora_a: nn.Module
    state: DiagonalMxState
    hook: torch.utils.hooks.RemovableHandle


def attach_mx_metrics(
    model: nn.Module,
    m_x_averaging: float,
    damping: float,
    scale_clip: float,
    rank: int,
    alpha: int,
) -> List[MxBinding]:
    """Attach one M_x state and adapter-input hook to every PEFT LoRA layer."""

    bindings: List[MxBinding] = []
    for name, module in list(model.named_modules()):
        if not all(hasattr(module, attr) for attr in ("lora_A", "lora_B", "scaling")):
            continue
        if "default" not in module.lora_A or "default" not in module.lora_B:
            continue

        lora_a = module.lora_A["default"]
        state = DiagonalMxState(
            in_features=lora_a.in_features,
            device=lora_a.weight.device,
            m_x_averaging=m_x_averaging,
            damping=damping,
            scale_clip=scale_clip,
        )
        if hasattr(module, "mx_metric"):
            raise RuntimeError(f"M_x state is already attached to {name}")
        module.add_module("mx_metric", state)

        # The patched LoRA-Pro optimizer reads these attributes from A.
        lora_a.weight._lorapro_mx_state = state
        actual_scaling = float(module.scaling["default"])
        expected_scaling = alpha / math.sqrt(rank)
        if not math.isclose(
            actual_scaling,
            expected_scaling,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"PEFT scaling is {actual_scaling}, expected {expected_scaling}"
            )
        lora_a.weight._lorapro_scaling = actual_scaling

        def pre_hook(
            _lora_a: nn.Module,
            inputs: Tuple[torch.Tensor, ...],
            metric_state: DiagonalMxState = state,
        ) -> Tuple[torch.Tensor, ...]:
            x = inputs[0]
            if _lora_a.training:
                metric_state.accumulate(x, MxBatchContext.token_mask)

            # metric_scale() is detached. Multiplication still propagates
            # gradients to x and A, but never into M_x.
            px = metric_state.metric_scale().to(device=x.device, dtype=x.dtype)
            return (x * px,) + tuple(inputs[1:])

        hook = lora_a.register_forward_pre_hook(pre_hook)
        bindings.append(MxBinding(name, module, lora_a, state, hook))

    if not bindings:
        raise RuntimeError("No PEFT LoRA layers were found for M_x attachment")
    if any(list(binding.state.parameters()) for binding in bindings):
        raise RuntimeError("M_x state must contain buffers only, not parameters")
    return bindings


@torch.no_grad()
def commit_all_mx_metrics(bindings: List[MxBinding]) -> None:
    """Synchronize all sufficient statistics using one collective per step."""

    pieces = [
        torch.cat((binding.state.mx_accum_sum, binding.state.mx_accum_count.reshape(1)))
        for binding in bindings
    ]
    packed = torch.cat(pieces)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    offset = 0
    for binding in bindings:
        width = binding.state.mx_accum_sum.numel()
        global_sum = packed[offset : offset + width]
        global_count = packed[offset + width]
        binding.state.commit(global_sum, global_count)
        offset += width + 1


class MxTrainer(Trainer):
    """Expose each batch's attention mask to the M_x forward hooks."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        MxBatchContext.set_mask(inputs.get("attention_mask"))
        try:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                **kwargs,
            )
        finally:
            MxBatchContext.clear()


class MxCommitCallback(TrainerCallback):
    def __init__(self, bindings: List[MxBinding]) -> None:
        self.bindings = bindings

    def on_step_end(self, args, state, control, **kwargs):
        # on_step_end fires once per optimizer step, after DeepSpeed has used
        # the same frozen P_x that was active for every accumulated forward.
        commit_all_mx_metrics(self.bindings)
        return control


@torch.no_grad()
def fold_mx_into_lora_a(bindings: List[MxBinding]) -> None:
    """Replace A by A P_x and remove hooks so ordinary PEFT can evaluate it."""

    for binding in bindings:
        px = binding.state.metric_scale().to(
            device=binding.lora_a.weight.device,
            dtype=binding.lora_a.weight.dtype,
        )
        binding.lora_a.weight.mul_(px.unsqueeze(0))
        binding.hook.remove()
        for attribute in ("_lorapro_mx_state", "_lorapro_scaling"):
            if hasattr(binding.lora_a.weight, attribute):
                delattr(binding.lora_a.weight, attribute)
        binding.owner._modules.pop("mx_metric", None)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def verify_python_hash_seed(seed: int, global_rank: int) -> None:
    configured = os.getenv("PYTHONHASHSEED")
    if configured != str(seed):
        raise RuntimeError(
            f"PYTHONHASHSEED={configured!r}; export PYTHONHASHSEED={seed} "
            "before torchrun starts"
        )
    if global_rank == 0:
        print(f"PYTHONHASHSEED: {configured}", flush=True)


def metric_hparams(averaging: float, damping: float, scale_clip: float) -> dict:
    """Extra hyperparameters folded into the run's directory name."""
    return {"mxavg": averaging, "damp": damping, "clip": scale_clip}


def verify_patched_deepspeed() -> Path:
    """Verify the source actually imported by Python, not just the checkout."""

    import deepspeed

    package = Path(deepspeed.__file__).resolve().parent
    stage2 = package / "runtime" / "zero" / "stage_1_and_2.py"
    if not stage2.is_file():
        raise RuntimeError(f"Cannot locate imported ZeRO-2 source at {stage2}")

    source = stage2.read_text()
    required_markers = (
        "def lorapro_full_adjustment",
        'getattr(A, "_lorapro_mx_state"',
        'getattr(A, "_lorapro_scaling"',
        "effective_A = A * px.unsqueeze(0)",
        "grad_A_effective_orin = grad_A_orin * px_inverse.unsqueeze(0)",
        "grad_A = grad_A * px_inverse.unsqueeze(0)",
    )
    missing = [marker for marker in required_markers if marker not in source]
    if missing:
        raise RuntimeError(
            "The imported DeepSpeed does not contain the complete M_x-aware "
            f"LoRA-Pro patch. Source: {stage2}; missing: {missing}"
        )
    return stage2


def run_metric_unit_test() -> None:
    """Fast CPU test of masking, EMA semantics, scaling, and frozen state."""

    state = DiagonalMxState(
        in_features=3,
        device=torch.device("cpu"),
        m_x_averaging=0.5,
        damping=1e-2,
        scale_clip=5.0,
    )
    first = torch.tensor([[[1.0, 2.0, 3.0], [99.0, 99.0, 99.0]]])
    mask = torch.tensor([[1, 0]])
    state.accumulate(first, mask)
    state.commit(state.mx_accum_sum.clone(), state.mx_accum_count.clone())
    torch.testing.assert_close(state.mx_ema, torch.tensor([1.0, 4.0, 9.0]))

    second = torch.tensor([[[3.0, 2.0, 1.0]]])
    state.accumulate(second, torch.ones(1, 1, dtype=torch.long))
    state.commit(state.mx_accum_sum.clone(), state.mx_accum_count.clone())
    torch.testing.assert_close(state.mx_ema, torch.tensor([5.0, 4.0, 5.0]))

    px = state.metric_scale()
    assert not px.requires_grad
    assert torch.isfinite(px).all()
    assert bool((px >= 0.2).all() and (px <= 5.0).all())
    assert int(state.mx_updates.item()) == 2
    print("M_x unit test passed", flush=True)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model key (e.g. qwen3-1.7b-base, llama-2-7b) or a local path.",
    )
    parser.add_argument("--lora", default=METHOD_NAME, choices=[METHOD_NAME])
    parser.add_argument("--rank", default=8, type=int)
    parser.add_argument("--alpha", default=16, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--m-x-averaging", default=0.90, type=float)
    parser.add_argument("--m-x-damping", default=1e-2, type=float)
    parser.add_argument("--m-x-scale-clip", default=5.0, type=float)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="e.g. sdpa (portable) or flash_attention_2 (requires flash-attn).",
    )
    parser.add_argument("--global-batch-size", default=32, type=int)
    parser.add_argument("--per-device-train-batch-size", default=2, type=int)
    parser.add_argument("--output-root", default="./checkpoints")
    parser.add_argument(
        "--deepspeed-config",
        default="./config/deepspeed_zero2_rslorapro_mx.json",
    )
    parser.add_argument("--metric-unit-test-only", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.m_x_averaging < 1.0:
        parser.error("--m-x-averaging must satisfy 0 <= value < 1")
    if args.m_x_damping <= 0.0:
        parser.error("--m-x-damping must be positive")
    if args.m_x_scale_clip < 1.0:
        parser.error("--m-x-scale-clip must be at least 1")
    return args


def main() -> None:
    args = get_arguments()
    if args.metric_unit_test_only:
        run_metric_unit_test()
        return

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    verify_python_hash_seed(args.seed, global_rank)
    seed_everything(args.seed)
    deepspeed_source = verify_patched_deepspeed()

    micro_global_batch = args.per_device_train_batch_size * world_size
    if args.global_batch_size % micro_global_batch != 0:
        raise ValueError(
            "global batch size must be divisible by "
            "per_device_train_batch_size * world_size"
        )
    gradient_accumulation_steps = args.global_batch_size // micro_global_batch

    extra_hparams = metric_hparams(
        args.m_x_averaging, args.m_x_damping, args.m_x_scale_clip
    )
    run_name = build_run_name(
        seed=args.seed,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        extra_hparams=extra_hparams,
    )
    output_dir = build_output_dir(
        method=args.lora,
        run_name=run_name,
        model=args.model,
        output_root=args.output_root,
    )
    wandb_run_name = f"{args.model}_math_rank{args.rank}_{args.lora}_lr{args.lr}_seed{args.seed}"

    wandb_run = None
    wandb_enabled = os.getenv("WANDB_MODE", "online").lower() != "disabled"
    if global_rank == 0 and wandb_enabled:
        wandb_run = wandb.init(
            entity=os.getenv("WANDB_ENTITY") or None,
            project=os.getenv("WANDB_PROJECT", "Qwen3-1.7B-Math"),
            name=wandb_run_name,
            group="Transformers-Math",
            config={
                "model": args.model,
                "method": args.lora,
                "learning_rate": args.lr,
                "seed": args.seed,
                "data_seed": args.seed,
                "rank": args.rank,
                "lora_alpha": args.alpha,
                "use_rslora": True,
                "scaling_formula": "alpha/sqrt(rank)",
                "scaling_factor": args.alpha / math.sqrt(args.rank),
                "m_x_averaging": args.m_x_averaging,
                "m_x_damping": args.m_x_damping,
                "m_x_scale_clip": args.m_x_scale_clip,
                "metric_definition": "diagonal EMA of mean squared adapter inputs",
                "metric_update_timing": "once per optimizer step",
                "metric_requires_grad": False,
                "global_batch_size": args.global_batch_size,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "epochs": 1,
            },
        )

    model_path = resolve_model_path(args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        local_files_only=True,
    )
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        use_rslora=True,
    )
    model = peft.get_peft_model(model, lora_config)
    if not model.peft_config["default"].use_rslora:
        raise RuntimeError("PEFT did not enable the required rsLoRA scaling")
    model.print_trainable_parameters()

    # LoRA-Pro's matrix algebra and M_x statistics are kept in float32.
    for name, module in model.named_modules():
        if "lora_" in name:
            module.to(torch.float32)

    mx_bindings = attach_mx_metrics(
        model,
        m_x_averaging=args.m_x_averaging,
        damping=args.m_x_damping,
        scale_clip=args.m_x_scale_clip,
        rank=args.rank,
        alpha=args.alpha,
    )
    expected_layers = int(model.config.num_hidden_layers) * len(TARGET_MODULES)
    if len(mx_bindings) != expected_layers:
        raise RuntimeError(
            f"Expected {expected_layers} M_x bindings, found {len(mx_bindings)}"
        )
    if global_rank == 0:
        print(f"DeepSpeed source: {deepspeed_source}", flush=True)
        print(f"Attached diagonal M_x to {len(mx_bindings)} LoRA layers", flush=True)
        print(
            "rsLoRA scale:",
            args.alpha / math.sqrt(args.rank),
            "gradient accumulation:",
            gradient_accumulation_steps,
            flush=True,
        )

    with TitledLog("load datasets and dataloaders", log_fn=log.info):
        datasets = peta.tasks.load_meta_math()
        if len(datasets["train"]) != 100000:
            raise RuntimeError(
                "Expected 100000 MetaMathQA training examples, got "
                f"{len(datasets['train'])}"
            )
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

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir="./logs/transformers_logs",
        run_name=wandb_run_name,
        do_train=True,
        do_eval=False,
        num_train_epochs=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # The repository's patched DeepSpeed optimizer constructs the Adam
        # update. The underlying optimizer must remain plain SGD.
        optim="sgd",
        logging_steps=1,
        bf16=True,
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to=["wandb"] if wandb_enabled else [],
        label_names=["labels"],
        ddp_find_unused_parameters=False,
        eval_strategy="no",
        save_strategy="no",
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=4,
        # Metric LoRA-Pro's gradient adjustment is implemented inside the
        # patched DeepSpeed ZeRO-2 optimizer (see verify_patched_deepspeed
        # above), so DeepSpeed must run even on a single GPU -- without it,
        # training silently degrades to plain SGD with no LoRA-Pro/M_x
        # correction.
        deepspeed=args.deepspeed_config,
    )

    trainer = MxTrainer(
        model=model,
        train_dataset=datasets["train"],
        tokenizer=tokenizer,
        args=train_args,
        data_collator=default_data_collator,
        callbacks=[MxCommitCallback(mx_bindings)],
    )
    trainer.train()

    update_counts = {int(binding.state.mx_updates.item()) for binding in mx_bindings}
    if update_counts != {int(trainer.state.global_step)}:
        raise RuntimeError(
            "M_x must be committed exactly once per optimizer step: "
            f"metric counts={sorted(update_counts)}, "
            f"Trainer global_step={trainer.state.global_step}"
        )
    for binding in mx_bindings:
        if not torch.isfinite(binding.state.mx_ema).all():
            raise RuntimeError(f"Non-finite M_x values in {binding.name}")
        if not (binding.state.mx_ema >= 0).all():
            raise RuntimeError(f"Negative M_x values in {binding.name}")

    # Preserve the learned statistics for audit/research, then make the PEFT
    # checkpoint self-contained for ordinary evaluation.
    mx_state_cpu = {
        binding.name: {
            "mx_ema": binding.state.mx_ema.detach().cpu(),
            "mx_updates": int(binding.state.mx_updates.item()),
        }
        for binding in mx_bindings
    }
    fold_mx_into_lora_a(mx_bindings)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if global_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(mx_state_cpu, output_dir / "mx_metric_state.pt")
        save_run_metadata(
            output_dir,
            {
                "method": args.lora,
                "model": args.model,
                "base_model_path": model_path,
                "learning_rate": args.lr,
                "seed": args.seed,
                "data_seed": args.seed,
                "rank": args.rank,
                "alpha": args.alpha,
                "use_rslora": True,
                "scaling_formula": "alpha/sqrt(rank)",
                "scaling_factor": args.alpha / math.sqrt(args.rank),
                "m_x_averaging": args.m_x_averaging,
                "m_x_damping": args.m_x_damping,
                "m_x_scale_clip": args.m_x_scale_clip,
                "metric_definition": "diagonal EMA of mean squared adapter inputs",
                "metric_normalization": "M_x / mean(M_x)",
                "metric_transform": "clamp(rsqrt(normalized_M_x + damping), 1/clip, clip)",
                "metric_update_timing": "once per optimizer step after parameter update",
                "metric_synchronized_across_data_parallel_ranks": True,
                "metric_requires_grad": False,
                "metric_folded_into_lora_A": True,
                "global_batch_size": args.global_batch_size,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "epochs": 1,
                "optimizer_steps": int(trainer.state.global_step),
                "metric_updates_per_layer": sorted(update_counts),
                "deepspeed_source": str(deepspeed_source),
            },
        )
        print(f"Saved evaluation-ready adapter at: {output_dir}", flush=True)
        if wandb_run is not None:
            wandb_run.summary["optimizer_steps"] = int(trainer.state.global_step)
            wandb_run.summary["metric_updates_per_layer"] = int(
                trainer.state.global_step
            )
            wandb.finish()


if __name__ == "__main__":
    main()

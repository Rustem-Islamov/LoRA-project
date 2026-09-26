r"""Shared helpers for naming and organizing fine-tuning runs.

Every training method (LoRA, LoRA-Pro, full fine-tuning, Metric LoRA-Pro)
saves its checkpoint through :func:`build_output_dir` /
:func:`build_run_name`, so that:

- every run's hyperparameters (learning rate, rank, seed, and any
  method-specific extras) are encoded in its directory name, so two runs
  can never silently collide or overwrite one another;
- every run's directory has the same shape, ``<output_root>/<model
  slug>/<task>/<method>/<run name>``, so the single shared evaluator in
  ``evaluation/eval_gsm8k.py`` can find and describe any of them the same
  way, regardless of which method produced it.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union
import json

__all__ = [
    "MODEL_PATHS",
    "DEFAULT_MODEL",
    "DEFAULT_TASK",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_EVAL_ROOT",
    "resolve_model_path",
    "model_slug",
    "canonical_float",
    "format_hparam",
    "build_run_name",
    "build_output_dir",
    "save_run_metadata",
    "load_run_metadata",
    "default_eval_output_path",
]

# Short, filesystem-friendly names for the base models used by the training
# and evaluation scripts. A path that is not one of these keys is used as-is,
# so pointing at an arbitrary local checkout still works.
MODEL_PATHS: Dict[str, str] = {
    "qwen3-1.7b-base": "./models/Qwen3-1.7B-Base",
    "llama-2-7b": "./models/llama-2-7b",
}

DEFAULT_MODEL = "qwen3-1.7b-base"
DEFAULT_TASK = "math"
DEFAULT_OUTPUT_ROOT = "./checkpoints"
DEFAULT_EVAL_ROOT = "./logs/eval"


def resolve_model_path(model: str) -> str:
    """Resolve a short model key (e.g. ``qwen3-1.7b-base``) to its local path.

    A value that is not a known key (e.g. an explicit path) is returned
    unchanged.
    """
    return MODEL_PATHS.get(model, model)


def model_slug(model: str) -> str:
    """A filesystem-safe identifier for a model, used as a directory name."""
    if model in MODEL_PATHS:
        return model
    return Path(model).name.lower().replace("_", "-")


def canonical_float(value: float) -> str:
    """Stable, short spelling for a float, shared by training and eval."""
    return format(float(value), ".12g")


def format_hparam(value: Any) -> str:
    """Render one hyperparameter value for use inside a directory name."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return canonical_float(value)
    return str(value)


def build_run_name(
    seed: int,
    lr: float,
    rank: Optional[int] = None,
    alpha: Optional[int] = None,
    extra_hparams: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a directory-safe run name that encodes every hyperparameter.

    Two runs that differ in rank, alpha, learning rate, seed, or any extra
    hyperparameter (e.g. Metric LoRA-Pro's ``m_x_averaging``) always get
    distinct names.
    """
    parts = []
    if rank is not None:
        parts.append(f"r{rank}")
    if alpha is not None:
        parts.append(f"a{format_hparam(alpha)}")
    parts.append(f"lr{canonical_float(lr)}")
    for key, value in (extra_hparams or {}).items():
        parts.append(f"{key}{format_hparam(value)}")
    parts.append(f"seed{seed}")
    return "_".join(parts)


def build_output_dir(
    method: str,
    run_name: str,
    model: str = DEFAULT_MODEL,
    task: str = DEFAULT_TASK,
    output_root: Union[str, Path] = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """``<output_root>/<model slug>/<task>/<method>/<run_name>``."""
    return Path(output_root) / model_slug(model) / task / method / run_name


def save_run_metadata(output_dir: Union[str, Path], metadata: Dict[str, Any]) -> Path:
    """Write ``metadata.json`` (hyperparameters + provenance) into a run dir.

    Every training script writes this file. The evaluator reads it back to
    recover the base model a checkpoint (in particular, a LoRA/LoRA-Pro
    adapter) was trained from, so ``--base-model`` does not need to be
    repeated at evaluation time.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata_path


def load_run_metadata(checkpoint_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read back ``metadata.json`` from a run directory, if present."""
    metadata_path = Path(checkpoint_dir) / "metadata.json"
    if not metadata_path.is_file():
        return None
    with metadata_path.open() as handle:
        return json.load(handle)


def default_eval_output_path(
    checkpoint_dir: Union[str, Path],
    output_root: Union[str, Path] = DEFAULT_OUTPUT_ROOT,
    eval_root: Union[str, Path] = DEFAULT_EVAL_ROOT,
) -> Path:
    """Mirror a checkpoint's path under ``eval_root`` for its result JSON.

    A checkpoint saved at ``<output_root>/qwen3-1.7b-base/math/lora/<run
    name>`` gets its evaluation result written to ``<eval_root>/qwen3-1.7b
    -base/math/lora/<run name>.json``, so results and checkpoints are always
    easy to line up by eye.
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    try:
        relative = checkpoint_dir.relative_to(Path(output_root).resolve())
    except ValueError:
        relative = Path(checkpoint_dir.name)
    # Run names contain dots (e.g. "lr0.0001"), so appending the suffix as a
    # string avoids Path.with_suffix() truncating at the wrong ".".
    return Path(eval_root) / relative.parent / (relative.name + ".json")

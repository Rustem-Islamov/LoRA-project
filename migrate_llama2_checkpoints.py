#!/usr/bin/env python3
"""One-time migration: move old Llama-2-7B checkpoints into ./checkpoints.

Before this repo's training scripts were unified (train_lora_math.py,
train_lorapro_math.py, train_fft_math.py, train_metric_lorapro_math.py),
checkpoints were saved under

    ./logs/transformers/llama-2-7b/math/<lr>/<method>[/<metric_tag>]/<seed>

using whatever string the training script or Slurm variable happened to
spell the learning rate as (e.g. "2e-05", "0.00016", "1e-5"). This script
finds every such checkpoint, re-derives its hyperparameters, and moves it to
the current, hyperparameter-qualified layout

    ./checkpoints/llama-2-7b/math/<method>/<run_name>

(see peta/utils/experiment.py), writing a metadata.json into each moved
checkpoint so evaluation/eval_gsm8k.py can find its base model automatically.

Defaults to a DRY RUN: it only prints the planned moves. Pass --apply to
actually move anything. Nothing is ever silently overwritten -- a
destination that already exists is reported as a conflict and skipped.
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from peta.utils import build_output_dir, build_run_name, resolve_model_path

METRIC_METHOD = "metric-lorapro-mx-rs-scale"
METRIC_TAG_PATTERN = re.compile(r"^mxavg_(.+)_damp_(.+)_clip_(.+)$")
DEFAULT_RANK = 8
DEFAULT_ALPHA = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        default="./logs/transformers/llama-2-7b/math",
        help="Old checkpoint root to scan.",
    )
    parser.add_argument(
        "--output-root",
        default="./checkpoints",
        help="New checkpoint root (matches peta.utils.experiment.DEFAULT_OUTPUT_ROOT).",
    )
    parser.add_argument("--model", default="llama-2-7b", help="Model key for the new layout.")
    parser.add_argument("--task", default="math")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move directories. Without this, only prints the plan.",
    )
    return parser.parse_args()


def find_checkpoint_dirs(logs_root: Path):
    """Yield every leaf directory that holds a model or adapter checkpoint."""

    if not logs_root.is_dir():
        return
    for candidate in sorted(logs_root.glob("**/")):
        has_adapter = (candidate / "adapter_config.json").is_file()
        has_full_model = (candidate / "config.json").is_file() and not has_adapter
        if has_adapter or has_full_model:
            yield candidate, has_adapter


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open() as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def parse_metric_tag(tag: str) -> Optional[Tuple[float, float, float]]:
    match = METRIC_TAG_PATTERN.match(tag)
    if not match:
        return None
    try:
        return tuple(float(value) for value in match.groups())  # type: ignore[return-value]
    except ValueError:
        return None


def parse_checkpoint(
    checkpoint_dir: Path, logs_root: Path, has_adapter: bool
) -> Optional[Dict[str, Any]]:
    """Recover (method, seed, lr, rank, alpha, extra_hparams) from a checkpoint's path."""

    relative_parts = checkpoint_dir.relative_to(logs_root).parts
    existing_config = (
        load_json(checkpoint_dir / "training_config.json")
        or load_json(checkpoint_dir / "mx_training_config.json")
    )

    rank: Optional[int] = None
    alpha: Optional[int] = None
    if has_adapter:
        adapter_config = load_json(checkpoint_dir / "adapter_config.json")
        rank = int(adapter_config.get("r", existing_config.get("rank", DEFAULT_RANK)))
        alpha = int(
            adapter_config.get("lora_alpha", existing_config.get("alpha", DEFAULT_ALPHA))
        )

    extra_hparams: Optional[Dict[str, Any]] = None

    if len(relative_parts) == 3:
        lr_str, method, seed_str = relative_parts
    elif len(relative_parts) == 4 and relative_parts[1] == METRIC_METHOD:
        lr_str, method, metric_tag, seed_str = relative_parts
        m_x_averaging = existing_config.get("m_x_averaging")
        m_x_damping = existing_config.get("m_x_damping")
        m_x_scale_clip = existing_config.get("m_x_scale_clip")
        if m_x_averaging is None or m_x_damping is None or m_x_scale_clip is None:
            parsed = parse_metric_tag(metric_tag)
            if parsed is None:
                return None
            m_x_averaging, m_x_damping, m_x_scale_clip = parsed
        extra_hparams = {
            "mxavg": float(m_x_averaging),
            "damp": float(m_x_damping),
            "clip": float(m_x_scale_clip),
        }
    else:
        return None

    try:
        lr = float(lr_str)
        seed = int(seed_str)
    except ValueError:
        return None

    return {
        "method": method,
        "seed": seed,
        "lr": lr,
        "rank": rank,
        "alpha": alpha,
        "extra_hparams": extra_hparams,
        "existing_config": existing_config,
    }


def build_metadata(parsed: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    metadata = dict(parsed["existing_config"])
    metadata["method"] = parsed["method"]
    metadata["model"] = model_key
    metadata["base_model_path"] = resolve_model_path(model_key)
    metadata["learning_rate"] = parsed["lr"]
    metadata["seed"] = parsed["seed"]
    if parsed["rank"] is not None:
        metadata["rank"] = parsed["rank"]
    if parsed["alpha"] is not None:
        metadata["alpha"] = parsed["alpha"]
    if parsed["extra_hparams"] is not None:
        metadata.setdefault("m_x_averaging", parsed["extra_hparams"]["mxavg"])
        metadata.setdefault("m_x_damping", parsed["extra_hparams"]["damp"])
        metadata.setdefault("m_x_scale_clip", parsed["extra_hparams"]["clip"])
    metadata["migrated_from"] = "logs/transformers/llama-2-7b/math"
    return metadata


def main() -> None:
    args = parse_args()
    logs_root = Path(args.logs_root)
    output_root = Path(args.output_root)

    planned = []  # (old_dir, new_dir, metadata)
    unrecognized = []
    destinations_seen: Dict[Path, Path] = {}
    collisions = []

    for checkpoint_dir, has_adapter in find_checkpoint_dirs(logs_root):
        parsed = parse_checkpoint(checkpoint_dir, logs_root, has_adapter)
        if parsed is None:
            unrecognized.append(checkpoint_dir)
            continue

        run_name = build_run_name(
            seed=parsed["seed"],
            lr=parsed["lr"],
            rank=parsed["rank"],
            alpha=parsed["alpha"],
            extra_hparams=parsed["extra_hparams"],
        )
        new_dir = build_output_dir(
            method=parsed["method"],
            run_name=run_name,
            model=args.model,
            task=args.task,
            output_root=output_root,
        )

        if new_dir in destinations_seen:
            collisions.append((checkpoint_dir, destinations_seen[new_dir], new_dir))
            continue
        destinations_seen[new_dir] = checkpoint_dir

        metadata = build_metadata(parsed, args.model)
        planned.append((checkpoint_dir, new_dir, metadata))

    print(f"Found {len(planned)} checkpoint(s) to migrate.")
    for old_dir, new_dir, _ in planned:
        exists = " [DESTINATION ALREADY EXISTS -- WILL SKIP]" if new_dir.exists() else ""
        print(f"  {old_dir}\n    -> {new_dir}{exists}")

    if unrecognized:
        print(f"\n{len(unrecognized)} directory(ies) had an unrecognized layout (skipped):")
        for path in unrecognized:
            print(f"  {path}")

    if collisions:
        print(f"\n{len(collisions)} collision(s) -- two old checkpoints map to the same new "
              "path (skipped, resolve manually):")
        for old_dir, other_old_dir, new_dir in collisions:
            print(f"  {old_dir}\n  {other_old_dir}\n    both -> {new_dir}")

    if not args.apply:
        print("\nDry run only -- nothing was moved. Re-run with --apply to perform these moves.")
        return

    moved = 0
    skipped = 0
    manifest = []
    for old_dir, new_dir, metadata in planned:
        if new_dir.exists():
            print(f"SKIP (destination exists): {new_dir}")
            skipped += 1
            continue
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_dir), str(new_dir))
        with (new_dir / "metadata.json").open("w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        manifest.append({"old": str(old_dir), "new": str(new_dir)})
        moved += 1
        print(f"Moved: {old_dir} -> {new_dir}")

    manifest_path = output_root / "_migration_manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nMoved {moved}, skipped {skipped}. Manifest written to {manifest_path}.")


if __name__ == "__main__":
    main()

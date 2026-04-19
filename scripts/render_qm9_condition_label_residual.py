from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from electrai.dataloader.dataset import RhoRead
from scripts.render_qm9_xz_slices import (
    resolve_torch_device,
    select_slice,
    to_batched_volume,
)


DEFAULT_ROOT = Path("/scratch/gpfs/ROSENGROUP/common/qm9/qm9_filelist.txt")
DEFAULT_SPLIT_FILE = REPO_ROOT / "examples/QM9/experiment_5/split_4000.json"

__all__ = ["visualize_qm9_condition_label_residual_sample"]


def _normalize_split_name(split: str) -> str:
    normalized = split.strip().lower()
    aliases = {
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "train": "train",
        "test": "test",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown split {split!r}. Expected one of train, validation/val, or test."
        )
    return aliases[normalized]


def _plot_condition_label_residual(
    volumes: dict[str, np.ndarray],
    *,
    split: str,
    sample_id: str,
    sample_idx: int,
    plane: str,
    slice_index: int | None,
    slice_frac: float,
    output_path: Path | None,
    show: bool,
):
    import matplotlib.pyplot as plt

    keys = ("condition", "label", "residual")
    titles = {
        "condition": "Condition",
        "label": "Label",
        "residual": "Residual\nLabel - Condition",
    }

    fig, axes = plt.subplots(1, len(keys), figsize=(14, 4.5), constrained_layout=True)
    slice_meta: dict[str, int] = {}
    slice_sums: dict[str, float] = {}
    for ax, key in zip(axes, keys, strict=True):
        slice_2d, axis_name, used_index, xlabel, ylabel = select_slice(
            volumes[key], plane, slice_index, slice_frac,
        )
        slice_meta[key] = used_index
        slice_sums[key] = float(np.asarray(slice_2d, dtype=np.float64).sum())
        finite = slice_2d[np.isfinite(slice_2d)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.percentile(finite, 1.0))
            vmax = float(np.percentile(finite, 99.0))
        if key == "residual" and finite.size:
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound
        if vmax <= vmin:
            center = float(finite.mean()) if finite.size else 0.0
            spread = float(np.max(np.abs(finite - center))) if finite.size else 1.0
            vmin, vmax = center - (spread or 1.0), center + (spread or 1.0)
        image = ax.imshow(
            slice_2d,
            origin="lower",
            cmap="coolwarm" if key == "residual" else "viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(
            f"{titles[key]}\n"
            f"{axis_name}={used_index}, shape={volumes[key].shape}\n"
            f"slice_sum={slice_sums[key]:.6g}",
            fontsize=10,
        )
        ax.set(xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Sample {sample_id} | {split}[{sample_idx}] | {plane.upper()} slice",
        fontsize=12,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, slice_meta, slice_sums


def visualize_qm9_condition_label_residual_sample(
    *,
    root: str | Path = DEFAULT_ROOT,
    split_file: str | Path = DEFAULT_SPLIT_FILE,
    split: str = "validation",
    sample_idx: int | None = None,
    seed: int | None = None,
    plane: str = "xz",
    slice_index: int | None = None,
    slice_frac: float = 0.5,
    output_path: str | Path | None = None,
    device: str | torch.device | None = None,
    show: bool = True,
    precision: str = "f32",
    augmentation: bool = False,
    random_seed: int = 42,
    downsample_data: int = 1,
    downsample_label: int = 1,
):
    split_name = _normalize_split_name(split)
    root_path = Path(root)
    split_path = Path(split_file)
    save_path = Path(output_path) if output_path is not None else None
    torch_device = resolve_torch_device(device)

    datamodule = RhoRead(
        root=str(root_path),
        precision=precision,
        batch_size=1,
        train_workers=0,
        val_workers=0,
        pin_memory=False,
        drop_last=False,
        split_file=str(split_path),
        augmentation=augmentation,
        random_seed=random_seed,
        downsample_data=downsample_data,
        downsample_label=downsample_label,
    )
    datamodule.setup(stage="fit")

    if not hasattr(datamodule, "subsets") or split_name not in datamodule.subsets:
        raise ValueError(f"Split {split_name!r} is not available in {split_path}.")
    subset = datamodule.subsets[split_name]
    if len(subset) == 0:
        raise ValueError(f"Split {split_name!r} is empty in {split_path}.")

    rng = np.random.default_rng(seed)
    chosen_idx = int(rng.integers(len(subset))) if sample_idx is None else sample_idx
    if not 0 <= chosen_idx < len(subset):
        raise IndexError(
            f"sample_idx={chosen_idx} is out of range for {split_name} size {len(subset)}"
        )

    item = subset[chosen_idx]
    if not isinstance(item, dict) or "data" not in item or "label" not in item:
        raise TypeError("Dataset samples must be dicts with data and label fields.")

    sample_id = item.get("index", chosen_idx)
    if isinstance(sample_id, torch.Tensor) and sample_id.ndim == 0:
        sample_id = sample_id.item()

    condition = to_batched_volume(item["data"], torch_device)
    label = to_batched_volume(item["label"], torch_device)
    if condition.shape != label.shape:
        raise ValueError(
            "Condition-label residual visualization requires matching shapes, got "
            f"{tuple(condition.shape)} and {tuple(label.shape)}."
        )
    residual = label - condition

    volumes: dict[str, np.ndarray] = {}
    for key, tensor in {
        "condition": condition,
        "label": label,
        "residual": residual,
    }.items():
        array = tensor.detach().cpu().float().numpy()
        while array.ndim > 3:
            array = array[0]
        if array.ndim != 3:
            raise ValueError(f"Expected a 3D volume for {key}, got shape {array.shape}")
        volumes[key] = array

    figure, slice_meta, slice_sums = _plot_condition_label_residual(
        volumes,
        split=split_name,
        sample_id=str(sample_id),
        sample_idx=chosen_idx,
        plane=plane,
        slice_index=slice_index,
        slice_frac=slice_frac,
        output_path=save_path,
        show=show,
    )

    return {
        "figure": figure,
        "sample_id": str(sample_id),
        "sample_idx": chosen_idx,
        "split": split_name,
        "device": str(torch_device),
        "plane": plane,
        "slice_indices": slice_meta,
        "slice_sums": slice_sums,
        "condition": condition.detach().cpu(),
        "label": label.detach().cpu(),
        "residual": residual.detach().cpu(),
        "output_path": save_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a random QM9 sample from a split and plot "
            "Condition, Label, and Residual (Label - Condition)."
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plane", type=str, default="xz", choices=("xy", "xz", "yz"))
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--slice-frac", type=float, default=0.5)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="f32")
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--downsample-data", type=int, default=1)
    parser.add_argument("--downsample-label", type=int, default=1)
    parser.add_argument("--hide", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = visualize_qm9_condition_label_residual_sample(
        root=args.root,
        split_file=args.split_file,
        split=args.split,
        sample_idx=args.sample_idx,
        seed=args.seed,
        plane=args.plane,
        slice_index=args.slice_index,
        slice_frac=args.slice_frac,
        output_path=args.output_path,
        device=args.device,
        show=not args.hide,
        precision=args.precision,
        augmentation=args.augmentation,
        random_seed=args.random_seed,
        downsample_data=args.downsample_data,
        downsample_label=args.downsample_label,
    )
    print(
        {
            "sample_id": result["sample_id"],
            "sample_idx": result["sample_idx"],
            "split": result["split"],
            "plane": result["plane"],
            "slice_indices": result["slice_indices"],
            "slice_sums": result["slice_sums"],
        }
    )


if __name__ == "__main__":
    main()

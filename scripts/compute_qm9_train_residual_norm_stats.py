from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FILELIST = Path("/scratch/gpfs/ROSENGROUP/common/qm9/qm9_filelist.txt")
FACTOR = 1.88973**3


def read_ids(path: Path) -> list[int]:
    with path.open() as handle:
        return [int(line.strip()) for line in handle if line.strip()]


def split_members(
    members: list[int],
    *,
    split_file: Path | None,
    split_key: str,
    val_frac: float,
    random_seed: int,
    max_samples: int | None,
) -> list[int]:
    if max_samples is not None and len(members) > max_samples:
        rng = np.random.default_rng(random_seed)
        positions = rng.choice(len(members), max_samples, replace=False)
        members = [members[int(position)] for position in positions]

    if split_file is not None:
        with split_file.open() as handle:
            splits = json.load(handle)
        return [members[int(position)] for position in splits[split_key]]

    try:
        import torch
    except ModuleNotFoundError:
        rng = np.random.default_rng(random_seed)
        positions = rng.permutation(len(members)).tolist()
        print(
            "warning: torch is unavailable, using numpy permutation for the "
            "generated split. Use uv run or pass --split-file to exactly match "
            "electrai.dataloader.split.",
            flush=True,
        )
    else:
        generator = torch.Generator().manual_seed(random_seed)
        positions = torch.randperm(len(members), generator=generator).tolist()
    validation_size = int(len(members) * val_frac)
    return [members[int(position)] for position in positions[validation_size:]]


def load_qm9_pair(
    root: Path,
    index: int,
    *,
    downsample_data: int,
    downsample_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    molecule = f"dsgdb9nsd_{index:06d}"
    data_dir = root / "data" / molecule
    label_dir = root / "label" / molecule
    data_shape = np.loadtxt(data_dir / "grid_sizes_22.dat", dtype=int)
    label_shape = np.loadtxt(label_dir / "grid_sizes_22.dat", dtype=int)
    sad = np.load(data_dir / "rho_22.npy").reshape(tuple(data_shape))
    dft = np.load(label_dir / "rho_22.npy").reshape(tuple(label_shape))

    nx, ny, nz = sad.shape[-3:]
    sad = sad[
        : nx // downsample_data * downsample_data : downsample_data,
        : ny // downsample_data * downsample_data : downsample_data,
        : nz // downsample_data * downsample_data : downsample_data,
    ]
    nx, ny, nz = dft.shape[-3:]
    dft = dft[
        : nx // downsample_label * downsample_label : downsample_label,
        : ny // downsample_label * downsample_label : downsample_label,
        : nz // downsample_label * downsample_label : downsample_label,
    ]
    if sad.shape != dft.shape:
        raise ValueError(
            f"Shape mismatch for {index}: SAD {sad.shape}, DFT {dft.shape}. "
            "Use matching downsample settings for residual training."
        )
    return sad * FACTOR, dft * FACTOR


def compute_stats(
    root: Path,
    indices: list[int],
    *,
    downsample_data: int,
    downsample_label: int,
    progress_every: int,
) -> dict[str, Any]:
    count = 0
    total = 0.0
    residual_min = math.inf
    residual_max = -math.inf
    for done, index in enumerate(indices, start=1):
        sad, dft = load_qm9_pair(
            root,
            index,
            downsample_data=downsample_data,
            downsample_label=downsample_label,
        )
        residual = np.subtract(dft, sad, dtype=np.float64)
        count += int(residual.size)
        total += float(residual.sum(dtype=np.float64))
        residual_min = min(residual_min, float(residual.min()))
        residual_max = max(residual_max, float(residual.max()))
        if progress_every and (done % progress_every == 0 or done == len(indices)):
            print(f"processed {done}/{len(indices)}", flush=True)

    residual_mean = total / count
    residual_scale = max(abs(residual_max - residual_mean), abs(residual_mean - residual_min))
    return {
        "residual_definition": "DFT - SAD",
        "n_samples": len(indices),
        "n_voxels": count,
        "residual_mean": residual_mean,
        "residual_min": residual_min,
        "residual_max": residual_max,
        "residual_scale": residual_scale,
        "unit_conversion_factor": FACTOR,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute train-split QM9 DFT-minus-SAD residual mean/min/max."
    )
    parser.add_argument("--filelist", type=Path, default=DEFAULT_FILELIST)
    parser.add_argument(
        "--train-indices",
        type=Path,
        default=None,
        help="Optional text file of QM9 molecule ids. If set, split arguments are ignored.",
    )
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split-key", default="train")
    parser.add_argument("--val-frac", type=float, default=0.005)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--downsample-data", type=int, default=1)
    parser.add_argument("--downsample-label", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    root = args.filelist.parent
    indices = (
        read_ids(args.train_indices)
        if args.train_indices is not None
        else split_members(
            read_ids(args.filelist),
            split_file=args.split_file,
            split_key=args.split_key,
            val_frac=args.val_frac,
            random_seed=args.random_seed,
            max_samples=args.max_samples,
        )
    )
    stats = compute_stats(
        root,
        indices,
        downsample_data=args.downsample_data,
        downsample_label=args.downsample_label,
        progress_every=args.progress_every,
    )
    payload = {
        "filelist": str(args.filelist),
        "split_file": str(args.split_file) if args.split_file else None,
        "split_key": args.split_key,
        "downsample_data": args.downsample_data,
        "downsample_label": args.downsample_label,
        **stats,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    print("\nConfig snippet:")
    print("residual_normalize: true")
    print(f"residual_mean: {stats['residual_mean']:.17g}")
    print(f"residual_min: {stats['residual_min']:.17g}")
    print(f"residual_max: {stats['residual_max']:.17g}")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()

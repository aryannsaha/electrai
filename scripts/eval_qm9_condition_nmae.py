from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from electrai.dataloader.dataset import RhoRead
from electrai.model.loss.charge import NormMAE


DEFAULT_ROOT = Path("/scratch/gpfs/ROSENGROUP/common/qm9/qm9_filelist.txt")
DEFAULT_SPLIT_FILE = REPO_ROOT / "examples/QM9/experiment_5/split_4000.json"


def normalize_splits(split_arg: str) -> list[str]:
    normalized = split_arg.strip().lower()
    if normalized == "all":
        return ["train", "validation", "test"]
    aliases = {
        "train": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown split {split_arg!r}. Expected train, validation/val, test, or all."
        )
    return [aliases[normalized]]


def evaluate_condition_nmae(
    *,
    root: Path,
    split_file: Path,
    splits: list[str],
    precision: str,
    batch_size: int,
    workers: int,
    random_seed: int,
    downsample_data: int,
    downsample_label: int,
) -> dict[str, dict[str, float | int]]:
    datamodule = RhoRead(
        root=str(root),
        precision=precision,
        batch_size=batch_size,
        train_workers=workers,
        val_workers=workers,
        pin_memory=False,
        drop_last=False,
        split_file=str(split_file),
        augmentation=False,
        random_seed=random_seed,
        downsample_data=downsample_data,
        downsample_label=downsample_label,
    )
    datamodule.setup(stage="fit")

    loss_fn = NormMAE()
    results: dict[str, dict[str, float | int]] = {}

    for split in splits:
        subset = datamodule.subsets.get(split)
        if subset is None:
            continue
        count = len(subset)
        if count == 0:
            results[split] = {"n_samples": 0, "mean_nmae": float("nan")}
            continue

        total_nmae = 0.0
        total_samples = 0
        print(f"Evaluating split={split} with n_samples={count}")
        with torch.no_grad():
            for idx in range(count):
                item = subset[idx]
                condition = item["data"].unsqueeze(0)
                label = item["label"].unsqueeze(0)
                total_nmae += loss_fn(condition, label).item()
                total_samples += 1
                if (idx + 1) % 500 == 0 or idx + 1 == count:
                    print(f"  processed {idx + 1}/{count}")

        results[split] = {
            "n_samples": total_samples,
            "mean_nmae": total_nmae / total_samples,
        }

    nonempty = [results[split] for split in splits if split in results and int(results[split]["n_samples"]) > 0]
    total_samples = sum(int(item["n_samples"]) for item in nonempty)
    if total_samples > 0:
        weighted_mean = sum(float(item["mean_nmae"]) * int(item["n_samples"]) for item in nonempty) / total_samples
        results["overall"] = {"n_samples": total_samples, "mean_nmae": weighted_mean}
    else:
        results["overall"] = {"n_samples": 0, "mean_nmae": float("nan")}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the condition-vs-label NMAE on the QM9 dataset.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Path to qm9_filelist.txt")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
        help="JSON split file used to define train/validation/test subsets.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="One of train, validation/val, test, or all.",
    )
    parser.add_argument("--precision", type=str, default="f32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--downsample-data", type=int, default=1)
    parser.add_argument("--downsample-label", type=int, default=1)
    args = parser.parse_args()

    splits = normalize_splits(args.split)
    results = evaluate_condition_nmae(
        root=args.root,
        split_file=args.split_file,
        splits=splits,
        precision=args.precision,
        batch_size=args.batch_size,
        workers=args.workers,
        random_seed=args.random_seed,
        downsample_data=args.downsample_data,
        downsample_label=args.downsample_label,
    )

    print(f"root={args.root}")
    print(f"split_file={args.split_file}")
    print(f"splits={splits}")
    print("-" * 60)
    for split in splits + ["overall"]:
        if split not in results:
            continue
        n_samples = int(results[split]["n_samples"])
        mean_nmae = float(results[split]["mean_nmae"])
        if n_samples == 0:
            print(f"{split:>10}: n_samples=0 | mean_nmae=nan")
        else:
            print(f"{split:>10}: n_samples={n_samples} | mean_nmae={mean_nmae:.6f}")


if __name__ == "__main__":
    main()

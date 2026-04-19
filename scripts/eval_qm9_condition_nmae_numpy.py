from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("/scratch/gpfs/ROSENGROUP/common/qm9")
DEFAULT_SPLIT_FILE = Path("/scratch/gpfs/ROSENGROUP/aryan/electrai/examples/QM9/experiment_5/split_4000.json")
FACTOR = 1.88973**3


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


def load_qm9_pair(root: Path, idx: int) -> tuple[np.ndarray, np.ndarray]:
    base = f"dsgdb9nsd_{idx:06d}"
    data_size = np.loadtxt(root / "data" / base / "grid_sizes_22.dat", dtype=int)
    label_size = np.loadtxt(root / "label" / base / "grid_sizes_22.dat", dtype=int)
    data = np.load(root / "data" / base / "rho_22.npy").reshape(data_size) * FACTOR
    label = np.load(root / "label" / base / "rho_22.npy").reshape(label_size) * FACTOR
    return data, label


def norm_mae(output: np.ndarray, target: np.ndarray) -> float:
    abs_error = np.abs(output - target).sum(dtype=np.float64)
    nelec = max(float(target.sum(dtype=np.float64)), 1e-12)
    return abs_error / nelec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute condition-vs-label NMAE directly from QM9 numpy files.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    splits = json.loads(args.split_file.read_text())
    requested_splits = normalize_splits(args.split)

    results: dict[str, dict[str, float | int]] = {}
    for split in requested_splits:
        indices = splits.get(split, [])
        if not indices:
            results[split] = {"n_samples": 0, "mean_nmae": float("nan")}
            print(f"{split}: n_samples=0 | mean_nmae=nan", flush=True)
            continue

        total_nmae = 0.0
        if args.threads <= 1:
            iterator = (norm_mae(*load_qm9_pair(args.root, int(idx))) for idx in indices)
        else:
            with ThreadPoolExecutor(max_workers=args.threads) as executor:
                iterator = executor.map(
                    lambda idx: norm_mae(*load_qm9_pair(args.root, int(idx))),
                    indices,
                )
                for position, value in enumerate(iterator, start=1):
                    total_nmae += value
                    if position % 500 == 0 or position == len(indices):
                        print(f"{split}: processed {position}/{len(indices)}", flush=True)
                iterator = None
        if iterator is not None:
            for position, value in enumerate(iterator, start=1):
                total_nmae += value
                if position % 500 == 0 or position == len(indices):
                    print(f"{split}: processed {position}/{len(indices)}", flush=True)

        mean_nmae = total_nmae / len(indices)
        results[split] = {"n_samples": len(indices), "mean_nmae": mean_nmae}
        print(f"{split}: n_samples={len(indices)} | mean_nmae={mean_nmae:.6f}", flush=True)

    nonempty = [item for item in results.values() if int(item["n_samples"]) > 0]
    if nonempty:
        total_samples = sum(int(item["n_samples"]) for item in nonempty)
        overall = sum(float(item["mean_nmae"]) * int(item["n_samples"]) for item in nonempty) / total_samples
        print(f"overall: n_samples={total_samples} | mean_nmae={overall:.6f}", flush=True)
    else:
        print("overall: n_samples=0 | mean_nmae=nan", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path("/scratch/gpfs/ROSENGROUP/common/qm9")
DEFAULT_FILELIST = DEFAULT_ROOT / "qm9_filelist.txt"
DEFAULT_OUTPUT_DIR = Path("scripts/qm9_residual_stats")
FACTOR = 1.88973**3
EPS = 1e-12


SAMPLE_FIELDNAMES = [
    "index",
    "n_voxels",
    "nx",
    "ny",
    "nz",
    "sad_sum",
    "dft_sum",
    "residual_sum",
    "charge_error_abs",
    "residual_mean",
    "residual_variance",
    "residual_std",
    "residual_min",
    "residual_max",
    "residual_abs_mean",
    "residual_mae",
    "residual_mse",
    "residual_rmse",
    "residual_l1",
    "residual_l2",
    "residual_linf",
    "nmae",
    "relative_l2",
    "positive_fraction",
    "negative_fraction",
    "zero_fraction",
    "residual_sumsq",
    "residual_abs_sum",
    "positive_count",
    "negative_count",
    "zero_count",
]


@dataclass
class VoxelTotals:
    count: int = 0
    sum: float = 0.0
    sumsq: float = 0.0
    abs_sum: float = 0.0
    min: float = math.inf
    max: float = -math.inf
    positive_count: int = 0
    negative_count: int = 0
    zero_count: int = 0

    def update(self, row: dict[str, Any]) -> None:
        count = int(row["n_voxels"])
        self.count += count
        self.sum += float(row["residual_sum"])
        self.sumsq += float(row["residual_sumsq"])
        self.abs_sum += float(row["residual_abs_sum"])
        self.min = min(self.min, float(row["residual_min"]))
        self.max = max(self.max, float(row["residual_max"]))
        self.positive_count += int(row["positive_count"])
        self.negative_count += int(row["negative_count"])
        self.zero_count += int(row["zero_count"])

    def to_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": float("nan"),
                "variance": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "mae": float("nan"),
                "mse": float("nan"),
                "rmse": float("nan"),
            }
        mean = self.sum / self.count
        variance = max((self.sumsq / self.count) - mean * mean, 0.0)
        return {
            "count": self.count,
            "sum": self.sum,
            "mean": mean,
            "variance": variance,
            "std": math.sqrt(variance),
            "min": self.min,
            "max": self.max,
            "mae": self.abs_sum / self.count,
            "mse": self.sumsq / self.count,
            "rmse": math.sqrt(self.sumsq / self.count),
            "l1": self.abs_sum,
            "l2": math.sqrt(self.sumsq),
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "zero_count": self.zero_count,
            "positive_fraction": self.positive_count / self.count,
            "negative_fraction": self.negative_count / self.count,
            "zero_fraction": self.zero_count / self.count,
        }


def read_indices(filelist: Path, limit: int | None) -> list[int]:
    with filelist.open() as handle:
        indices = [int(line.strip()) for line in handle if line.strip()]
    if limit is not None:
        return indices[:limit]
    return indices


def qm9_paths(root: Path, index: int) -> tuple[Path, Path, Path, Path]:
    molecule = f"dsgdb9nsd_{index:06d}"
    data_dir = root / "data" / molecule
    label_dir = root / "label" / molecule
    return (
        data_dir / "rho_22.npy",
        label_dir / "rho_22.npy",
        data_dir / "grid_sizes_22.dat",
        label_dir / "grid_sizes_22.dat",
    )


def load_pair(root: Path, index: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    data_path, label_path, data_grid_path, label_grid_path = qm9_paths(root, index)
    data_shape = tuple(int(value) for value in np.loadtxt(data_grid_path, dtype=int))
    label_shape = tuple(int(value) for value in np.loadtxt(label_grid_path, dtype=int))
    if data_shape != label_shape:
        raise ValueError(
            f"Grid mismatch for {index}: SAD shape={data_shape}, DFT shape={label_shape}"
        )
    sad = np.load(data_path, mmap_mode="r").reshape(data_shape)
    dft = np.load(label_path, mmap_mode="r").reshape(label_shape)
    return sad, dft, data_shape


def compute_sample_row(
    root: Path, index: int, include_sample_median: bool
) -> dict[str, float | int]:
    sad, dft, shape = load_pair(root, index)
    residual = np.subtract(dft, sad, dtype=np.float64)
    residual *= FACTOR

    flat = residual.ravel()
    n_voxels = int(flat.size)
    residual_sum = float(flat.sum(dtype=np.float64))
    residual_sumsq = float(np.dot(flat, flat))
    residual_abs_sum = float(np.abs(flat).sum(dtype=np.float64))
    residual_min = float(flat.min())
    residual_max = float(flat.max())
    residual_mean = residual_sum / n_voxels
    residual_variance = max((residual_sumsq / n_voxels) - residual_mean**2, 0.0)
    residual_l2 = math.sqrt(residual_sumsq)
    positive_count = int(np.count_nonzero(flat > 0.0))
    negative_count = int(np.count_nonzero(flat < 0.0))
    zero_count = n_voxels - positive_count - negative_count

    dft_flat = dft.ravel()
    dft_sum = float(dft_flat.sum(dtype=np.float64) * FACTOR)
    dft_sumsq = float(np.dot(dft_flat, dft_flat) * FACTOR * FACTOR)
    sad_sum = float(sad.sum(dtype=np.float64) * FACTOR)

    row: dict[str, float | int] = {
        "index": index,
        "n_voxels": n_voxels,
        "nx": shape[0],
        "ny": shape[1],
        "nz": shape[2],
        "sad_sum": sad_sum,
        "dft_sum": dft_sum,
        "residual_sum": residual_sum,
        "charge_error_abs": abs(residual_sum),
        "residual_mean": residual_mean,
        "residual_variance": residual_variance,
        "residual_std": math.sqrt(residual_variance),
        "residual_min": residual_min,
        "residual_max": residual_max,
        "residual_abs_mean": residual_abs_sum / n_voxels,
        "residual_mae": residual_abs_sum / n_voxels,
        "residual_mse": residual_sumsq / n_voxels,
        "residual_rmse": math.sqrt(residual_sumsq / n_voxels),
        "residual_l1": residual_abs_sum,
        "residual_l2": residual_l2,
        "residual_linf": max(abs(residual_min), abs(residual_max)),
        "nmae": residual_abs_sum / max(abs(dft_sum), EPS),
        "relative_l2": residual_l2 / max(math.sqrt(dft_sumsq), EPS),
        "positive_fraction": positive_count / n_voxels,
        "negative_fraction": negative_count / n_voxels,
        "zero_fraction": zero_count / n_voxels,
        "residual_sumsq": residual_sumsq,
        "residual_abs_sum": residual_abs_sum,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "zero_count": zero_count,
    }
    if include_sample_median:
        row["residual_median"] = float(np.median(flat))
    return row


def histogram_for_sample(
    root: Path, index: int, bins: int, hist_min: float, hist_max: float
) -> np.ndarray:
    sad, dft, _shape = load_pair(root, index)
    residual = np.subtract(dft, sad, dtype=np.float64)
    residual *= FACTOR
    counts, _edges = np.histogram(residual, bins=bins, range=(hist_min, hist_max))
    return counts.astype(np.int64, copy=False)


def print_progress(prefix: str, done: int, total: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    rate = done / max(elapsed, EPS)
    remaining = (total - done) / max(rate, EPS)
    print(
        f"{prefix}: processed {done}/{total} "
        f"({done / total:.1%}); elapsed={elapsed / 60:.1f}m; eta={remaining / 60:.1f}m",
        flush=True,
    )


def summarize_sample_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    ignored = {"index", "nx", "ny", "nz", "positive_count", "negative_count", "zero_count"}
    numeric_fields = [
        field for field in rows[0] if field not in ignored and field != "n_voxels"
    ]
    summary: dict[str, dict[str, float]] = {}
    for field in numeric_fields:
        values = np.array([float(row[field]) for row in rows], dtype=np.float64)
        summary[field] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "variance": float(values.var()),
            "std": float(values.std()),
        }
    return summary


def histogram_summary(
    counts: np.ndarray, hist_min: float, hist_max: float
) -> dict[str, Any]:
    total = int(counts.sum())
    if total == 0:
        return {}
    edges = np.linspace(hist_min, hist_max, len(counts) + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cdf = np.cumsum(counts)

    def quantile(probability: float) -> float:
        target = probability * (total - 1)
        bin_index = int(np.searchsorted(cdf, target, side="left"))
        bin_index = min(max(bin_index, 0), len(counts) - 1)
        previous = int(cdf[bin_index - 1]) if bin_index > 0 else 0
        in_bin = int(counts[bin_index])
        if in_bin <= 0:
            return float(centers[bin_index])
        fraction = (target - previous) / in_bin
        return float(edges[bin_index] + fraction * (edges[bin_index + 1] - edges[bin_index]))

    mode_index = int(np.argmax(counts))
    return {
        "method": "fixed-width histogram over residual values",
        "bins": int(len(counts)),
        "range": [hist_min, hist_max],
        "median_approx": quantile(0.5),
        "mode_approx": float(centers[mode_index]),
        "mode_bin": [float(edges[mode_index]), float(edges[mode_index + 1])],
        "mode_count": int(counts[mode_index]),
        "q01_approx": quantile(0.01),
        "q05_approx": quantile(0.05),
        "q25_approx": quantile(0.25),
        "q75_approx": quantile(0.75),
        "q95_approx": quantile(0.95),
        "q99_approx": quantile(0.99),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute DFT-minus-SAD residual statistics for the QM9 numpy density files."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--filelist", type=Path, default=DEFAULT_FILELIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--histogram-bins", type=int, default=4096)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--include-sample-median",
        action="store_true",
        help="Also compute exact per-sample residual medians. This is slower.",
    )
    parser.add_argument(
        "--skip-histogram",
        action="store_true",
        help="Skip the second pass used for approximate global median/mode/quantiles.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = args.output_dir / "qm9_residual_sample_metrics.csv"
    summary_json = args.output_dir / "qm9_residual_summary.json"
    histogram_npz = args.output_dir / "qm9_residual_histogram.npz"

    indices = read_indices(args.filelist, args.limit)
    fieldnames = SAMPLE_FIELDNAMES.copy()
    if args.include_sample_median:
        fieldnames.append("residual_median")

    print(f"root={args.root}", flush=True)
    print(f"filelist={args.filelist}", flush=True)
    print(f"n_samples={len(indices)}", flush=True)
    print(f"sample_csv={sample_csv}", flush=True)
    print(f"summary_json={summary_json}", flush=True)

    voxel_totals = VoxelTotals()
    rows: list[dict[str, Any]] = []
    start_time = time.time()
    with sample_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=max(args.threads, 1)) as executor:
            iterator = executor.map(
                lambda idx: compute_sample_row(
                    args.root, idx, args.include_sample_median
                ),
                indices,
            )
            for done, row in enumerate(iterator, start=1):
                writer.writerow(row)
                rows.append(row)
                voxel_totals.update(row)
                if done % args.progress_every == 0 or done == len(indices):
                    handle.flush()
                    print_progress("sample pass", done, len(indices), start_time)

    payload: dict[str, Any] = {
        "root": str(args.root),
        "filelist": str(args.filelist),
        "residual_definition": "DFT - SAD",
        "unit_conversion": {
            "factor": FACTOR,
            "description": "Bohr^-3 densities converted to Angstrom^-3, matching electrai.dataloader.utils.load_npy.",
        },
        "n_samples": len(rows),
        "voxel_residual_stats_exact": voxel_totals.to_dict(),
        "sample_metric_summaries": summarize_sample_metrics(rows) if rows else {},
    }

    if not args.skip_histogram and rows:
        hist_min = float(payload["voxel_residual_stats_exact"]["min"])
        hist_max = float(payload["voxel_residual_stats_exact"]["max"])
        if math.isclose(hist_min, hist_max):
            counts = np.array([voxel_totals.count], dtype=np.int64)
            payload["voxel_residual_distribution_approx"] = {
                "method": "single occupied value",
                "bins": 1,
                "range": [hist_min, hist_max],
                "median_approx": hist_min,
                "mode_approx": hist_min,
            }
        else:
            print(
                f"histogram pass: bins={args.histogram_bins}, range=({hist_min}, {hist_max})",
                flush=True,
            )
            counts = np.zeros(args.histogram_bins, dtype=np.int64)
            hist_start_time = time.time()
            with ThreadPoolExecutor(max_workers=max(args.threads, 1)) as executor:
                iterator = executor.map(
                    lambda idx: histogram_for_sample(
                        args.root, idx, args.histogram_bins, hist_min, hist_max
                    ),
                    indices,
                )
                for done, sample_counts in enumerate(iterator, start=1):
                    counts += sample_counts
                    if done % args.progress_every == 0 or done == len(indices):
                        print_progress(
                            "histogram pass", done, len(indices), hist_start_time
                        )
            payload["voxel_residual_distribution_approx"] = histogram_summary(
                counts, hist_min, hist_max
            )
            edges = np.linspace(hist_min, hist_max, len(counts) + 1)
            np.savez_compressed(histogram_npz, counts=counts, edges=edges)
            payload["histogram_npz"] = str(histogram_npz)

    write_json(summary_json, payload)
    print(f"wrote {sample_csv}", flush=True)
    print(f"wrote {summary_json}", flush=True)
    if not args.skip_histogram and rows:
        print(f"wrote {histogram_npz}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
METRIC_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_.\-/]*?)="
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
)
EPOCH_RE = re.compile(r"Epoch\s+(\d+)")
PROGRESS_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _is_step_metric(name: str) -> bool:
    return name.endswith("_step") or name in {"train_loss_step", "lr-Adam"}


def _is_epoch_metric(name: str) -> bool:
    return (
        name.startswith("val_")
        or name.endswith("_epoch")
        or name in {"val_loss", "val_nmae", "val_rollout_nmae", "val_rollout_sq_fro"}
    )


def _clean_metric_name(name: str) -> str:
    return name.strip().replace("/", "_")


def _append_unique(
    series: dict[str, list[tuple[float, float]]],
    key: str,
    x_value: float,
    y_value: float,
) -> None:
    points = series[key]
    if points and points[-1] == (x_value, y_value):
        return
    points.append((x_value, y_value))


def _read_lightning_csv(
    path: Path,
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, list[tuple[float, float]]]]:
    step_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    epoch_series: dict[str, list[tuple[float, float]]] = defaultdict(list)

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            step = _to_float(row.get("step"))
            epoch = _to_float(row.get("epoch"))
            for raw_name, raw_value in row.items():
                if raw_name in {"step", "epoch"}:
                    continue
                value = _to_float(raw_value)
                if value is None:
                    continue
                name = _clean_metric_name(raw_name)
                if _is_step_metric(name):
                    _append_unique(step_series, name, step if step is not None else row_index, value)
                elif _is_epoch_metric(name):
                    _append_unique(epoch_series, name, epoch if epoch is not None else row_index, value)

    return dict(step_series), dict(epoch_series)


def _parse_slurm_logs(
    paths: Iterable[Path],
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, list[tuple[float, float]]]]:
    step_points: dict[tuple[str, int, int], float] = {}
    epoch_points: dict[tuple[str, int], float] = {}
    fallback_step = 0

    for path in sorted(paths):
        with path.open(errors="ignore") as handle:
            for line in handle:
                for part in line.split("\r"):
                    if "Epoch " not in part or "=" not in part:
                        continue
                    epoch_matches = EPOCH_RE.findall(part)
                    if not epoch_matches:
                        continue
                    epoch = int(epoch_matches[-1])
                    progress_match = PROGRESS_RE.search(part)
                    if progress_match:
                        local_step = int(progress_match.group(1))
                        total_steps = int(progress_match.group(2))
                        step_x = epoch * max(total_steps, 1) + local_step
                    else:
                        fallback_step += 1
                        step_x = fallback_step

                    for raw_name, raw_value in METRIC_RE.findall(part):
                        name = _clean_metric_name(raw_name)
                        value = _to_float(raw_value)
                        if value is None:
                            continue
                        if _is_step_metric(name):
                            step_points[(name, epoch, step_x)] = value
                        elif _is_epoch_metric(name):
                            epoch_points[(name, epoch)] = value

    step_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (name, _epoch, step_x), value in sorted(step_points.items(), key=lambda item: (item[0][0], item[0][2])):
        step_series[name].append((float(step_x), value))

    epoch_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (name, epoch), value in sorted(epoch_points.items(), key=lambda item: (item[0][0], item[0][1])):
        epoch_series[name].append((float(epoch), value))

    return dict(step_series), dict(epoch_series)


def _parse_checkpoint_metrics(
    experiment_path: Path,
) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for checkpoint in sorted(experiment_path.rglob("*.ckpt")):
        if checkpoint.name == "last.ckpt":
            continue
        epoch_match = re.search(r"epoch[=_-](\d+)", checkpoint.stem)
        if not epoch_match:
            continue
        epoch = float(epoch_match.group(1))
        for raw_name, raw_value in METRIC_RE.findall(checkpoint.stem):
            name = _clean_metric_name(raw_name)
            if name == "epoch" or name.endswith("_epoch"):
                continue
            value = _to_float(raw_value)
            if value is not None:
                series[f"ckpt_{name}"].append((epoch, value))

    return {name: sorted(points) for name, points in series.items()}


def _find_metrics_csvs(experiment_path: Path) -> list[Path]:
    return sorted(experiment_path.rglob("metrics.csv"))


def _merge_series(
    target: dict[str, list[tuple[float, float]]],
    source: dict[str, list[tuple[float, float]]],
) -> None:
    for name, points in source.items():
        target.setdefault(name, []).extend(points)
        target[name] = sorted(target[name])


def _smoothed(points: list[tuple[float, float]], window: int) -> list[tuple[float, float]]:
    if window <= 1 or len(points) < window:
        return points
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    smoothed: list[tuple[float, float]] = []
    running_sum = 0.0
    for index, value in enumerate(ys):
        running_sum += value
        if index >= window:
            running_sum -= ys[index - window]
        count = min(index + 1, window)
        smoothed.append((xs[index], running_sum / count))
    return smoothed


def _pick_series(
    series: dict[str, list[tuple[float, float]]],
    max_series: int,
) -> list[tuple[str, list[tuple[float, float]]]]:
    preferred = [
        "train_loss_step",
        "train_nmae_step",
        "train_loss_epoch",
        "train_nmae_epoch",
        "val_loss",
        "val_nmae",
        "val_rollout_nmae",
        "val_rollout_sq_fro",
    ]
    names = sorted(series)
    ordered = [name for name in preferred if name in series]
    ordered.extend(name for name in names if name not in ordered)
    return [(name, series[name]) for name in ordered[:max_series]]


def _plot_series(
    ax,
    series: dict[str, list[tuple[float, float]]],
    *,
    title: str,
    xlabel: str,
    max_series: int,
    smooth_window: int = 1,
) -> None:
    chosen = _pick_series(series, max_series)
    if not chosen:
        ax.text(0.5, 0.5, "No data found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title(title)
        return

    for name, points in chosen:
        points = _smoothed(points, smooth_window)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, linewidth=1.6, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def plot_experiment_summary(
    experiment_path: str | Path,
    *,
    output_path: str | Path | None = None,
    metrics_csv: str | Path | None = None,
    max_series: int = 4,
    smooth_window: int = 25,
    parse_slurm: bool = True,
    show: bool = False,
):
    import matplotlib.pyplot as plt

    experiment_path = Path(experiment_path).expanduser().resolve()
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path does not exist: {experiment_path}")
    if not experiment_path.is_dir():
        raise NotADirectoryError(f"Expected an experiment directory: {experiment_path}")

    step_series: dict[str, list[tuple[float, float]]] = {}
    epoch_series: dict[str, list[tuple[float, float]]] = {}

    csv_paths = [Path(metrics_csv).expanduser().resolve()] if metrics_csv else _find_metrics_csvs(experiment_path)
    for csv_path in csv_paths:
        if csv_path.exists():
            csv_step, csv_epoch = _read_lightning_csv(csv_path)
            _merge_series(step_series, csv_step)
            _merge_series(epoch_series, csv_epoch)

    if parse_slurm and not csv_paths:
        slurm_paths = sorted(experiment_path.glob("slurm-*.out"))
        slurm_step, slurm_epoch = _parse_slurm_logs(slurm_paths)
        _merge_series(step_series, slurm_step)
        _merge_series(epoch_series, slurm_epoch)

    checkpoint_series = _parse_checkpoint_metrics(experiment_path)
    if not step_series and not epoch_series and not checkpoint_series:
        raise ValueError(
            "No plottable metrics found. Expected metrics.csv, slurm-*.out, "
            f"or checkpoint filenames with metrics under {experiment_path}."
        )

    save_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else experiment_path / "experiment_summary.png"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    _plot_series(
        axes[0],
        step_series,
        title="Training Step Metrics",
        xlabel="global step",
        max_series=max_series,
        smooth_window=smooth_window,
    )
    _plot_series(
        axes[1],
        epoch_series,
        title="Epoch Metrics",
        xlabel="epoch",
        max_series=max_series,
    )
    _plot_series(
        axes[2],
        checkpoint_series,
        title="Checkpoint Metrics",
        xlabel="epoch",
        max_series=max_series,
    )
    fig.suptitle(experiment_path.name, fontsize=13)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, save_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create one PNG with three side-by-side plots for an experiment: "
            "training step metrics, epoch metrics, and checkpoint metrics."
        ),
    )
    parser.add_argument("experiment_path", type=Path)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--max-series", type=int, default=4)
    parser.add_argument("--smooth-window", type=int, default=25)
    parser.add_argument("--no-slurm", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    _, save_path = plot_experiment_summary(
        args.experiment_path,
        output_path=args.output_path,
        metrics_csv=args.metrics_csv,
        max_series=args.max_series,
        smooth_window=args.smooth_window,
        parse_slurm=not args.no_slurm,
        show=args.show,
    )
    print(f"Wrote {save_path}")


if __name__ == "__main__":
    main()

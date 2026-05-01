from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

from plot_experiment_summary import (
    EPOCH_RE,
    PROGRESS_RE,
    _find_metrics_csvs,
    _merge_series,
    _parse_checkpoint_metrics,
    _parse_slurm_logs,
    _read_lightning_csv,
    _smoothed,
    _to_float,
)


DEFAULT_PANELS = ("val_nmae", "epoch", "train_loss_step")
ALIASES = {
    "val_name": "val_nmae",
    "val_nmae": "val_nmae",
    "train_loss": "train_loss_step",
}


def _clean_panel_name(name: str) -> str:
    return ALIASES.get(name.strip(), name.strip())


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


def _read_epoch_from_metrics_csv(path: Path) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            step = _to_float(row.get("step"))
            epoch = _to_float(row.get("epoch"))
            if epoch is None:
                continue
            _append_unique(series, "epoch", step if step is not None else row_index, epoch)
    return dict(series)


def _parse_epoch_from_slurm_logs(paths: list[Path]) -> dict[str, list[tuple[float, float]]]:
    points: dict[tuple[int, int], float] = {}
    fallback_step = 0

    for path in sorted(paths):
        with path.open(errors="ignore") as handle:
            for line in handle:
                for part in line.split("\r"):
                    if "Epoch " not in part:
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
                    points[(step_x, epoch)] = float(epoch)

    return {"epoch": [(float(step), value) for (step, _epoch), value in sorted(points.items())]}


def _load_series(
    experiment_path: Path,
    *,
    metrics_csv: Path | None,
    parse_slurm: bool,
) -> dict[str, list[tuple[float, float]]]:
    all_series: dict[str, list[tuple[float, float]]] = {}
    step_series: dict[str, list[tuple[float, float]]] = {}
    epoch_series: dict[str, list[tuple[float, float]]] = {}

    csv_paths = [metrics_csv.expanduser().resolve()] if metrics_csv else _find_metrics_csvs(experiment_path)
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        csv_step, csv_epoch = _read_lightning_csv(csv_path)
        _merge_series(step_series, csv_step)
        _merge_series(epoch_series, csv_epoch)
        _merge_series(all_series, _read_epoch_from_metrics_csv(csv_path))

    if parse_slurm and not csv_paths:
        slurm_paths = sorted(experiment_path.glob("slurm-*.out"))
        slurm_step, slurm_epoch = _parse_slurm_logs(slurm_paths)
        _merge_series(step_series, slurm_step)
        _merge_series(epoch_series, slurm_epoch)
        _merge_series(all_series, _parse_epoch_from_slurm_logs(slurm_paths))

    checkpoint_series = _parse_checkpoint_metrics(experiment_path)

    _merge_series(all_series, step_series)
    _merge_series(all_series, epoch_series)
    _merge_series(all_series, checkpoint_series)
    return all_series


def _candidate_names(name: str) -> list[str]:
    if name == "val_nmae":
        return ["val_nmae", "ckpt_val_nmae", "val_loss", "ckpt_val_loss"]
    if name == "train_loss_step":
        return ["train_loss_step", "train_nmae_step"]
    return [name]


def _pick_metric(
    series: dict[str, list[tuple[float, float]]],
    requested_name: str,
) -> tuple[str | None, list[tuple[float, float]]]:
    for candidate in _candidate_names(requested_name):
        points = series.get(candidate)
        if points:
            return candidate, points
    return None, []


def _xlabel_for_metric(name: str | None) -> str:
    if name is None:
        return ""
    if name.startswith("val_") or name.startswith("ckpt_"):
        return "epoch"
    return "global step"


def _plot_metric(ax, name: str, points: list[tuple[float, float]], *, smooth_window: int) -> None:
    if not points:
        ax.text(0.5, 0.5, "No data found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title(name)
        return

    smooth = smooth_window if name.endswith("_step") else 1
    plotted_points = _smoothed(points, smooth)
    xs = [point[0] for point in plotted_points]
    ys = [point[1] for point in plotted_points]
    ax.plot(xs, ys, linewidth=1.6)
    ax.set_title(name)
    ax.set_xlabel(_xlabel_for_metric(name))
    ax.grid(alpha=0.25)

    finite_ys = [value for value in ys if math.isfinite(value)]
    if finite_ys:
        best = min(finite_ys)
        ax.text(
            0.02,
            0.96,
            f"min: {best:.6g}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )


def plot_experiment_triptych(
    experiment_path: str | Path,
    *,
    output_path: str | Path | None = None,
    panels: tuple[str, str, str] = DEFAULT_PANELS,
    metrics_csv: str | Path | None = None,
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

    save_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else experiment_path / "experiment_triptych.png"
    )
    metric_csv_path = Path(metrics_csv).expanduser().resolve() if metrics_csv else None

    series = _load_series(
        experiment_path,
        metrics_csv=metric_csv_path,
        parse_slurm=parse_slurm,
    )
    if not series:
        raise ValueError(
            "No plottable metrics found. Expected metrics.csv, slurm-*.out, "
            f"or checkpoint filenames with metrics under {experiment_path}."
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, raw_name in zip(axes, panels, strict=True):
        requested_name = _clean_panel_name(raw_name)
        actual_name, points = _pick_metric(series, requested_name)
        _plot_metric(
            ax,
            actual_name or requested_name,
            points,
            smooth_window=smooth_window,
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
            "Create one PNG with three side-by-side plots for an experiment. "
            "Defaults: val_nmae, epoch, train_loss_step."
        ),
    )
    parser.add_argument("experiment_path", type=Path)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument(
        "--panels",
        nargs=3,
        default=DEFAULT_PANELS,
        metavar=("LEFT", "MIDDLE", "RIGHT"),
    )
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

    _, save_path = plot_experiment_triptych(
        args.experiment_path,
        output_path=args.output_path,
        panels=tuple(args.panels),
        metrics_csv=args.metrics_csv,
        smooth_window=args.smooth_window,
        parse_slurm=not args.no_slurm,
        show=args.show,
    )
    print(f"Wrote {save_path}")


if __name__ == "__main__":
    main()

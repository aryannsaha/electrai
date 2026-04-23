from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from hydra.utils import instantiate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from electrai.lightning_w_time_flow import LightningGenerator


def log(message: str) -> None:
    print(f"[eval_qm9_timestep_errors] {message}", flush=True)


def read_progress_every() -> int:
    value = os.environ.get("QM9_TIMESTEP_PROGRESS_EVERY", "25")
    try:
        return max(1, int(value))
    except ValueError:
        log(f"Invalid QM9_TIMESTEP_PROGRESS_EVERY={value!r}; using 25.")
        return 25


def read_n_steps(default: int) -> int:
    value = os.environ.get("QM9_TIMESTEP_N_STEPS")
    if value is None:
        return max(1, int(default))
    try:
        return max(1, int(value))
    except ValueError:
        log(f"Invalid QM9_TIMESTEP_N_STEPS={value!r}; using {default}.")
        return max(1, int(default))


def loader_length(loader) -> int | None:
    try:
        return len(loader)
    except TypeError:
        return None


def format_progress(current: int, total: int | None) -> str:
    if total is None:
        return str(current)
    return f"{current}/{total}"


def write_qm9_timestep_errors_csv() -> None:
    start_time = time.perf_counter()
    progress_every = read_progress_every()
    config_path = REPO_ROOT / "examples/QM9/experiment_16/config.yaml"
    checkpoint_path = REPO_ROOT / "examples/QM9/experiment_16/checkpoints/last.ckpt"
    split_file = REPO_ROOT / "examples/QM9/experiment_5/split_4000.json"
    output_csv = REPO_ROOT / "scripts/qm9_timestep_errors.csv"

    log("Starting QM9 timestep error evaluation.")
    log(f"Config: {config_path}")
    log(f"Checkpoint: {checkpoint_path}")
    log(f"Split file: {split_file}")
    log(f"Output CSV: {output_csv}")
    log(f"Progress print interval: every {progress_every} batch(es).")

    log("Loading config.")
    with config_path.open() as handle:
        cfg = SimpleNamespace(**yaml.safe_load(handle))
    cfg.data["split_file"] = str(split_file)
    cfg.data["batch_size"] = 1
    cfg.data["augmentation"] = False
    cfg.data["val_workers"] = 0

    log("Instantiating data module.")
    datamodule = instantiate(cfg.data)
    log("Setting up data module with stage='fit'.")
    datamodule.setup(stage="fit")
    loader = datamodule.val_dataloader()
    total_batches = loader_length(loader)
    if total_batches is None:
        log("Validation loader ready; total batch count is unavailable.")
    else:
        log(f"Validation loader ready with {total_batches} batch(es).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_detail = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    log(f"Using device: {device} ({device_detail}).")
    log("Loading model checkpoint.")
    model = LightningGenerator.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        cfg=cfg,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    log("Model loaded and set to eval mode.")

    timestep_totals: dict[int, dict[str, float]] = {}
    default_n_steps = int(getattr(model, "n_inference_steps", getattr(cfg, "n_inference_steps", 10)))
    n_steps = read_n_steps(default_n_steps)
    log(f"Running {n_steps} inference steps per batch.")

    samples_processed = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            source = batch["data"].to(device)
            target = batch["label"].to(device)
            x_t = source.clone()
            batch_size = source.shape[0]
            t_values = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=source.dtype)

            for timestep, t_value in enumerate(t_values):
                linear_target = (1.0 - t_value) * source + t_value * target

                linear_diff = x_t - linear_target
                linear_frobenius_error = linear_diff.reshape(linear_diff.shape[0], -1).pow(2).sum(dim=1).sqrt()
                linear_abs_error = linear_diff.abs().reshape(linear_diff.shape[0], -1).sum(dim=1)
                linear_electron_count = linear_target.reshape(linear_target.shape[0], -1).sum(dim=1).clamp_min(1e-12)
                linear_nmae = linear_abs_error / linear_electron_count

                final_diff = x_t - target
                final_frobenius_error = final_diff.reshape(final_diff.shape[0], -1).pow(2).sum(dim=1).sqrt()
                final_abs_error = final_diff.abs().reshape(final_diff.shape[0], -1).sum(dim=1)
                final_electron_count = target.reshape(target.shape[0], -1).sum(dim=1).clamp_min(1e-12)
                final_nmae = final_abs_error / final_electron_count

                bucket = timestep_totals.setdefault(
                    timestep,
                    {
                        "t": float(t_value.item()),
                        "rollout_vs_linear_frobenius_sum": 0.0,
                        "rollout_vs_linear_nmae_sum": 0.0,
                        "rollout_vs_final_frobenius_sum": 0.0,
                        "rollout_vs_final_nmae_sum": 0.0,
                        "count": 0.0,
                    },
                )
                bucket["rollout_vs_linear_frobenius_sum"] += float(linear_frobenius_error.sum().item())
                bucket["rollout_vs_linear_nmae_sum"] += float(linear_nmae.sum().item())
                bucket["rollout_vs_final_frobenius_sum"] += float(final_frobenius_error.sum().item())
                bucket["rollout_vs_final_nmae_sum"] += float(final_nmae.sum().item())
                bucket["count"] += float(x_t.shape[0])

                if timestep == n_steps:
                    break

                t_cur = t_values[timestep]
                dt = t_values[timestep + 1] - t_cur
                t_batch = t_cur.expand(x_t.shape[0])
                y_hat = model(x_t, t_batch)
                denom = (1.0 - t_cur).clamp(min=model.eps)
                x_t = x_t + dt * (y_hat - x_t) / denom

            samples_processed += batch_size
            should_log_progress = (
                batch_idx == 1
                or batch_idx % progress_every == 0
                or batch_idx == total_batches
            )
            if should_log_progress:
                elapsed = time.perf_counter() - start_time
                final_bucket = timestep_totals[n_steps]
                final_count = final_bucket["count"]
                final_frobenius = final_bucket["rollout_vs_linear_frobenius_sum"] / final_count
                final_nmae = final_bucket["rollout_vs_linear_nmae_sum"] / final_count
                log(
                    "Finished batch "
                    f"{format_progress(batch_idx, total_batches)}; "
                    f"samples={samples_processed}; elapsed={elapsed:.1f}s; "
                    f"final-step avg_rollout_vs_linear_frobenius={final_frobenius:.6g}; "
                    f"final-step avg_rollout_vs_linear_nmae={final_nmae:.6g}."
                )

    log("Writing timestep averages to CSV.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestep",
                "t",
                "average_rollout_vs_linear_frobenius_error",
                "average_rollout_vs_linear_nmae",
                "average_rollout_vs_final_frobenius_error",
                "average_rollout_vs_final_nmae",
            ],
        )
        writer.writeheader()
        for timestep in sorted(timestep_totals):
            bucket = timestep_totals[timestep]
            count = bucket["count"]
            writer.writerow(
                {
                    "timestep": timestep,
                    "t": bucket["t"],
                    "average_rollout_vs_linear_frobenius_error": bucket[
                        "rollout_vs_linear_frobenius_sum"
                    ]
                    / count,
                    "average_rollout_vs_linear_nmae": bucket["rollout_vs_linear_nmae_sum"] / count,
                    "average_rollout_vs_final_frobenius_error": bucket[
                        "rollout_vs_final_frobenius_sum"
                    ]
                    / count,
                    "average_rollout_vs_final_nmae": bucket["rollout_vs_final_nmae_sum"] / count,
                },
            )

    elapsed = time.perf_counter() - start_time
    log(f"Wrote {output_csv}")
    log(f"Completed in {elapsed:.1f}s.")


if __name__ == "__main__":
    write_qm9_timestep_errors_csv()

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)
__all__ = ["visualize_qm9_checkpoint_sample"]


def load_cfg_from_checkpoint(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    cfg = hyper_parameters.get("cfg") if isinstance(hyper_parameters, dict) else None
    if cfg is not None:
        return OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    config_path = checkpoint_path.parent.parent / "config.yaml"
    if config_path.exists():
        return OmegaConf.load(config_path)
    raise ValueError("Could not find a config inside the checkpoint or a sibling config.yaml.")


def to_batched_volume(tensor, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(tensor)
    if tensor.ndim == 3:
        tensor = tensor[None, None]
    elif tensor.ndim == 4:
        tensor = tensor[None]
    elif tensor.ndim != 5:
        raise ValueError(f"Expected a 3D, 4D, or 5D tensor, got shape {tuple(tensor.shape)}")
    return tensor.to(device)


def select_slice(volume: np.ndarray, plane: str, slice_index: int | None, slice_frac: float):
    if not 0.0 <= slice_frac <= 1.0:
        raise ValueError(f"slice_frac must be in [0, 1], got {slice_frac}")
    axis, fixed_axis, xlabel, ylabel = {"xy": (2, "z", "x", "y"), "xz": (1, "y", "x", "z"), "yz": (0, "x", "y", "z")}[plane]
    size = volume.shape[axis]
    index = int(round(slice_frac * (size - 1))) if slice_index is None else slice_index
    if not 0 <= index < size:
        raise IndexError(f"Slice index {index} is out of range for plane {plane} with size {size}")
    slice_2d = {"xy": volume[:, :, index], "xz": volume[:, index, :], "yz": volume[index, :, :]}[plane].T
    return np.asarray(slice_2d, dtype=np.float64), fixed_axis, index, xlabel, ylabel


def plot_slices(
    volumes: dict[str, np.ndarray],
    *,
    sample_id: str,
    sample_idx: int,
    plane: str,
    slice_index: int | None,
    slice_frac: float,
    nmae: float,
    sampler_label: str,
    output_path: Path | None,
    show: bool,
):
    import matplotlib.pyplot as plt

    titles = {"noise": "Gaussian Noise", "condition": "Condition", "label": "Label", "output": f"Final Output\nNMAE={nmae:.4f}"}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    slice_meta: dict[str, int] = {}
    for ax, key in zip(axes, ("noise", "condition", "label", "output"), strict=True):
        slice_2d, axis_name, used_index, xlabel, ylabel = select_slice(volumes[key], plane, slice_index, slice_frac)
        slice_meta[key] = used_index
        finite = slice_2d[np.isfinite(slice_2d)]
        vmin, vmax = (0.0, 1.0) if finite.size == 0 else (float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.0)))
        if vmin < 0.0 < vmax:
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound
        if vmax <= vmin:
            center = float(finite.mean()) if finite.size else 0.0
            spread = float(np.max(np.abs(finite - center))) if finite.size else 1.0
            vmin, vmax = center - (spread or 1.0), center + (spread or 1.0)
        image = ax.imshow(slice_2d, origin="lower", cmap="coolwarm" if vmin < 0.0 < vmax else "viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"{titles[key]}\n{axis_name}={used_index}, shape={volumes[key].shape}", fontsize=10)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Sample {sample_id} | validation[{sample_idx}] | {plane.upper()} slice | {sampler_label}", fontsize=12)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, slice_meta


def visualize_qm9_checkpoint_sample(
    checkpoint_path: str | Path,
    split_file: str | Path | None = None,
    *,
    sample_idx: int | None = None,
    seed: int | None = None,
    plane: str = "xz",
    slice_index: int | None = None,
    slice_frac: float = 0.5,
    output_path: str | Path | None = None,
    device: str | None = None,
    show: bool = True,
):
    checkpoint_path = Path(checkpoint_path)
    split_path = Path(split_file) if split_file is not None else None
    save_path = Path(output_path) if output_path is not None else None
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    cfg = load_cfg_from_checkpoint(checkpoint_path)
    if not hasattr(cfg, "data"):
        raise ValueError("Checkpoint config must contain a Hydra datamodule at cfg.data.")
    for key, value in (("augmentation", False), ("split_file", str(split_path) if split_path else None)):
        if value is None:
            continue
        try:
            cfg.data[key] = value
        except Exception:
            try:
                setattr(cfg.data, key, value)
            except Exception:
                pass

    datamodule = instantiate(cfg.data)
    datamodule.setup(stage="fit")
    val_set = datamodule.val_set
    rng = np.random.default_rng(seed)
    chosen_idx = int(rng.integers(len(val_set))) if sample_idx is None else sample_idx
    if not 0 <= chosen_idx < len(val_set):
        raise IndexError(f"sample_idx={chosen_idx} is out of range for validation set size {len(val_set)}")

    item = val_set[chosen_idx]
    if not isinstance(item, dict) or "data" not in item or "label" not in item:
        raise TypeError("Validation samples must be dicts with data and label fields.")
    sample_id = item.get("index", chosen_idx)
    if isinstance(sample_id, torch.Tensor) and sample_id.ndim == 0:
        sample_id = sample_id.item()

    condition = to_batched_volume(item["data"], torch_device)
    label = to_batched_volume(item["label"], torch_device)
    noise_seed = int(rng.integers(2**31 - 1))

    training_mode = str(getattr(cfg, "training_mode", "")).lower()
    if training_mode == "flow_match":
        from electrai.lightning_flow import LightningFlowMatch

        model = LightningFlowMatch.load_from_checkpoint(checkpoint_path, cfg=cfg).to(torch_device).eval()
        module_type = "flow"
    else:
        from electrai.lightning import LightningGenerator

        model = LightningGenerator.load_from_checkpoint(checkpoint_path, cfg=cfg).to(torch_device).eval()
        module_type = "generator"

    generator = torch.Generator(device="cpu")
    generator.manual_seed(noise_seed)
    if module_type == "flow":
        solver = model.sample_solver.lower()
        if solver not in {"euler", "heun"}:
            raise ValueError(f"Unsupported flow solver: {solver}")
        noise = torch.randn((condition.shape[0], 1, *condition.shape[2:]), generator=generator, dtype=torch.float32).to(device=torch_device, dtype=condition.dtype)
        output, ema_model = noise.clone(), model.ema_model
        was_training = ema_model.training
        ema_model.eval()
        t_schedule = torch.linspace(model.eps, 1.0, model.n_sample_steps + 1, device=torch_device, dtype=condition.dtype)
        for t_cur, t_next in zip(t_schedule[:-1], t_schedule[1:], strict=True):
            dt = t_next - t_cur
            t_batch = torch.full((condition.shape[0],), t_cur, device=torch_device, dtype=condition.dtype)
            velocity = model._predict_velocity(ema_model, output, condition, t_batch)
            if solver == "euler":
                output = output + velocity * dt
            else:
                euler_step = output + velocity * dt
                t_next_batch = torch.full((condition.shape[0],), t_next, device=torch_device, dtype=condition.dtype)
                velocity_next = model._predict_velocity(ema_model, euler_step, condition, t_next_batch)
                output = output + 0.5 * dt * (velocity + velocity_next)
        if was_training:
            ema_model.train()
        sampler_label = f"{solver}, {model.n_sample_steps} steps"
    else:
        noise = torch.randn(tuple(label.shape), generator=generator, dtype=torch.float32).to(device=torch_device, dtype=label.dtype)
        output, sampler_label = model(condition), "deterministic forward pass"

    from electrai.model.loss.charge import NormMAE

    nmae = NormMAE()(output, label).item()
    volumes: dict[str, np.ndarray] = {}
    for key, tensor in {"noise": noise, "condition": condition, "label": label, "output": output}.items():
        array = tensor.detach().cpu().float().numpy()
        while array.ndim > 3:
            array = array[0]
        if array.ndim != 3:
            raise ValueError(f"Expected a 3D volume for {key}, got shape {array.shape}")
        volumes[key] = array

    figure, slice_meta = plot_slices(
        volumes,
        sample_id=str(sample_id),
        sample_idx=chosen_idx,
        plane=plane,
        slice_index=slice_index,
        slice_frac=slice_frac,
        nmae=nmae,
        sampler_label=sampler_label,
        output_path=save_path,
        show=show,
    )
    LOGGER.info("Visualized sample %s from validation[%s]", sample_id, chosen_idx)
    return {
        "figure": figure,
        "sample_id": str(sample_id),
        "sample_idx": chosen_idx,
        "module_type": module_type,
        "plane": plane,
        "slice_indices": slice_meta,
        "nmae": nmae,
        "noise": noise.detach().cpu(),
        "condition": condition.detach().cpu(),
        "label": label.detach().cpu(),
        "output": output.detach().cpu(),
        "output_path": save_path,
    }

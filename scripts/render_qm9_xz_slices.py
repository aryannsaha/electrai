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


def resolve_torch_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device

    requested = "auto" if device is None else str(device).strip().lower()
    if requested in {"auto", ""}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested a GPU device, but neither CUDA nor MPS is available.")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device}', but CUDA is not available.")
    if requested.startswith("mps") and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError(f"Requested device '{device}', but MPS is not available.")
    return torch.device(device)


def load_lightning_module_from_checkpoint(model_cls, checkpoint_path: Path, *, cfg, device: torch.device):
    model = model_cls.load_from_checkpoint(checkpoint_path, map_location="cpu", cfg=cfg)
    model.requires_grad_(False)
    return model.to(device).eval()


def load_checkpoint_model_for_training_mode(
    training_mode: str,
    checkpoint_path: Path,
    *,
    cfg,
    device: torch.device,
):
    if training_mode == "flow_match":
        from electrai.lightning_flow import LightningFlowMatch

        model_cls = LightningFlowMatch
        module_type = "flow"
    elif training_mode == "flow_match_reflow":
        from electrai.lightning_flow_reflow import LightningFlowMatchReflow

        model_cls = LightningFlowMatchReflow
        module_type = "flow"
    elif training_mode == "flow_match_residual":
        from electrai.lightning_flow_residual import LightningFlowMatchResidual

        model_cls = LightningFlowMatchResidual
        module_type = "flow_residual"
    elif training_mode == "flow_match_residual_displacement":
        from electrai.lightning_flow_residual_displacement import (
            LightningFlowMatchResidualDisplacement,
        )

        model_cls = LightningFlowMatchResidualDisplacement
        module_type = "flow_residual_displacement"
    elif training_mode == "flow_match_cond_aug":
        from electrai.lightning_flow_cond_aug import LightningFlowMatchCondAug

        model_cls = LightningFlowMatchCondAug
        module_type = "flow_cond_aug"
    elif training_mode == "flow_match_pretrained_cond":
        from electrai.lightning_flow_pretrained_cond import (
            LightningFlowMatchPretrainedCond,
        )

        model_cls = LightningFlowMatchPretrainedCond
        module_type = "flow_pretrained_cond"
    elif training_mode == "flow_match_with_time":
        from electrai.lightning_w_time_flow import (
            LightningGenerator as LightningGeneratorFlowWithTime,
        )

        model_cls = LightningGeneratorFlowWithTime
        module_type = "flow_with_time"
    elif training_mode == "regression_with_time":
        from electrai.lightning_w_time import (
            LightningGenerator as LightningGeneratorWithTime,
        )

        model_cls = LightningGeneratorWithTime
        module_type = "generator_with_time"
    else:
        from electrai.lightning import LightningGenerator

        model_cls = LightningGenerator
        module_type = "generator"

    model = load_lightning_module_from_checkpoint(
        model_cls,
        checkpoint_path,
        cfg=cfg,
        device=device,
    )
    return model, module_type


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
    if plane == "xy":
        slice_2d = volume[:, :, index].T
    elif plane == "xz":
        slice_2d = volume[:, index, :].T
    else:
        slice_2d = volume[index, :, :].T
    return np.asarray(slice_2d, dtype=np.float64), fixed_axis, index, xlabel, ylabel


def plot_slices(
    volumes: dict[str, np.ndarray],
    *,
    source_title: str,
    sample_id: str,
    sample_idx: int,
    plane: str,
    slice_index: int | None,
    slice_frac: float,
    nmae: float,
    condition_nmae: float | None,
    sampler_label: str,
    plot_condition_residual: bool,
    output_path: Path | None,
    show: bool,
):
    import matplotlib.pyplot as plt

    keys = ["source", "condition", "label", "output", "residual"]
    if plot_condition_residual:
        keys.append("condition_residual")
    titles = {
        "source": source_title,
        "condition": "Condition",
        "label": "Label" if condition_nmae is None else f"Label\nCondition NMAE={condition_nmae:.4f}",
        "output": f"Final Output\nNMAE={nmae:.4f}",
        "residual": "Residual\nOutput - Label",
        "condition_residual": "Residual\nLabel - Condition",
    }
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    slice_meta: dict[str, int] = {}
    for ax, key in zip(axes, keys, strict=True):
        slice_2d, axis_name, used_index, xlabel, ylabel = select_slice(volumes[key], plane, slice_index, slice_frac)
        slice_meta[key] = used_index
        finite = slice_2d[np.isfinite(slice_2d)]
        vmin, vmax = (0.0, 1.0) if finite.size == 0 else (float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.0)))
        if key in {"residual", "condition_residual"} and finite.size:
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound
        elif vmin < 0.0 < vmax:
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound
        if vmax <= vmin:
            center = float(finite.mean()) if finite.size else 0.0
            spread = float(np.max(np.abs(finite - center))) if finite.size else 1.0
            vmin, vmax = center - (spread or 1.0), center + (spread or 1.0)
        image = ax.imshow(
            slice_2d,
            origin="lower",
            cmap="coolwarm" if key in {"residual", "condition_residual"} or vmin < 0.0 < vmax else "viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
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
    n_steps: int | None = None,
    solver: str | None = None,
    plane: str = "xz",
    slice_index: int | None = None,
    slice_frac: float = 0.5,
    plot_condition_residual: bool = False,
    output_path: str | Path | None = None,
    device: str | None = None,
    show: bool = True,
):
    checkpoint_path = Path(checkpoint_path)
    split_path = Path(split_file) if split_file is not None else None
    save_path = Path(output_path) if output_path is not None else None
    torch_device = resolve_torch_device(device)
    solver_override = None if solver is None else str(solver).strip().lower()
    if n_steps is not None and n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    if solver_override is not None and solver_override not in {"euler", "heun"}:
        raise ValueError(f"Unsupported solver override: {solver}")

    cfg = load_cfg_from_checkpoint(checkpoint_path)
    if not hasattr(cfg, "data"):
        raise ValueError("Checkpoint config must contain a Hydra datamodule at cfg.data.")
    for key, value in (
        ("augmentation", False),
        ("split_file", str(split_path) if split_path else None),
        ("n_sample_steps", n_steps),
        ("sample_solver", solver_override),
    ):
        if value is None:
            continue
        target = cfg if key in {"n_sample_steps", "sample_solver"} else cfg.data
        try:
            target[key] = value
        except Exception:
            try:
                setattr(target, key, value)
            except Exception:
                if key in {"n_sample_steps", "sample_solver"}:
                    try:
                        cfg[key] = value
                    except Exception:
                        pass
                else:
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

    noise_seed = int(rng.integers(2**31 - 1))

    training_mode = str(getattr(cfg, "training_mode", "")).lower()
    model, module_type = load_checkpoint_model_for_training_mode(
        training_mode,
        checkpoint_path,
        cfg=cfg,
        device=torch_device,
    )

    condition = to_batched_volume(item["data"], torch_device)
    label = to_batched_volume(item["label"], torch_device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(noise_seed)
    with torch.inference_mode():
        if module_type in {
            "flow",
            "flow_residual",
            "flow_residual_displacement",
            "flow_cond_aug",
            "flow_pretrained_cond",
        }:
            if n_steps is not None:
                model.n_sample_steps = int(n_steps)
            if solver_override is not None:
                model.sample_solver = solver_override
            solver = model.sample_solver.lower()
            if solver not in {"euler", "heun"}:
                raise ValueError(f"Unsupported flow solver: {solver}")
            if module_type == "flow" and str(getattr(model, "source_distribution", "")).lower() in {
                "conditioned",
                "conditioned_gaussian",
            }:
                source_title = "Conditioned Source"
            elif module_type == "flow" and str(getattr(model, "source_distribution", "")).lower() in {
                "condition",
                "condition_only",
                "deterministic_condition",
                "sad",
                "sad_guess",
            }:
                source_title = "SAD Source"
            elif module_type == "flow_residual" and str(getattr(model, "source_distribution", "")).lower() in {
                "zero",
                "zeros",
                "deterministic_zero",
                "residual_zero",
            }:
                source_title = "Zero Residual Source"
            elif module_type == "flow_residual":
                source_title = "Residual Source"
            else:
                source_title = "Gaussian Source"

            fork_devices = list(range(torch.cuda.device_count())) if torch_device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(noise_seed)
                if torch_device.type == "cuda":
                    torch.cuda.manual_seed_all(noise_seed)

                source = model._sample_source_state(
                    condition,
                    reference_state=model._reference_state_from_condition(condition),
                )

            output = source.clone()
            rollout_model = model._default_sampling_model(stage="sample")
            was_training = rollout_model.training
            rollout_model.eval()
            t_schedule = torch.linspace(model.eps, 1.0, model.n_sample_steps + 1, device=torch_device, dtype=condition.dtype)
            model_cond = model._condition_for_model(condition, stage="sample")
            initial_state = source.clone()
            initial_model_input = torch.cat([initial_state, model_cond], dim=1)
            for t_cur, t_next in zip(t_schedule[:-1], t_schedule[1:], strict=True):
                dt = t_next - t_cur
                t_batch = torch.full((condition.shape[0],), t_cur, device=torch_device, dtype=condition.dtype)
                velocity = model._predict_velocity(rollout_model, output, model_cond, t_batch)
                if solver == "euler":
                    output = output + velocity * dt
                else:
                    euler_step = output + velocity * dt
                    t_next_batch = torch.full((condition.shape[0],), t_next, device=torch_device, dtype=condition.dtype)
                    velocity_next = model._predict_velocity(rollout_model, euler_step, model_cond, t_next_batch)
                    output = output + 0.5 * dt * (velocity + velocity_next)
            output = model._prediction_from_state(output, condition)
            if was_training:
                rollout_model.train()
            sampler_label = f"{solver}, {model.n_sample_steps} steps"
        elif module_type == "flow_with_time":
            if solver_override is not None and solver_override != "euler":
                raise ValueError(
                    "flow_match_with_time checkpoints currently support only "
                    f"the euler visualization rollout, got solver={solver_override!r}."
                )
            if n_steps is not None:
                model.n_inference_steps = int(n_steps)

            source = condition.clone()
            source_title = "Condition Source"
            model_cond = condition.clone()
            initial_state = source.clone()
            initial_model_input = initial_state.clone()
            output = model._sample(condition)
            sampler_label = f"euler, {model.n_inference_steps} steps"
        else:
            source = torch.randn(tuple(label.shape), generator=generator, dtype=torch.float32).to(device=torch_device, dtype=label.dtype)
            if module_type == "generator_with_time":
                t_batch = torch.ones(condition.shape[0], device=torch_device, dtype=condition.dtype)
                output = model(condition, t_batch)
                sampler_label = "deterministic t=1 forward pass"
            else:
                output = model(condition)
                sampler_label = "deterministic forward pass"
            source_title = "Gaussian Source"
            model_cond = condition.clone()
            initial_state = source.clone()
            initial_model_input = condition.clone()

        from electrai.model.loss.charge import NormMAE

        nmae = NormMAE()(output, label).item()
        condition_nmae = None
        condition_residual = None
        if condition.shape == label.shape:
            condition_nmae = NormMAE()(condition, label).item()
            condition_residual = label - condition
        elif plot_condition_residual:
            raise ValueError(
                "plot_condition_residual=True requires matching condition and label shapes, got "
                f"{tuple(condition.shape)} and {tuple(label.shape)}."
            )
        residual = output - label
        volumes: dict[str, np.ndarray] = {}
        tensor_map = {
            "source": source,
            "condition": condition,
            "label": label,
            "output": output,
            "residual": residual,
        }
        if condition_residual is not None:
            tensor_map["condition_residual"] = condition_residual
        for key, tensor in tensor_map.items():
            array = tensor.detach().cpu().float().numpy()
            while array.ndim > 3:
                array = array[0]
            if array.ndim != 3:
                raise ValueError(f"Expected a 3D volume for {key}, got shape {array.shape}")
            volumes[key] = array

    figure, slice_meta = plot_slices(
        volumes,
        source_title=source_title,
        sample_id=str(sample_id),
        sample_idx=chosen_idx,
        plane=plane,
        slice_index=slice_index,
        slice_frac=slice_frac,
        nmae=nmae,
        condition_nmae=condition_nmae,
        sampler_label=sampler_label,
        plot_condition_residual=plot_condition_residual,
        output_path=save_path,
        show=show,
    )
    LOGGER.info("Visualized sample %s from validation[%s]", sample_id, chosen_idx)
    return {
        "figure": figure,
        "sample_id": str(sample_id),
        "sample_idx": chosen_idx,
        "module_type": module_type,
        "device": str(torch_device),
        "n_steps": getattr(model, "n_sample_steps", getattr(model, "n_inference_steps", None)),
        "plane": plane,
        "solver": getattr(model, "sample_solver", "euler" if module_type == "flow_with_time" else None),
        "slice_indices": slice_meta,
        "nmae": nmae,
        "condition_nmae": condition_nmae,
        "source": source.detach().cpu(),
        "noise": source.detach().cpu(),
        "condition": condition.detach().cpu(),
        "model_condition": model_cond.detach().cpu(),
        "initial_state": initial_state.detach().cpu(),
        "initial_model_input": initial_model_input.detach().cpu(),
        "label": label.detach().cpu(),
        "output": output.detach().cpu(),
        "residual": residual.detach().cpu(),
        "condition_residual": None if condition_residual is None else condition_residual.detach().cpu(),
        "output_path": save_path,
    }

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.profilers import PyTorchProfiler, SimpleProfiler

from electrai.lightning import LightningGenerator as StandardLightningGenerator
from electrai.lightning_w_time_flow_res import (
    LightningGenerator as ResidualFlowLightningGenerator,
)


def load_config(path: Path) -> SimpleNamespace:
    with path.open() as fp:
        return SimpleNamespace(**yaml.safe_load(fp))


def choose_lightning_module(cfg: SimpleNamespace):
    training_mode = str(getattr(cfg, "training_mode", "")).lower()
    if training_mode == "flow_match_with_time_res":
        return ResidualFlowLightningGenerator(cfg)
    return StandardLightningGenerator(cfg)


def build_profiler(args: argparse.Namespace):
    if args.profiler == "none":
        return None

    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    if args.profiler == "simple":
        return SimpleProfiler(dirpath=profile_dir, filename="simple_profiler")
    return PyTorchProfiler(
        dirpath=profile_dir,
        filename="pytorch_profiler",
        export_to_chrome=True,
        row_limit=100,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RunPod single-node ElectraI trainer")
    parser.add_argument(
        "--config",
        default="examples/QM9/experiment_r1/config_runpod_8k.yml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--profiler",
        choices=("none", "simple", "pytorch"),
        default="none",
        help="Enable a Lightning profiler for short diagnostic runs.",
    )
    parser.add_argument(
        "--profile-dir",
        default="examples/QM9/experiment_r1/logs/profiler",
        help="Directory for profiler output.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=None,
        help="Optional Lightning limit_train_batches value for short profiling runs.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=None,
        help="Optional Lightning limit_val_batches value for short profiling runs.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override cfg.epochs. Leave unset when resuming a long run from checkpoint.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum optimizer steps. Useful for short training profiler runs.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation only, loading last.ckpt when available.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="Override cfg.wandb_mode, useful for profiler-only runs.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.wandb_mode is not None:
        cfg.wandb_mode = args.wandb_mode
    datamodule = instantiate(cfg.data)
    lit_model = choose_lightning_module(cfg)

    wandb_mode = str(getattr(cfg, "wandb_mode", "disabled")).lower()
    os.environ["WANDB_MODE"] = wandb_mode
    if wandb_mode != "disabled":
        from lightning.pytorch.loggers import WandbLogger

        wandb_run_id = getattr(cfg, "wandb_run_id", None)
        wandb_run_name = getattr(cfg, "wandb_run_name", None) or wandb_run_id
        wandb_kwargs = {
            "project": cfg.wb_pname,
            "entity": cfg.entity,
            "config": vars(cfg),
        }
        if wandb_run_id:
            wandb_kwargs.update(
                id=str(wandb_run_id),
                name=str(wandb_run_name),
                resume=getattr(cfg, "wandb_resume", "allow"),
            )
        logger = WandbLogger(**wandb_kwargs)
    else:
        logger = None

    ckpt_path = Path(getattr(cfg, "ckpt_path", "./checkpoints"))
    ckpt_path.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_path,
        monitor=getattr(cfg, "checkpoint_monitor", "val_loss"),
        save_top_k=2,
        mode=getattr(cfg, "checkpoint_mode", "min"),
        filename="ckpt_{epoch:02d}_{val_loss:.6f}",
        save_last=True,
    )

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    world_size = int(os.environ.get("WORLD_SIZE", local_world_size))
    num_nodes = max(1, world_size // max(1, local_world_size))
    trainer = Trainer(
        max_epochs=args.max_epochs if args.max_epochs is not None else int(cfg.epochs),
        logger=logger,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="epoch")],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=cfg.precision,
        devices="auto",
        num_nodes=num_nodes,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        log_every_n_steps=1,
        gradient_clip_val=getattr(cfg, "gradient_clip_value", 1.0),
        max_steps=args.max_steps,
        limit_train_batches=(
            args.limit_train_batches if args.limit_train_batches is not None else 1.0
        ),
        limit_val_batches=(
            args.limit_val_batches if args.limit_val_batches is not None else 1.0
        ),
        profiler=build_profiler(args),
    )

    ckpt = ckpt_path / "last.ckpt"
    if args.validate_only:
        trainer.validate(
            lit_model,
            datamodule=datamodule,
            ckpt_path=ckpt if ckpt.exists() else None,
        )
        return

    trainer.fit(
        lit_model,
        datamodule=datamodule,
        ckpt_path=ckpt if ckpt.exists() else None,
    )


if __name__ == "__main__":
    main()

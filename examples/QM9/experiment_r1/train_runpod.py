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


def main() -> None:
    parser = argparse.ArgumentParser(description="RunPod single-node ElectraI trainer")
    parser.add_argument(
        "--config",
        default="examples/QM9/experiment_r1/config_runpod_8k.yml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
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
        max_epochs=int(cfg.epochs),
        logger=logger,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="epoch")],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=cfg.precision,
        devices="auto",
        num_nodes=num_nodes,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        log_every_n_steps=1,
        gradient_clip_val=getattr(cfg, "gradient_clip_value", 1.0),
    )

    ckpt = ckpt_path / "last.ckpt"
    trainer.fit(
        lit_model,
        datamodule=datamodule,
        ckpt_path=ckpt if ckpt.exists() else None,
    )


if __name__ == "__main__":
    main()

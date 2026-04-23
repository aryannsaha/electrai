from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from electrai.lightning import LightningGenerator
from electrai.lightning_flow import LightningFlowMatch
from electrai.lightning_flow_cond_aug import LightningFlowMatchCondAug
from electrai.lightning_flow_pretrained_cond import LightningFlowMatchPretrainedCond
from electrai.lightning_flow_reflow import LightningFlowMatchReflow
from electrai.lightning_flow_residual import LightningFlowMatchResidual
from electrai.lightning_flow_residual_displacement import (
    LightningFlowMatchResidualDisplacement,
)
from electrai.lightning_flow_test import LightningFlowTest
from electrai.lightning_w_time import LightningGenerator as LightningGeneratorWithTime
from electrai.lightning_w_time_flow import LightningGenerator as LightningGeneratorFlowWithTime
from electrai.lightning_w_time_flow_res import (
    LightningGenerator as LightningGeneratorFlowWithTimeResidual,
)


def train(args):
    # -----------------------------
    # Load YAML config
    # -----------------------------
    config_path = Path(args.config)
    with Path.open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)

    # -----------------------------
    # Data
    # -----------------------------
    datamodule = instantiate(cfg.data)

    # -----------------------------
    # Model (LightningModule handles architecture + loss + optimizer)
    # -----------------------------
    training_mode = getattr(cfg, 'training_mode', 'default')
    if training_mode == 'flow_match':
        lit_model = LightningFlowMatch(cfg)
    elif training_mode == 'flow_match_reflow':
        lit_model = LightningFlowMatchReflow(cfg)
    elif training_mode == 'flow_match_residual':
        lit_model = LightningFlowMatchResidual(cfg)
    elif training_mode == 'flow_match_residual_displacement':
        lit_model = LightningFlowMatchResidualDisplacement(cfg)
    elif training_mode == 'flow_match_cond_aug':
        lit_model = LightningFlowMatchCondAug(cfg)
    elif training_mode == 'flow_match_pretrained_cond':
        lit_model = LightningFlowMatchPretrainedCond(cfg)
    elif training_mode == 'flow_match_test':
        lit_model = LightningFlowTest(cfg)
    elif training_mode == 'regression_with_time':
        lit_model = LightningGeneratorWithTime(cfg)
    elif training_mode == 'flow_match_with_time':
        lit_model = LightningGeneratorFlowWithTime(cfg)
    elif training_mode == 'flow_match_with_time_res':
        lit_model = LightningGeneratorFlowWithTimeResidual(cfg)
    else:
        lit_model = LightningGenerator(cfg)

    ckpt_path = Path(getattr(cfg, 'ckpt_path', './checkpoints'))

    # -----------------------------
    # Weight initialization from pretrained checkpoint
    # -----------------------------
    pretrain_ckpt = getattr(cfg, "pretrain_ckpt_path", None)
    if pretrain_ckpt and Path(pretrain_ckpt).exists() and not (ckpt_path / 'last.ckpt').exists():
        # Pretraining checkpoints are trusted local artifacts and may contain
        # metadata objects such as SimpleNamespace in addition to tensor weights.
        ckpt_data = torch.load(
            pretrain_ckpt,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = ckpt_data.get("state_dict", ckpt_data)
        if not isinstance(state_dict, dict):
            raise TypeError(
                "Expected a checkpoint dict or raw state_dict, got "
                f"{type(state_dict)!r} from {pretrain_ckpt}."
            )
        model_state = lit_model.state_dict()
        filtered_state_dict = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        skipped = sorted(set(state_dict) - set(filtered_state_dict))
        missing, unexpected = lit_model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrain_ckpt}")
        print(f"  Missing keys:    {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        print(f"  Skipped keys:    {len(skipped)}")

    # -----------------------------
    # Logging and callbacks
    # -----------------------------
    wandb_mode = getattr(cfg, 'wandb_mode', 'disabled').lower()
    os.environ['WANDB_MODE'] = wandb_mode
    if wandb_mode != 'disabled':
        from lightning.pytorch.loggers import WandbLogger

        wandb_run_id = getattr(cfg, 'wandb_run_id', None)
        wandb_logger = WandbLogger(
            project=cfg.wb_pname,
            entity=cfg.entity,
            config=vars(cfg),
            id=wandb_run_id,
            resume='allow' if wandb_run_id else None,
        )
    else:
        wandb_logger = None
    monitor = getattr(cfg, 'checkpoint_monitor', 'val_loss')
    mode = getattr(cfg, 'checkpoint_mode', 'min')
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_path,
        monitor=monitor,
        save_top_k=5,
        mode=mode,
        filename=f'ckpt_{{epoch:02d}}_{{{monitor}:.6f}}',
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # -----------------------------
    # Trainer
    # -----------------------------
    local_world_size = int(
        os.environ.get('LOCAL_WORLD_SIZE', torch.cuda.device_count())
    )
    world_size = int(os.environ.get('WORLD_SIZE', local_world_size))
    num_nodes = max(1, world_size // local_world_size)
    trainer = Trainer(
        max_epochs=int(cfg.epochs),
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision=cfg.precision,
        devices='auto',
        num_nodes=num_nodes,
        strategy='ddp',
        log_every_n_steps=1,
        gradient_clip_val=getattr(cfg, 'gradient_clip_value', 1.0),
    )

    # -----------------------------
    # Train
    # -----------------------------
    ckpt = ckpt_path / 'last.ckpt'
    trainer.fit(
        lit_model, datamodule=datamodule, ckpt_path=ckpt if ckpt.exists() else None
    )

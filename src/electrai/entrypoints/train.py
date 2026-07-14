from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

from electrai.lightning import LightningGenerator
from electrai.lightning_flow import LightningFlowMatch
from electrai.lightning_flow_cond_aug import LightningFlowMatchCondAug
from electrai.lightning_w_time_flow_res import (
    LightningGenerator as LightningGeneratorFlowWithTimeResidual,
)
from electrai.lightning_w_time_flow_res_norm import (
    LightningGenerator as LightningGeneratorFlowWithTimeResidualNorm,
)


class StopAfterCheckpointForRequeue(Callback):
    def __init__(self, ckpt_path: Path, flag_path: Path):
        self.ckpt_path = ckpt_path
        self.flag_path = flag_path
        self._seen_checkpoints: dict[Path, int] = {}

    def on_fit_start(self, trainer, pl_module) -> None:
        self._seen_checkpoints = self._checkpoint_state()

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        checkpoint = (
            self._new_or_updated_checkpoint() if trainer.is_global_zero else None
        )
        checkpoint_text = str(checkpoint) if checkpoint is not None else ''
        checkpoint_text = trainer.strategy.broadcast(checkpoint_text, src=0)
        if not checkpoint_text:
            return

        if trainer.is_global_zero:
            self.flag_path.parent.mkdir(parents=True, exist_ok=True)
            self.flag_path.write_text(
                f"checkpoint={checkpoint_text}\nepoch={trainer.current_epoch}\n"
            )
            pl_module.print(
                "Validation checkpoint written; stopping so SLURM can requeue."
            )
        trainer.should_stop = True

    def _checkpoint_state(self) -> dict[Path, int]:
        if not self.ckpt_path.exists():
            return {}
        return {
            checkpoint: checkpoint.stat().st_mtime_ns
            for checkpoint in self.ckpt_path.glob("*.ckpt")
            if checkpoint.is_file()
        }

    def _new_or_updated_checkpoint(self) -> Path | None:
        current = self._checkpoint_state()
        for checkpoint, mtime_ns in sorted(
            current.items(), key=lambda item: item[1], reverse=True
        ):
            if self._seen_checkpoints.get(checkpoint) != mtime_ns:
                self._seen_checkpoints = current
                return checkpoint
        self._seen_checkpoints = current
        return None


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
    elif training_mode == 'flow_match_cond_aug':
        lit_model = LightningFlowMatchCondAug(cfg)
    elif training_mode == 'flow_match_with_time_res':
        lit_model = LightningGeneratorFlowWithTimeResidual(cfg)
    elif training_mode == 'flow_match_with_time_res_norm':
        lit_model = LightningGeneratorFlowWithTimeResidualNorm(cfg)
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
    callbacks = [checkpoint_cb, lr_monitor]
    requeue_flag_path = os.environ.get('ELECTRAI_REQUEUE_AFTER_CHECKPOINT_FLAG')
    if requeue_flag_path:
        callbacks.append(
            StopAfterCheckpointForRequeue(ckpt_path, Path(requeue_flag_path))
        )

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
        callbacks=callbacks,
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

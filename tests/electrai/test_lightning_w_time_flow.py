from __future__ import annotations

from types import MethodType
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from electrai.lightning_w_time_flow import LightningGenerator


def make_cfg():
    model_cfg = OmegaConf.create(
        {
            "_target_": "torch.nn.Identity",
        }
    )
    return SimpleNamespace(
        model=model_cfg,
        lr=1e-3,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.99,
        warmup_length=1,
        epochs=4,
        n_inference_steps=3,
        eps=1e-4,
    )


def make_batch():
    return {
        "data": torch.full((2, 1, 2, 2, 2), 0.25),
        "label": torch.full((2, 1, 2, 2, 2), 0.5),
        "index": torch.tensor([0, 1]),
    }


def capture_logs(module):
    logged = []

    def fake_log(self, name, value, **kwargs):
        logged.append((name, value, kwargs))

    module.log = MethodType(fake_log, module)
    return logged


def test_training_step_logs_full_wandb_metric_bundle():
    module = LightningGenerator(make_cfg())
    module._trainer = SimpleNamespace(current_epoch=3)
    module._random_t_metrics = MethodType(
        lambda self, batch: (torch.tensor(1.25), torch.tensor(0.5)),
        module,
    )
    logged = capture_logs(module)

    loss = module.training_step(make_batch())

    assert torch.isclose(loss, torch.tensor(1.25))
    names = [name for name, _value, _kwargs in logged]
    assert names == [
        "epoch",
        "train_loss_step",
        "train_loss",
        "train_random_t_sq_fro_step",
        "train_random_t_sq_fro",
        "train_random_t_nmae_step",
        "train_random_t_nmae",
    ]

    log_kwargs = {name: kwargs for name, _value, kwargs in logged}
    assert log_kwargs["epoch"]["on_epoch"] is True
    assert log_kwargs["train_loss_step"]["on_step"] is True
    assert log_kwargs["train_loss"]["on_epoch"] is True
    assert log_kwargs["train_random_t_nmae"]["on_epoch"] is True


def test_validation_step_logs_epoch_and_rollout_metrics():
    module = LightningGenerator(make_cfg())
    module._trainer = SimpleNamespace(current_epoch=3)
    module._random_t_metrics = MethodType(
        lambda self, batch: (torch.tensor(2.0), torch.tensor(0.75)),
        module,
    )
    module._sample = MethodType(lambda self, x: x + 0.1, module)
    logged = capture_logs(module)

    loss = module.validation_step(make_batch())

    assert torch.isclose(loss, torch.tensor(2.0))
    names = [name for name, _value, _kwargs in logged]
    assert names == [
        "epoch",
        "val_loss_step",
        "val_loss",
        "val_random_t_sq_fro",
        "val_random_t_nmae",
        "val_rollout_nmae",
        "val_rollout_sq_fro",
    ]

    log_kwargs = {name: kwargs for name, _value, kwargs in logged}
    assert log_kwargs["val_loss_step"]["on_step"] is True
    assert log_kwargs["val_loss"]["on_epoch"] is True
    assert log_kwargs["val_rollout_nmae"]["sync_dist"] is True
    assert log_kwargs["val_rollout_sq_fro"]["sync_dist"] is True

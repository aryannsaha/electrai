from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from electrai.lightning_w_time_flow_res import LightningGenerator


class ConstantResidual(torch.nn.Module):
    def forward(self, x, t, cond=None):
        del t, cond
        return torch.ones_like(x)


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
        n_inference_steps=1,
        eps=1e-4,
        source_distribution="zero",
    )


def test_zero_charge_residual_removes_per_sample_mean():
    module = LightningGenerator(make_cfg())
    residual = torch.arange(16, dtype=torch.float32).reshape(2, 1, 2, 2, 2)

    projected = module._zero_charge_residual(residual)

    assert torch.allclose(projected.flatten(start_dim=1).sum(dim=1), torch.zeros(2))


def test_flow_metrics_projects_predicted_residual_before_loss():
    module = LightningGenerator(make_cfg())
    module.model = ConstantResidual()
    x_0 = torch.zeros((2, 1, 2, 2, 2))
    x_1 = torch.zeros_like(x_0)
    cond = torch.full_like(x_0, 0.25)

    sq_fro, nmae = module._flow_metrics(x_0, x_1, cond=cond)

    assert torch.isclose(sq_fro, torch.tensor(0.0))
    assert torch.isclose(nmae, torch.tensor(0.0))


def test_sample_projects_residual_before_adding_condition():
    module = LightningGenerator(make_cfg())
    module.model = ConstantResidual()
    cond = torch.full((2, 1, 2, 2, 2), 0.25)

    pred = module._sample(cond)

    assert torch.allclose(pred, cond)

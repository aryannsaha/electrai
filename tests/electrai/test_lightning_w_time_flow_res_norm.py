from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from electrai.lightning_w_time_flow_res_norm import LightningGenerator


class FixedResidual(torch.nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, x, t, cond=None):
        del t, cond
        return self.output.to(device=x.device, dtype=x.dtype).expand_as(x)


def make_cfg(**overrides):
    cfg = {
        "model": OmegaConf.create({"_target_": "torch.nn.Identity"}),
        "lr": 1e-3,
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.99,
        "warmup_length": 1,
        "epochs": 4,
        "n_inference_steps": 1,
        "eps": 1e-4,
        "source_distribution": "zero",
        "residual_normalize": True,
        "residual_mean": 2.0,
        "residual_min": 0.0,
        "residual_max": 6.0,
    }
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_flow_metrics_train_in_normalized_space_but_score_physical_density():
    module = LightningGenerator(make_cfg())
    physical_residual = torch.tensor(
        [-1.0, 1.0, -2.0, 2.0], dtype=torch.float32
    ).reshape(1, 1, 1, 2, 2)
    normalized_residual = module._normalize_residual(physical_residual)
    module.model = FixedResidual(normalized_residual)
    cond = torch.full_like(normalized_residual, 3.0)

    sq_fro, nmae = module._flow_metrics(
        torch.zeros_like(normalized_residual), normalized_residual, cond=cond
    )

    assert torch.isclose(sq_fro, torch.tensor(0.0))
    assert torch.isclose(nmae, torch.tensor(0.0))


def test_sample_denormalizes_residual_before_adding_condition():
    module = LightningGenerator(make_cfg())
    constant_physical_residual = torch.full((1, 1, 2, 2, 2), 6.0)
    module.model = FixedResidual(module._normalize_residual(constant_physical_residual))
    cond = torch.full((1, 1, 2, 2, 2), 0.25)

    pred = module._sample(cond)

    torch.testing.assert_close(pred, cond)

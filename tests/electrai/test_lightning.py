from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from electrai.lightning import LightningGenerator


def make_cfg():
    model_cfg = OmegaConf.create(
        {
            '_target_': 'electrai.model.srgan_layernorm_pbc.GeneratorResNet',
            'in_channels': 1,
            'out_channels': 1,
            'n_residual_blocks': 1,
            'n_upscale_layers': 0,
            'n_channels': 8,
            'kernel_size1': 3,
            'kernel_size2': 3,
            'normalize': True,
            'use_checkpoint': False,
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
    )


def test_metrics_from_tensor_batch_returns_finite_explicit_nmae():
    module = LightningGenerator(make_cfg())
    batch = {
        'data': torch.rand(2, 1, 4, 4, 4) + 0.1,
        'label': torch.rand(2, 1, 4, 4, 4) + 0.1,
        'index': torch.tensor([0, 1]),
    }

    metrics = module._metrics_from_batch(batch)

    assert set(metrics) == {'loss', 'nmae'}
    assert module._infer_batch_size(batch) == 2
    for value in metrics.values():
        assert torch.isfinite(value)
        assert value.ndim == 0
    torch.testing.assert_close(metrics['loss'], metrics['nmae'])


def test_metrics_from_list_batch_support_variable_grid_sizes():
    module = LightningGenerator(make_cfg())
    batch = {
        'data': [torch.rand(1, 4, 4, 4) + 0.1, torch.rand(1, 5, 5, 5) + 0.1],
        'label': [torch.rand(1, 4, 4, 4) + 0.1, torch.rand(1, 5, 5, 5) + 0.1],
        'index': [3, 7],
    }

    metrics = module._metrics_from_batch(batch)

    assert set(metrics) == {'loss', 'nmae'}
    assert module._infer_batch_size(batch) == 2
    for value in metrics.values():
        assert torch.isfinite(value)
        assert value.ndim == 0
    torch.testing.assert_close(metrics['loss'], metrics['nmae'])

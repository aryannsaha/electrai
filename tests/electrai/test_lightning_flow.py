from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from electrai.lightning_flow import LightningFlowMatch


def make_cfg():
    model_cfg = OmegaConf.create(
        {
            '_target_': 'electrai.model.flow_match_resnet.FlowMatchGeneratorResNet',
            'in_channels': 2,
            'out_channels': 1,
            'n_channels': 8,
            'n_residual_blocks': 2,
            'kernel_size1': 3,
            'kernel_size2': 3,
            'state_channels': 1,
            'cond_channels': 1,
            'norm_groups': 4,
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
        n_sample_steps=4,
        val_rollout_steps=3,
        sample_solver='heun',
        val_use_ema=False,
        flow_loss_weight=1.0,
        endpoint_nmae_weight=0.25,
        endpoint_mass_weight=0.05,
        ema_rate=0.9,
    )


def make_tensor_batch():
    return {
        'data': torch.randn(2, 1, 4, 4, 4),
        'label': torch.rand(2, 1, 4, 4, 4) + 0.1,
        'index': torch.tensor([0, 1]),
    }


def test_objective_from_batch_returns_finite_hybrid_metrics():
    module = LightningFlowMatch(make_cfg())

    metrics = module._objective_from_batch(make_tensor_batch())

    assert set(metrics) == {'loss', 'flow_mse', 'endpoint_nmae', 'endpoint_mass'}
    for value in metrics.values():
        assert torch.isfinite(value)
        assert value.ndim == 0


def test_sample_supports_heun_solver_on_native_grid_shapes():
    module = LightningFlowMatch(make_cfg())
    cond = torch.randn(2, 1, 4, 5, 6)

    samples = module.sample(cond, n_steps=3, model=module.model, solver='heun')

    assert samples.shape == cond.shape


def test_rollout_metrics_accept_variable_sized_list_batches():
    module = LightningFlowMatch(make_cfg())
    batch = {
        'data': [torch.randn(1, 4, 4, 4), torch.randn(1, 5, 5, 5)],
        'label': [torch.rand(1, 4, 4, 4) + 0.1, torch.rand(1, 5, 5, 5) + 0.1],
        'index': [0, 1],
    }

    metrics = module._rollout_metrics_from_batch(
        batch,
        model=module.model,
        n_steps=2,
        solver='euler',
    )

    assert set(metrics) == {'rollout_nmae', 'rollout_mass'}
    for value in metrics.values():
        assert torch.isfinite(value)
        assert value.ndim == 0


def test_test_step_runs_on_cpu_without_cuda_events(tmp_path):
    module = LightningFlowMatch(make_cfg())
    tmp_dir = tmp_path / 'tmp'
    tmp_dir.mkdir()
    module.test_cfg = SimpleNamespace(
        log_dir=tmp_path,
        out_dir=None,
        tmp_dir=tmp_dir,
        save_pred=False,
    )
    module.on_test_start()

    outputs = module.test_step(
        {
            'data': torch.randn(1, 1, 4, 4, 4),
            'label': torch.rand(1, 1, 4, 4, 4) + 0.1,
            'index': torch.tensor([7]),
        }
    )

    assert {'target', 'index', 'nmae', 'mass_error', 'duration'} <= outputs.keys()
    assert outputs['duration'] >= 0.0
    assert torch.isfinite(outputs['nmae'])
    assert torch.isfinite(outputs['mass_error'])

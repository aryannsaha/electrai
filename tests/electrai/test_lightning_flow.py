from __future__ import annotations

from types import MethodType
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from electrai.lightning_flow import LightningFlowMatch
from electrai.lightning_flow_pretrained_cond import LightningFlowMatchPretrainedCond
from electrai.lightning_flow_reflow import LightningFlowMatchReflow
from electrai.model.srgan_layernorm_pbc import GeneratorResNet


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


def make_pretrained_cond_cfg(ckpt_path):
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
    condition_model_cfg = OmegaConf.create(
        {
            '_target_': 'electrai.model.srgan_layernorm_pbc.GeneratorResNet',
            'n_channels': 8,
            'n_residual_blocks': 2,
            'n_upscale_layers': 1,
            'kernel_size1': 3,
            'kernel_size2': 3,
            'normalize': False,
            'use_checkpoint': False,
        }
    )
    return SimpleNamespace(
        model=model_cfg,
        condition_model=condition_model_cfg,
        condition_model_ckpt=str(ckpt_path),
        condition_model_state_dict_prefix='model.',
        condition_model_strict=True,
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
        source_distribution='standard_gaussian',
    )


def make_tensor_batch():
    return {
        'data': torch.randn(2, 1, 4, 4, 4),
        'label': torch.rand(2, 1, 4, 4, 4) + 0.1,
        'index': torch.tensor([0, 1]),
    }


def test_objective_from_batch_returns_finite_hybrid_metrics():
    module = LightningFlowMatch(make_cfg())

    metrics = module._objective_from_batch(make_tensor_batch(), stage='train')

    assert set(metrics) == {'loss', 'flow_mse', 'endpoint_nmae', 'endpoint_mass'}
    for value in metrics.values():
        assert torch.isfinite(value)
        assert value.ndim == 0


def test_sample_supports_heun_solver_on_native_grid_shapes():
    module = LightningFlowMatch(make_cfg())
    cond = torch.randn(2, 1, 4, 5, 6)

    samples = module.sample(cond, n_steps=3, model=module.model, solver='heun')

    assert samples.shape == cond.shape


def test_sample_defaults_to_rollout_model_selected_by_config():
    module = LightningFlowMatch(make_cfg())
    cond = torch.randn(1, 1, 4, 4, 4)
    chosen = {}

    def fake_sample_state_impl(
        self,
        cond,
        *,
        model,
        n_steps,
        solver,
        stage,
        initial_state=None,
    ):
        del n_steps, solver, initial_state
        chosen[stage] = model
        return torch.zeros_like(cond)

    module._sample_state_impl = MethodType(fake_sample_state_impl, module)

    _ = module.sample(cond)
    _ = module.sample(cond, stage='test')

    assert chosen['sample'] is module.model
    assert chosen['test'] is module.model

    module.cfg.test_use_ema = True
    _ = module.sample(cond, stage='test')

    assert chosen['test'] is module.ema_model


def test_sample_from_source_uses_provided_initial_state():
    module = LightningFlowMatch(make_cfg())
    cond = torch.randn(2, 1, 4, 4, 4)
    source = torch.full_like(cond, 0.25)

    samples = module.sample_from_source(
        cond,
        source,
        n_steps=2,
        model=module.model,
        solver='euler',
    )

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
        stage='validation',
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


def test_reflow_objective_uses_fixed_source_and_target():
    module = LightningFlowMatchReflow(make_cfg())
    batch = {
        'data': torch.randn(2, 1, 4, 4, 4),
        'label': torch.rand(2, 1, 4, 4, 4) + 0.1,
        'source': torch.randn(2, 1, 4, 4, 4),
        'reflow_target': torch.rand(2, 1, 4, 4, 4) + 0.1,
        'index': torch.tensor([0, 1]),
    }

    metrics = module._objective_from_batch(batch, stage='train')

    assert set(metrics) == {'loss', 'flow_mse', 'endpoint_nmae', 'endpoint_mass'}
    for value in metrics.values():
        assert torch.isfinite(value)
        assert value.ndim == 0


def test_pretrained_condition_module_uses_frozen_superres_output(tmp_path):
    cond_model = GeneratorResNet(
        n_channels=8,
        n_residual_blocks=2,
        n_upscale_layers=1,
        kernel_size1=3,
        kernel_size2=3,
        normalize=False,
        use_checkpoint=False,
    ).eval()
    ckpt_path = tmp_path / 'condition_model.ckpt'
    torch.save(
        {
            'state_dict': {
                f'model.{key}': value
                for key, value in cond_model.state_dict().items()
            }
        },
        ckpt_path,
    )

    module = LightningFlowMatchPretrainedCond(make_pretrained_cond_cfg(ckpt_path))
    cond = torch.randn(2, 1, 4, 4, 4)

    expected = cond_model(cond)
    actual = module._condition_for_model(cond, stage='train')
    reference = module._reference_state_from_condition(cond)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(reference, expected)
    assert actual.shape == (2, 1, 8, 8, 8)
    module.train()
    assert module.condition_model.training is False
    assert all(not parameter.requires_grad for parameter in module.condition_model.parameters())

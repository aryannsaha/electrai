from __future__ import annotations

from types import MethodType
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from electrai.lightning_flow_pretrained_cond import LightningFlowMatchPretrainedCond
from electrai.model.srgan_layernorm_pbc import GeneratorResNet


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
        source_distribution='condition_only',
        source_mean_scale=1.0,
    )


def test_pretrained_condition_sample_uses_projected_sad_source_at_test_time(tmp_path):
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
    cond = torch.randn(1, 1, 4, 4, 4)
    expected = cond_model(cond)

    def zero_velocity(self, model, x, cond, t):
        del self, model, cond, t
        return torch.zeros_like(x)

    module._predict_velocity = MethodType(zero_velocity, module)

    actual = module.sample(
        cond,
        n_steps=1,
        model=module.model,
        solver='euler',
        stage='test',
    )

    torch.testing.assert_close(actual, expected)

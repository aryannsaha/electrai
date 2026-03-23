from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from electrai.model.flow_match_resnet import FlowMatchGeneratorResNet


def test_flow_match_resnet_preserves_native_grid_shape():
    model = FlowMatchGeneratorResNet(
        in_channels=2,
        out_channels=1,
        n_residual_blocks=2,
        n_channels=8,
        kernel_size1=3,
        kernel_size2=3,
        state_channels=1,
        cond_channels=1,
        norm_groups=4,
        use_checkpoint=False,
    )
    x = torch.randn(2, 2, 4, 5, 6)
    t = torch.rand(2)

    output = model(x, t)

    assert output.shape == (2, 1, 4, 5, 6)


def test_flow_match_resnet_uses_group_norm_instead_of_instance_norm():
    model = FlowMatchGeneratorResNet(
        n_residual_blocks=1,
        n_channels=8,
        state_channels=1,
        cond_channels=1,
        norm_groups=4,
        use_checkpoint=False,
    )

    modules = list(model.modules())

    assert any(isinstance(module, nn.GroupNorm) for module in modules)
    assert not any(isinstance(module, nn.InstanceNorm3d) for module in modules)


def test_flow_match_resnet_conditioning_changes_the_prediction():
    torch.manual_seed(0)
    model = FlowMatchGeneratorResNet(
        n_residual_blocks=1,
        n_channels=8,
        kernel_size1=3,
        kernel_size2=3,
        state_channels=1,
        cond_channels=1,
        norm_groups=4,
        use_checkpoint=False,
    )
    model.eval()

    state = torch.zeros(1, 1, 4, 4, 4)
    cond_zeros = torch.zeros(1, 1, 4, 4, 4)
    cond_ones = torch.ones(1, 1, 4, 4, 4)
    t = torch.tensor([0.5])

    with torch.no_grad():
        output_zeros = model(torch.cat([state, cond_zeros], dim=1), t)
        output_ones = model(torch.cat([state, cond_ones], dim=1), t)

    assert not torch.allclose(output_zeros, output_ones)


def test_flow_match_resnet_requires_consistent_channel_split():
    with pytest.raises(ValueError, match=r'state_channels \+ cond_channels must equal in_channels'):
        FlowMatchGeneratorResNet(
            in_channels=2,
            state_channels=1,
            cond_channels=2,
        )

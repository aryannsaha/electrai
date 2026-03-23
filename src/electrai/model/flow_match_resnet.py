"""
Rectified Flow ResNet for 3D charge density super-resolution.

Adapted from the RectifiedFlow repo (https://github.com/gnobitab/RectifiedFlow)
using the existing ElectrAI GeneratorResNet backbone.

Key differences from GeneratorResNet:
  - Sinusoidal timestep embedding with additive injection into residual blocks
  - No ReLU at output (velocity can be negative)
  - No charge normalization (that's a post-processing step at inference)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding, following the original RectifiedFlow/NCSN++ implementation."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb,
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


def _make_group_norm(num_channels: int, num_groups: int) -> nn.GroupNorm:
    groups = min(num_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class TimeResidualBlock(nn.Module):
    """Residual block with time and conditioning injection."""

    def __init__(
        self,
        in_features: int,
        temb_dim: int,
        cond_features: int,
        K: int = 3,
        norm_groups: int = 8,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.conv1 = nn.Conv3d(
            in_features,
            in_features,
            kernel_size=K,
            stride=1,
            padding='same',
            padding_mode='circular',
        )
        self.norm1 = _make_group_norm(in_features, norm_groups)
        self.act1 = nn.PReLU()

        self.time_dense = nn.Linear(temb_dim, in_features)
        self.cond_proj = nn.Conv3d(cond_features, in_features, kernel_size=1)
        self.cond_gate = nn.Linear(temb_dim, in_features)

        self.conv2 = nn.Conv3d(
            in_features,
            in_features,
            kernel_size=K,
            stride=1,
            padding='same',
            padding_mode='circular',
        )
        self.norm2 = _make_group_norm(in_features, norm_groups)

    def _forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        temb_act = torch.nn.functional.silu(temb)
        time_term = self.time_dense(temb_act)[:, :, None, None, None]
        cond_gate = torch.sigmoid(self.cond_gate(temb_act))[:, :, None, None, None]
        cond_term = self.cond_proj(cond)
        h = h + time_term + cond_gate * cond_term

        h = self.conv2(h)
        h = self.norm2(h)
        return x + h

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, temb, cond, use_reentrant=False)
        return self._forward(x, temb, cond)


class FlowMatchGeneratorResNet(nn.Module):
    """ResNet backbone for Rectified Flow, with timestep conditioning."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        n_residual_blocks: int = 16,
        n_channels: int = 64,
        kernel_size1: int = 5,
        kernel_size2: int = 3,
        state_channels: int = 1,
        cond_channels: int | None = None,
        norm_groups: int = 8,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.state_channels = state_channels
        self.cond_channels = (
            in_channels - state_channels if cond_channels is None else cond_channels
        )
        if self.cond_channels <= 0:
            raise ValueError('cond_channels must be positive for conditional flow matching.')
        if self.state_channels + self.cond_channels != in_channels:
            raise ValueError(
                'state_channels + cond_channels must equal in_channels '
                f'(got {self.state_channels} + {self.cond_channels} != {in_channels}).',
            )

        temb_dim = n_channels * 4
        self.temb_net = nn.Sequential(
            nn.Linear(n_channels, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )
        self.temb_input_dim = n_channels

        self.state_encoder = nn.Sequential(
            nn.Conv3d(
                self.state_channels,
                n_channels,
                kernel_size=kernel_size1,
                stride=1,
                padding='same',
                padding_mode='circular',
            ),
            nn.PReLU(),
        )
        self.cond_encoder = nn.Sequential(
            nn.Conv3d(
                self.cond_channels,
                n_channels,
                kernel_size=kernel_size1,
                stride=1,
                padding='same',
                padding_mode='circular',
            ),
            nn.PReLU(),
        )
        self.input_fuse = nn.Conv3d(n_channels * 2, n_channels, kernel_size=1)

        self.res_blocks = nn.ModuleList([
            TimeResidualBlock(
                n_channels,
                temb_dim,
                cond_features=n_channels,
                K=kernel_size2,
                norm_groups=norm_groups,
                use_checkpoint=use_checkpoint,
            )
            for _ in range(n_residual_blocks)
        ])

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                n_channels,
                n_channels,
                kernel_size=kernel_size2,
                stride=1,
                padding='same',
                padding_mode='circular',
            ),
            _make_group_norm(n_channels, norm_groups),
        )
        self.cond_skip = nn.Conv3d(n_channels, n_channels, kernel_size=1)

        self.conv3 = nn.Conv3d(
            n_channels,
            out_channels,
            kernel_size=kernel_size1,
            stride=1,
            padding='same',
            padding_mode='circular',
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        temb = get_timestep_embedding(t * 999, self.temb_input_dim)
        temb = self.temb_net(temb)

        state, cond = torch.split(
            x,
            [self.state_channels, self.cond_channels],
            dim=1,
        )
        state_feat = self.state_encoder(state)
        cond_feat = self.cond_encoder(cond)
        out1 = self.input_fuse(torch.cat([state_feat, cond_feat], dim=1))

        out = out1
        for block in self.res_blocks:
            out = block(out, temb, cond_feat)

        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = torch.add(out, self.cond_skip(cond_feat))
        return self.conv3(out)

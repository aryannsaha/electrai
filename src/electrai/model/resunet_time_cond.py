"""
Conditional ResUNet3D with sinusoidal time embedding.

This variant predicts a residual/state tensor while also receiving the original
input density as an extra conditioning channel.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / (half - 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock3DTime(nn.Module):
    def __init__(self, cin, cout, k, temb_dim, use_checkpoint=True):
        #cin -> channels in, cout -> channels out, k -> kernel size
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv1 = nn.Conv3d(cin, cout, k, padding=k // 2, padding_mode="circular")
        self.norm1 = nn.InstanceNorm3d(cout)
        self.act1 = nn.PReLU()
        self.time_proj = nn.Linear(temb_dim, cout)
        self.conv2 = nn.Conv3d(cout, cout, k, padding=k // 2, padding_mode="circular")
        self.norm2 = nn.InstanceNorm3d(cout)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act_out = nn.PReLU()

    def _forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.norm1(self.conv1(x)))
        h = h + self.time_proj(temb)[:, :, None, None, None]
        h = self.norm2(self.conv2(h))
        return self.act_out(h + self.skip(x))

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, temb, use_reentrant=False)
        return self._forward(x, temb)


class ResUNet3DTimeCond(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_channels,
        depth,
        n_residual_blocks,
        kernel_size,
        cond_channels=None,
        use_checkpoint=True,
    ):
        super().__init__()

        self.state_channels = in_channels
        self.cond_channels = in_channels if cond_channels is None else cond_channels
        self.model_in_channels = self.state_channels + self.cond_channels

        temb_dim = n_channels * 4
        self._temb_raw_dim = n_channels
        self.temb_net = nn.Sequential(
            nn.Linear(n_channels, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        self.in_conv = ResBlock3DTime(
            self.model_in_channels, n_channels, kernel_size, temb_dim, use_checkpoint
        )

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        ch = n_channels
        for _ in range(depth):
            self.enc_blocks.append(nn.ModuleList([
                ResBlock3DTime(ch, ch, kernel_size, temb_dim, use_checkpoint)
                for _ in range(n_residual_blocks)
            ]))
            self.downs.append(_downsample(ch, 2 * ch))
            ch *= 2

        # Bottleneck
        self.mid = nn.ModuleList([
            ResBlock3DTime(ch, ch, kernel_size, temb_dim, use_checkpoint)
            for _ in range(2 * n_residual_blocks)
        ])

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for _ in range(depth):
            self.ups.append(PeriodicUpsampleConv3d(ch, ch // 2))
            ch //= 2
            level = nn.ModuleList()
            level.append(ResBlock3DTime(2 * ch, ch, kernel_size, temb_dim, use_checkpoint))
            level.extend([
                ResBlock3DTime(ch, ch, kernel_size, temb_dim, use_checkpoint)
                for _ in range(n_residual_blocks - 1)
            ])
            self.dec_blocks.append(level)

        self.out_conv = nn.Conv3d(n_channels, out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, state_channels, H, W, D) residual/state tensor if cond is
               provided, otherwise concatenated [state, condition].
            t: (B,) timestep in [0, 1]
            cond: (B, cond_channels, H, W, D) original input density

        Returns:
            (B, out_channels, H, W, D) predicted residual/state tensor
        """
        if cond is not None:
            if x.shape[1] != self.state_channels:
                raise ValueError(
                    f"Expected {self.state_channels} state channels, got {x.shape[1]}."
                )
            if cond.shape[1] != self.cond_channels:
                raise ValueError(
                    f"Expected {self.cond_channels} condition channels, got {cond.shape[1]}."
                )
            if x.shape[2:] != cond.shape[2:]:
                raise ValueError(
                    "State and condition tensors must have matching spatial shapes, "
                    f"got {tuple(x.shape[2:])} and {tuple(cond.shape[2:])}."
                )
            x = torch.cat([x, cond], dim=1)
        elif x.shape[1] != self.model_in_channels:
            raise ValueError(
                "Conditional ResUNet3DTime expects either cond=... or a concatenated "
                f"[state, condition] tensor with {self.model_in_channels} channels, "
                f"got {x.shape[1]}."
            )

        temb = _sinusoidal_embedding(t * 999, self._temb_raw_dim)
        temb = self.temb_net(temb)

        skips = []
        out = self.in_conv(x, temb)

        for enc_level, down in zip(self.enc_blocks, self.downs, strict=False):
            for block in cast(nn.ModuleList, enc_level):
                out = block(out, temb)
            skips.append(out)
            out = down(out)

        for block in self.mid:
            out = block(out, temb)

        for up, dec_level in zip(self.ups, self.dec_blocks, strict=False):
            out = up(out)
            out = torch.cat([out, skips.pop()], dim=1)
            for block in cast(nn.ModuleList, dec_level):
                out = block(out, temb)

        out = self.out_conv(out)
        return out


class PeriodicUpsampleConv3d(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = nn.Conv3d(cin, cout, 3, padding=1, padding_mode="circular")
        self.norm = nn.InstanceNorm3d(cout)
        self.act = nn.PReLU()

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode="circular")
        x = self.up(x)
        x = x[..., 2:-2, 2:-2, 2:-2]
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


def _downsample(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, stride=2, padding=1, padding_mode="circular"),
        nn.InstanceNorm3d(cout),
        nn.PReLU(),
    )


ResUNet3dTimeCond = ResUNet3DTimeCond
ResUNet3DTime = ResUNet3DTimeCond
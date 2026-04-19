"""
GeneratorResNet with sinusoidal timestep embedding.

Adapted from srgan_layernorm_pbc.py — adds time conditioning for flow-matching
or diffusion use cases. The only structural change is:

  - A sinusoidal embedding of ``t`` is computed and projected via a small MLP.
  - Each residual block receives ``temb`` and adds it additively between the two
    convolutions (same pattern as flow_match_resnet.py).
  - ``forward`` gains a required ``t: torch.Tensor`` argument (shape ``[B]``,
    values in [0, 1]).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding (NCSN++ / RectifiedFlow convention)."""
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


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, temb_dim: int, K: int = 3, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.conv1 = nn.Conv3d(
            in_features, in_features, kernel_size=K, stride=1,
            padding="same", padding_mode="circular",
        )
        self.norm1 = nn.InstanceNorm3d(in_features)
        self.act1 = nn.PReLU()

        # Projects time embedding to feature dim and injects additively
        self.time_proj = nn.Linear(temb_dim, in_features)

        self.conv2 = nn.Conv3d(
            in_features, in_features, kernel_size=K, stride=1,
            padding="same", padding_mode="circular",
        )
        self.norm2 = nn.InstanceNorm3d(in_features)

    def _forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        # additive time injection between the two convolutions
        h = h + self.time_proj(torch.nn.functional.silu(temb))[:, :, None, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        return x + h

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, temb, use_reentrant=False)
        return self._forward(x, temb)


class PixelShuffle3d(nn.Module):
    def __init__(self, in_channels: int, upscale_factor: int = 2):
        assert in_channels % (upscale_factor**3) == 0
        super().__init__()
        self.u = upscale_factor
        self.Cin = in_channels

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, Cin*u**3, H, W, D)"""
        assert X.shape[1] == self.Cin
        u = self.u
        Cout = self.Cin // u**3
        out = X.reshape(-1, Cout, u, u, u, *X.shape[-3:])
        out = out.permute((0, 1, 5, 2, 6, 3, 7, 4))
        return out.reshape(-1, Cout, u * X.shape[-3], u * X.shape[-2], u * X.shape[-1])


class GeneratorResNetTime(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_residual_blocks: int = 16,
        n_upscale_layers: int = 2,
        n_channels: int = 64,
        kernel_size1: int = 5,
        kernel_size2: int = 3,
        normalize: bool = True,
        use_checkpoint: bool = True,
    ):
        """
        GeneratorResNet with sinusoidal time conditioning.

        Identical to GeneratorResNet except:
          - ``forward(x, t)`` requires a timestep tensor ``t`` of shape ``[B]``.
          - Each residual block receives the time embedding additively.

        n_channels       : feature width throughout the network
        kernel_size1     : kernel for first / last conv
        kernel_size2     : kernel inside residual blocks
        n_upscale_layers : each layer doubles spatial resolution (via PixelShuffle3d)
        normalize        : conserve electron count in output
        """
        super().__init__()
        self.n_upscale_layers = n_upscale_layers
        self.normalize = normalize
        self.use_checkpoint = use_checkpoint

        # ---- Time embedding ----
        temb_dim = n_channels * 4
        self.temb_input_dim = n_channels
        self.temb_net = nn.Sequential(
            nn.Linear(n_channels, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        # ---- First layer ----
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels, n_channels, kernel_size=kernel_size1,
                stride=1, padding="same", padding_mode="circular",
            ),
            nn.PReLU(),
        )

        # ---- Residual blocks (ModuleList — each block needs temb) ----
        self.res_blocks = nn.ModuleList([
            ResidualBlock(n_channels, temb_dim, K=kernel_size2, use_checkpoint=use_checkpoint)
            for _ in range(n_residual_blocks)
        ])

        # ---- Second conv post residual blocks ----
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                n_channels, n_channels, kernel_size=kernel_size2,
                stride=1, padding="same", padding_mode="circular",
            ),
            nn.InstanceNorm3d(n_channels),
        )

        # ---- Upsampling layers ----
        upsampling = []
        for _ in range(n_upscale_layers):
            upsampling += [
                nn.Conv3d(
                    n_channels, n_channels * 8, kernel_size=kernel_size2,
                    stride=1, padding="same", padding_mode="circular",
                ),
                nn.InstanceNorm3d(n_channels * 8),
                PixelShuffle3d(n_channels * 8, upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # ---- Final output layer ----
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                n_channels, out_channels, kernel_size=kernel_size1,
                stride=1, padding="same", padding_mode="circular",
            ),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input charge density, shape (B, in_channels, H, W, D)
            t: timestep, shape (B,), values in [0, 1]
        """
        # Compute time embedding once for the whole batch
        temb = get_timestep_embedding(t * 999, self.temb_input_dim)
        temb = self.temb_net(temb)  # (B, temb_dim)

        out1 = self.conv1(x)

        out = out1
        for block in self.res_blocks:
            out = block(out, temb)

        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)

        if self.normalize:
            upscale_factor = 8 ** self.n_upscale_layers
            out = out / torch.sum(out, axis=(-3, -2, -1))[..., None, None, None]
            out = (
                out
                * torch.sum(x, axis=(-3, -2, -1))[..., None, None, None]
                * upscale_factor
            )
        return out

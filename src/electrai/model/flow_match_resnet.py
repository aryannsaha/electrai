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
    """Sinusoidal timestep embedding, following the original RectifiedFlow/NCSN++ implementation.

    Parameters
    ----------
    timesteps : torch.Tensor
        1D tensor of timestep values, shape (B,).
    embedding_dim : int
        Dimension of the embedding.

    Returns
    -------
    torch.Tensor
        Embedding of shape (B, embedding_dim).
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class TimeResidualBlock(nn.Module):
    """Residual block with additive timestep injection.

    Matches the NCSN++ ResBlock style: after the first conv+norm+act,
    the time embedding is added to the feature map via a linear projection.
    """

    def __init__(self, in_features: int, temb_dim: int, K: int = 3, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.conv1 = nn.Conv3d(
            in_features, in_features, kernel_size=K,
            stride=1, padding="same", padding_mode="circular",
        )
        self.norm1 = nn.InstanceNorm3d(in_features)
        self.act1 = nn.PReLU()

        # Time embedding projection: temb_dim -> in_features
        self.dense = nn.Linear(temb_dim, in_features)

        self.conv2 = nn.Conv3d(
            in_features, in_features, kernel_size=K,
            stride=1, padding="same", padding_mode="circular",
        )
        self.norm2 = nn.InstanceNorm3d(in_features)

    def _forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        # Additive time injection: project temb and broadcast over spatial dims
        h = h + self.dense(torch.nn.functional.silu(temb))[:, :, None, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        return x + h

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, temb, use_reentrant=False)
        return self._forward(x, temb)


class FlowMatchGeneratorResNet(nn.Module):
    """ResNet backbone for Rectified Flow, with timestep conditioning.

    Architecture mirrors GeneratorResNet but with:
      - Sinusoidal time embedding -> MLP -> additive injection per block
      - No output activation (velocity field can be negative)
      - No charge normalization (applied as post-processing if needed)
      - n_upscale_layers should be 0 for faithful rectified flow
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        n_residual_blocks: int = 16,
        n_channels: int = 64,
        kernel_size1: int = 5,
        kernel_size2: int = 3,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Timestep embedding: sinusoidal -> MLP
        temb_dim = n_channels * 4
        self.temb_net = nn.Sequential(
            nn.Linear(n_channels, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )
        self.temb_input_dim = n_channels  # for get_timestep_embedding

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels, n_channels, kernel_size=kernel_size1,
                stride=1, padding="same", padding_mode="circular",
            ),
            nn.PReLU(),
        )

        # Residual blocks with time conditioning
        self.res_blocks = nn.ModuleList([
            TimeResidualBlock(
                n_channels, temb_dim, K=kernel_size2, use_checkpoint=use_checkpoint,
            )
            for _ in range(n_residual_blocks)
        ])

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                n_channels, n_channels, kernel_size=kernel_size2,
                stride=1, padding="same", padding_mode="circular",
            ),
            nn.InstanceNorm3d(n_channels),
        )

        # Final output layer — no activation (velocity can be negative)
        self.conv3 = nn.Conv3d(
            n_channels, out_channels, kernel_size=kernel_size1,
            stride=1, padding="same", padding_mode="circular",
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, D, H, W).
            Typically [z_t, LR_conditioning] concatenated along channel dim.
        t : torch.Tensor
            Timestep values, shape (B,), in [0, 1].

        Returns
        -------
        torch.Tensor
            Predicted velocity field, shape (B, out_channels, D, H, W).
        """
        # Timestep embedding
        temb = get_timestep_embedding(t * 999, self.temb_input_dim)
        temb = self.temb_net(temb)

        # First conv
        out1 = self.conv1(x)

        # Residual blocks with time conditioning
        out = out1
        for block in self.res_blocks:
            out = block(out, temb)

        # Post-residual conv + skip connection
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        # Output
        out = self.conv3(out)
        return out

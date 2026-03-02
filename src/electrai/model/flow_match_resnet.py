"""
Flow Matching variant of the GeneratorResNet architecture.

=== WHAT IS FLOW MATCHING? ===

Flow matching (also called "Rectified Flow") learns a velocity field v(x_t, t)
that transports samples from a noise distribution (t=0) to the data distribution (t=1).

During training:
  - We sample a random timestep t ~ LogitNormal(0, 1) (focus on mid-range)
  - We create an interpolant: x_t = (1 - t) * noise + t * data
  - The model predicts the velocity: v_pred = model(x_t, t)
  - The target velocity is simply: v_target = data - noise
  - Loss = MSE(v_pred, v_target)

During inference:
  - Start from pure noise x_0 ~ N(0, I)
  - Integrate the ODE: dx/dt = v(x_t, t) from t=0 to t=1 using Euler steps
  - The result x_1 is the generated sample

=== KEY DESIGN DECISIONS ===

1. Adaptive Layer Normalization (AdaLN) for time conditioning:
   Time is injected at EVERY residual block via learned scale/shift parameters
   that modulate the InstanceNorm output. This prevents InstanceNorm from
   destroying the time signal (a known failure mode when time is added once
   as a spatially-uniform offset).

2. LR-resolution processing with PixelShuffle upscaling:
   The model operates at LR resolution (like the original GeneratorResNet)
   and uses PixelShuffle3d to upscale to HR at the end. This is 8x faster
   per forward pass than processing at HR resolution.

3. Input: (B, 2, Nx_lr, Ny_lr, Nz_lr) -- [x_t_downsampled, LR]
   Output: (B, 1, Nx_hr, Ny_hr, Nz_hr) -- velocity at HR resolution

References:
  - Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
    https://arxiv.org/abs/2210.02747
  - Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data
    with Rectified Flow" (ICLR 2023)
    https://arxiv.org/abs/2209.03003
  - Esser et al., "Scaling Rectified Flow Transformers" (2024)
    https://arxiv.org/abs/2403.03206
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# =============================================================================
# TimeEmbedding: Converts a scalar timestep into a feature vector
# =============================================================================
class TimeEmbedding(nn.Module):
    """
    Maps a scalar timestep t in [0, 1] to a flat feature vector of size `dim`.

    Architecture:
      1. Sinusoidal positional encoding (like in transformers)
      2. Two-layer MLP with SiLU activation

    Output is a flat (B, dim) vector that gets passed to each AdaLN block.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : torch.Tensor
            Shape (B,) or (B, 1). Values in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (B, dim) -- flat time embedding vector.
        """
        t = t.view(-1)
        device = t.device

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device) / half_dim
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return self.mlp(embedding)  # (B, dim)


# =============================================================================
# AdaLNResidualBlock: Residual block with Adaptive Layer Normalization
# =============================================================================
# Time conditioning is injected at EVERY block via learned scale/shift
# parameters that modulate the InstanceNorm output. This follows the AdaLN
# pattern from DiT (Peebles & Xie, 2023).
#
# Key: modulation is applied AFTER normalization, so InstanceNorm cannot
# destroy the time signal. Zero-init ensures blocks start as identity.
# =============================================================================
class AdaLNResidualBlock(nn.Module):
    def __init__(self, in_features, time_dim, K=3, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.conv1 = nn.Conv3d(
            in_features,
            in_features,
            kernel_size=K,
            stride=1,
            padding="same",
            padding_mode="circular",
        )
        self.norm1 = nn.InstanceNorm3d(in_features)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv3d(
            in_features,
            in_features,
            kernel_size=K,
            stride=1,
            padding="same",
            padding_mode="circular",
        )
        self.norm2 = nn.InstanceNorm3d(in_features)

        # Modulation: time_dim -> 4 * C (scale1, shift1, scale2, shift2)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 4 * in_features),
        )
        # Zero-init: block starts as identity (stable training)
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def _block_fn(self, x, t_emb):
        """Inner computation for gradient checkpointing."""
        mod = self.modulation(t_emb)  # (B, 4*C)
        scale1, shift1, scale2, shift2 = mod.chunk(4, dim=1)
        # Reshape for broadcasting: (B, C) -> (B, C, 1, 1, 1)
        scale1 = scale1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        shift1 = shift1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        scale2 = scale2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        shift2 = shift2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        h = self.conv1(x)
        h = self.norm1(h) * (1 + scale1) + shift1  # AdaLN
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h) * (1 + scale2) + shift2  # AdaLN
        return h

    def forward(self, x, t_emb):
        if self.use_checkpoint and self.training:
            return x + checkpoint(self._block_fn, x, t_emb, use_reentrant=False)
        else:
            return x + self._block_fn(x, t_emb)


# =============================================================================
# PixelShuffle3d: Rearranges channels into spatial dimensions for upsampling
# =============================================================================
class PixelShuffle3d(nn.Module):
    def __init__(self, in_channels, upscale_factor=2):
        assert in_channels % (upscale_factor**3) == 0
        super().__init__()
        self.u = upscale_factor
        self.Cin = in_channels

    def forward(self, X):
        assert X.shape[1] == self.Cin
        u = self.u
        Cout = self.Cin // u**3
        out = X.reshape(-1, Cout, u, u, u, *X.shape[-3:])
        out = out.permute((0, 1, 5, 2, 6, 3, 7, 4))
        return out.reshape(-1, Cout, u * X.shape[-3], u * X.shape[-2], u * X.shape[-1])


# =============================================================================
# FlowMatchGeneratorResNet: The main model for flow matching
# =============================================================================
class FlowMatchGeneratorResNet(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        n_residual_blocks=16,
        n_upscale_layers=1,
        n_channels=64,
        kernel_size1=5,
        kernel_size2=3,
        use_checkpoint=True,
    ):
        """
        Flow matching variant of GeneratorResNet.

        Processes at LR resolution (like the original) with PixelShuffle
        upscaling at the end. Time is injected via AdaLN at every residual
        block to prevent InstanceNorm from destroying the time signal.

        Parameters
        ----------
        in_channels : int
            Number of input channels. Default 2 = [x_t_downsampled, LR].
        out_channels : int
            Number of output channels. Default 1 = velocity field.
        n_residual_blocks : int
            Number of residual blocks in the trunk.
        n_upscale_layers : int
            Number of 2x PixelShuffle upsampling layers. Default 1 for 2x SR.
        n_channels : int
            Width of the hidden feature maps.
        kernel_size1 : int
            Kernel size for first and last conv layers.
        kernel_size2 : int
            Kernel size for residual blocks.
        use_checkpoint : bool
            Enable gradient checkpointing to save GPU memory.
        """
        super().__init__()
        self.n_upscale_layers = n_upscale_layers
        self.use_checkpoint = use_checkpoint

        # Time embedding: scalar t -> flat (B, n_channels) vector
        self.time_emb = TimeEmbedding(n_channels)

        # First layer: project 2-channel input to n_channels
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                n_channels,
                kernel_size=kernel_size1,
                stride=1,
                padding="same",
                padding_mode="circular",
            ),
            nn.PReLU(),
        )

        # AdaLN residual blocks with per-block time injection
        self.res_blocks = nn.ModuleList([
            AdaLNResidualBlock(
                n_channels,
                time_dim=n_channels,
                K=kernel_size2,
                use_checkpoint=use_checkpoint,
            )
            for _ in range(n_residual_blocks)
        ])

        # Post-residual conv layer
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                n_channels,
                n_channels,
                kernel_size=kernel_size2,
                stride=1,
                padding="same",
                padding_mode="circular",
            ),
            nn.InstanceNorm3d(n_channels),
        )

        # PixelShuffle upsampling (LR -> HR)
        upsampling = []
        for _ in range(n_upscale_layers):
            upsampling += [
                nn.Conv3d(
                    n_channels,
                    n_channels * 8,
                    kernel_size=kernel_size2,
                    stride=1,
                    padding="same",
                    padding_mode="circular",
                ),
                nn.InstanceNorm3d(n_channels * 8),
                PixelShuffle3d(n_channels * 8, upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final conv: no activation (velocity can be negative)
        self.conv3 = nn.Conv3d(
            n_channels,
            out_channels,
            kernel_size=kernel_size1,
            stride=1,
            padding="same",
            padding_mode="circular",
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 2, Nx_lr, Ny_lr, Nz_lr) — [x_t_downsampled, LR]
        t : torch.Tensor
            Shape (B,) — timestep in [0, 1]

        Returns
        -------
        torch.Tensor
            Shape (B, 1, Nx_hr, Ny_hr, Nz_hr) — velocity at HR resolution
        """
        if isinstance(x, list):
            return [
                self._forward(xi.unsqueeze(0), t[i : i + 1]).squeeze(0)
                for i, xi in enumerate(x)
            ]
        return self._forward(x, t)

    def _forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Step 1: First convolution at LR resolution
        out1 = self.conv1(x)

        # Step 2: Compute flat time embedding for all AdaLN blocks
        t_emb = self.time_emb(t)  # (B, n_channels)

        # Step 3: Residual blocks with per-block AdaLN time injection
        out = out1
        for block in self.res_blocks:
            out = block(out, t_emb)

        # Step 4: Post-residual conv + long skip connection
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        # Step 5: PixelShuffle upsampling (LR -> HR)
        out = self.upsampling(out)

        # Step 6: Final conv to velocity field (no activation)
        return self.conv3(out)

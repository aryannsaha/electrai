"""Residual-space flow matching ablation.

This module keeps the same backbone and training loop as the standard
`LightningFlowMatch`, but it changes the modeled state from the full target
density `x_1` to the residual `x_1 - cond`.

That means:
  - the model still conditions on the SAD guess;
  - the flow is learned in the correction space instead of the full density;
  - the final density prediction is reconstructed as `cond + residual`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from electrai.lightning_flow import LightningFlowMatch


class LightningFlowMatchResidual(LightningFlowMatch):
    """Flow matching in residual space.

    This class is intentionally small: it reuses the base class machinery and
    only overrides the semantic pieces that define "what state are we modeling?"
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.residual_density_projection = getattr(
            cfg, 'residual_density_projection', 'softplus',
        ).lower()
        self.residual_density_floor = float(
            getattr(cfg, 'residual_density_floor', 0.0),
        )
        self.residual_softplus_beta = float(
            getattr(cfg, 'residual_softplus_beta', 4.0),
        )
        self.residual_negative_penalty_weight = float(
            getattr(cfg, 'residual_negative_penalty_weight', 0.1),
        )

    def _validate_residual_shapes(
        self,
        cond: torch.Tensor,
        other: torch.Tensor,
        *,
        context: str,
    ) -> None:
        """Ensure the residual-space ablation only runs on matching grids."""
        if cond.shape != other.shape:
            raise ValueError(
                'Residual-space flow matching requires the condition and target '
                f'to share a shape during {context}, got '
                f'{tuple(cond.shape)} and {tuple(other.shape)}.',
            )

    def _target_state(self, cond: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Model the residual correction instead of the full density."""
        self._validate_residual_shapes(cond, x_1, context='target construction')
        return x_1 - cond

    def _prediction_from_state(
        self,
        state: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct the physical density from the residual state.

        A raw additive residual can flip positive densities negative. The
        residual ablation therefore supports a positivity-preserving projection
        so the final density remains physical at train and sample time.
        """
        self._validate_residual_shapes(cond, state, context='prediction reconstruction')
        raw_density = cond + state

        if self.residual_density_projection in {'identity', 'none'}:
            return raw_density
        if self.residual_density_projection == 'clamp':
            return raw_density.clamp_min(self.residual_density_floor)
        if self.residual_density_projection == 'softplus':
            shifted = raw_density - self.residual_density_floor
            return (
                F.softplus(shifted, beta=self.residual_softplus_beta)
                + self.residual_density_floor
            )

        raise ValueError(
            'Unknown residual_density_projection='
            f'{self.residual_density_projection!r}. Expected one of '
            "'none', 'clamp', or 'softplus'.",
        )

    def _reference_state_from_condition(self, cond: torch.Tensor) -> torch.Tensor:
        """Residual-space sampling starts from a zero-mean latent state."""
        return torch.zeros_like(cond)

    def _sample_source_state(
        self,
        cond: torch.Tensor,
        *,
        reference_state: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the residual-space source state.

        The default residual ablation still uses a Gaussian latent source so it
        isolates the effect of residual-space modeling. Follow-up ablations can
        switch this to a deterministic zero residual start.
        """
        del cond
        if self.source_distribution in {'standard', 'gaussian', 'standard_gaussian'}:
            return torch.randn_like(reference_state)

        if self.source_distribution in {'zero', 'zeros', 'deterministic_zero', 'residual_zero'}:
            return torch.zeros_like(reference_state)

        raise ValueError(
            f'Unknown residual source_distribution={self.source_distribution!r}. '
            "Expected 'standard_gaussian' or 'zero'.",
        )

    def _objective_terms(
        self,
        cond: torch.Tensor,
        x_1: torch.Tensor,
        *,
        stage: str,
    ):
        """Add an explicit penalty for any negative raw-density reconstruction."""
        bsz = x_1.shape[0]
        device = x_1.device

        model_cond = self._condition_for_model(cond, stage=stage)
        target_state = self._target_state(cond, x_1)
        x_0 = self._sample_source_state(cond, reference_state=target_state)
        t = torch.rand(bsz, device=device) * (1.0 - self.eps) + self.eps
        t_expand = t.view(bsz, 1, 1, 1, 1)
        z_t = t_expand * target_state + (1.0 - t_expand) * x_0
        target = target_state - x_0

        model_input = torch.cat([z_t, model_cond], dim=1)
        v_pred = self.model(model_input, t)

        flow_mse = ((v_pred - target) ** 2).mean()
        state_hat = z_t + (1.0 - t_expand) * v_pred
        raw_density_hat = cond + state_hat
        x_1_hat = self._prediction_from_state(state_hat, cond)
        endpoint_nmae = self.nmae_loss(x_1_hat, x_1)
        endpoint_mass = self.mass_loss(x_1_hat, x_1)
        negative_density_penalty = raw_density_hat.neg().relu().mean()

        loss = (
            self.flow_loss_weight * flow_mse
            + self.endpoint_nmae_weight * endpoint_nmae
            + self.endpoint_mass_weight * endpoint_mass
            + self.residual_negative_penalty_weight * negative_density_penalty
        )
        return {
            'loss': loss,
            'flow_mse': flow_mse,
            'endpoint_nmae': endpoint_nmae,
            'endpoint_mass': endpoint_mass,
            'negative_density_penalty': negative_density_penalty,
        }

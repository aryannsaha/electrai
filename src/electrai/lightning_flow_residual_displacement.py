"""Residual-space flow matching with displacement (x0) parameterization.

Extends the residual-space ablation with two changes vs experiment 7:

1. **Displacement parameterization**: the model predicts the remaining
   displacement to the endpoint (``target_state - z_t``) rather than the
   constant rectified-flow velocity (``target_state - x_0``).  The target
   shrinks to zero as t → 1, giving the model an explicit signal of how much
   correction is still needed at each noise level.

2. **No softplus projection**: the default projection is ``none``, which makes
   ``flow_mse`` and ``endpoint_nmae`` consistent (both minimised when
   ``state_hat = target_state``).  The softplus in experiment 7 created
   conflicting gradients because the two losses were driven to different optima.

Training target:  ``d* = target_state - z_t``   (= (1-t) * (target_state - x_0))
Predicted state:  ``state_hat = z_t + d_pred``
Sampling:         ``velocity = d_pred / (1 - t)`` (converts displacement → velocity)
"""

from __future__ import annotations

import torch

from electrai.lightning_flow_residual import LightningFlowMatchResidual


class LightningFlowMatchResidualDisplacement(LightningFlowMatchResidual):
    """Residual flow matching with displacement parameterization.

    Inherits all residual-space machinery (``_target_state``,
    ``_prediction_from_state``, ``_sample_source_state``, etc.) from
    ``LightningFlowMatchResidual`` and overrides only the objective and the
    velocity-prediction step used by the sampler.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Re-read with 'none' as the default so the softplus inconsistency
        # from experiment 7 doesn't carry over when the config omits the key.
        self.residual_density_projection = getattr(
            cfg, 'residual_density_projection', 'none',
        ).lower()

    def _objective_terms(
        self,
        cond: torch.Tensor,
        x_1: torch.Tensor,
        *,
        stage: str,
    ):
        """Flow loss with displacement parameterization.

        The model predicts ``d ≈ target_state - z_t`` (how far the current
        state needs to move to reach the endpoint).  This is equivalent to
        the velocity parameterisation at the optimal solution but differs in
        gradient scale: the target shrinks toward zero as t → 1, so the model
        gets a natural curriculum of increasingly refined corrections.
        """
        bsz = x_1.shape[0]
        device = x_1.device

        model_cond = self._condition_for_model(cond, stage=stage)
        target_state = self._target_state(cond, x_1)
        x_0 = self._sample_source_state(cond, reference_state=target_state)
        t = torch.rand(bsz, device=device) * (1.0 - self.eps) + self.eps
        t_expand = t.view(bsz, 1, 1, 1, 1)
        z_t = t_expand * target_state + (1.0 - t_expand) * x_0

        # Displacement to endpoint: (1-t) * (target_state - x_0)
        target = target_state - z_t

        model_input = torch.cat([z_t, model_cond], dim=1)
        d_pred = self.model(model_input, t)

        flow_mse = ((d_pred - target) ** 2).mean()

        # Predicted endpoint and density reconstruction
        state_hat = z_t + d_pred
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

    def _predict_velocity(
        self,
        model,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Convert the model's displacement prediction to a velocity.

        The model outputs ``d ≈ target_state - x`` (remaining displacement).
        The rectified-flow velocity implied by that displacement is
        ``v = d / (1 - t)``.  We clamp ``(1 - t)`` away from zero to avoid
        numerical blow-up at the very end of the trajectory.
        """
        model_input = torch.cat([x, cond], dim=1)
        d_pred = model(model_input, t)
        # t has shape (batch,); broadcast over spatial dims
        t_expand = t.view(t.shape[0], *([1] * (x.ndim - 1)))
        return d_pred / (1.0 - t_expand).clamp(min=1e-3)

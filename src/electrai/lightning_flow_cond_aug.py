"""Condition-augmentation flow matching ablation.

This ablation follows the "conditioning augmentation" idea from conditional
diffusion literature: the model still starts from a standard Gaussian source,
but the conditioning input is perturbed during training so the sampler learns
to tolerate imperfect conditions at inference time.

For this QM9-specific experiment, the augmentation is density-guided rather
than uniform: voxels near higher SAD density receive more Gaussian noise after
the guidance map is smoothed with an average-pooling kernel.
"""

from __future__ import annotations

import torch

from electrai.lightning_flow import LightningFlowMatch


class LightningFlowMatchCondAug(LightningFlowMatch):
    """Flow matching with density-guided condition augmentation."""

    def __init__(self, cfg):
        super().__init__(cfg)

        self.cond_aug_train = bool(getattr(cfg, 'cond_aug_train', True))
        self.cond_aug_validation = bool(getattr(cfg, 'cond_aug_validation', False))
        self.cond_aug_sampling = bool(getattr(cfg, 'cond_aug_sampling', False))

        self.cond_aug_base_std = float(getattr(cfg, 'cond_aug_base_std', 0.0))
        self.cond_aug_guided_scale = float(getattr(cfg, 'cond_aug_guided_scale', 0.0))
        self.cond_aug_power = float(getattr(cfg, 'cond_aug_power', 0.5))
        self.cond_aug_blur_kernel = int(getattr(cfg, 'cond_aug_blur_kernel', 1))
        self.cond_aug_eps = float(getattr(cfg, 'cond_aug_eps', 1e-8))
        self.cond_aug_clamp_min = float(getattr(cfg, 'cond_aug_clamp_min', 0.0))

    def _stage_uses_condition_augmentation(self, stage: str) -> bool:
        """Map trainer stages to augmentation policy.

        By default we only perturb the condition during training. Validation
        rollout and sampling stay clean so metrics reflect the actual inference
        setup unless the experiment config explicitly opts in.
        """
        if stage == 'train':
            return self.cond_aug_train
        if stage == 'validation':
            return self.cond_aug_validation
        if stage in {'validation_rollout', 'sample'}:
            return self.cond_aug_sampling
        return False

    def _augment_condition(self, cond: torch.Tensor) -> torch.Tensor:
        """Inject spatially concentrated Gaussian noise into the condition."""
        std_map = self._density_guided_std_map(
            cond,
            base_std=self.cond_aug_base_std,
            guided_scale=self.cond_aug_guided_scale,
            power=self.cond_aug_power,
            blur_kernel=self.cond_aug_blur_kernel,
            eps=self.cond_aug_eps,
        )
        noisy_cond = cond + std_map * torch.randn_like(cond)
        return noisy_cond.clamp_min(self.cond_aug_clamp_min)

    def _condition_for_model(self, cond: torch.Tensor, *, stage: str) -> torch.Tensor:
        """Use the clean condition unless this stage explicitly requests augmentation."""
        if not self._stage_uses_condition_augmentation(stage):
            return cond
        return self._augment_condition(cond)

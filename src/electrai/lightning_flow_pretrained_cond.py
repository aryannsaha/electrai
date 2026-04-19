from __future__ import annotations

from pathlib import Path

import torch
from hydra.utils import instantiate

from electrai.lightning_flow import LightningFlowMatch


class LightningFlowMatchPretrainedCond(LightningFlowMatch):
    """Flow matching conditioned on a frozen pretrained density predictor."""

    def __init__(self, cfg):
        super().__init__(cfg)

        self.condition_model = instantiate(cfg.condition_model)
        self.condition_model_ckpt = Path(getattr(cfg, 'condition_model_ckpt'))
        self.condition_model_state_dict_prefix = str(
            getattr(cfg, 'condition_model_state_dict_prefix', 'model.'),
        )
        self.condition_model_strict = bool(
            getattr(cfg, 'condition_model_strict', True),
        )

        self._load_condition_model_checkpoint()
        self.condition_model.requires_grad_(False)
        self.condition_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.condition_model.eval()
        return self

    def _extract_condition_model_state(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        prefix = self.condition_model_state_dict_prefix
        if prefix and any(key.startswith(prefix) for key in state_dict):
            return {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        return state_dict

    def _load_condition_model_checkpoint(self) -> None:
        if not self.condition_model_ckpt.exists():
            raise FileNotFoundError(
                'Condition-model checkpoint not found: '
                f'{self.condition_model_ckpt}',
            )

        checkpoint = torch.load(
            self.condition_model_ckpt,
            map_location='cpu',
            weights_only=False,
        )
        state_dict = checkpoint.get('state_dict', checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError(
                'Expected a checkpoint dict or a raw state_dict, got '
                f'{type(state_dict)!r}.',
            )

        condition_state = self._extract_condition_model_state(state_dict)
        self.condition_model.load_state_dict(
            condition_state,
            strict=self.condition_model_strict,
        )

    @torch.no_grad()
    def _project_condition(self, cond: torch.Tensor) -> torch.Tensor:
        projected = self.condition_model(cond)
        if not isinstance(projected, torch.Tensor):
            raise TypeError(
                'Pretrained condition model must return a tensor, got '
                f'{type(projected)!r}.',
            )
        return projected.to(device=cond.device, dtype=cond.dtype)

    def _condition_for_model(self, cond: torch.Tensor, *, stage: str) -> torch.Tensor:
        del stage
        return self._project_condition(cond)

    def _reference_state_from_condition(self, cond: torch.Tensor) -> torch.Tensor:
        return self._project_condition(cond)

    def _sample_source_state(
        self,
        cond: torch.Tensor,
        *,
        reference_state: torch.Tensor,
    ) -> torch.Tensor:
        cond_for_source = cond
        if self.source_distribution not in {
            'standard',
            'gaussian',
            'standard_gaussian',
        }:
            cond_for_source = self._project_condition(cond)
        return super()._sample_source_state(
            cond_for_source,
            reference_state=reference_state,
        )

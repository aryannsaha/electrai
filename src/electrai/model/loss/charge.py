from __future__ import annotations

import torch


def _flatten_nonbatch(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1)


def _electron_count(tensor: torch.Tensor) -> torch.Tensor:
    return _flatten_nonbatch(tensor).sum(dim=1).clamp_min(1e-12)


class NormMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        if isinstance(output, torch.Tensor):
            return self._forward(output, target)

        losses = []
        for out, tar in zip(output, target, strict=False):
            losses.append(self._forward(out.unsqueeze(0), tar.unsqueeze(0)))
        return torch.stack(losses).mean()

    def _forward(self, output, target):
        abs_error = torch.abs(output - target)
        error_sum = _flatten_nonbatch(abs_error).sum(dim=1)
        nelec = _electron_count(target)
        return (error_sum / nelec).mean()


class ElectronCountLoss(torch.nn.Module):
    def forward(self, output, target):
        if isinstance(output, torch.Tensor):
            return self._forward(output, target)

        losses = []
        for out, tar in zip(output, target, strict=False):
            losses.append(self._forward(out.unsqueeze(0), tar.unsqueeze(0)))
        return torch.stack(losses).mean()

    def _forward(self, output, target):
        output_count = _electron_count(output)
        target_count = _electron_count(target)
        return (torch.abs(output_count - target_count) / target_count).mean()

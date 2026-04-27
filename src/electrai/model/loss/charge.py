from __future__ import annotations

import torch


class NormMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = torch.nn.L1Loss(reduction="none")

    def forward(self, output, target):
        if isinstance(output, torch.Tensor):
            return self._forward(output, target)

        losses = []
        for out, tar in zip(output, target, strict=False):
            losses.append(self._forward(out.unsqueeze(0), tar.unsqueeze(0)))
        return torch.stack(losses).mean()

    def _forward(self, output, target):
        mae = self.mae(output, target)
        nelec = torch.sum(target, axis=(-3, -2, -1))
        nelec = torch.where(nelec.abs() > 0, nelec, torch.ones_like(nelec))
        mae = mae / nelec[..., None, None, None]
        return torch.sum(mae)


class SquaredFrobeniusLoss(torch.nn.Module):
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
        diff = output - target
        return diff.flatten(start_dim=1).pow(2).sum(dim=1).mean()

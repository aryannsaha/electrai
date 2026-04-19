from __future__ import annotations

import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningModule

from electrai.model.loss.charge import NormMAE
from electrai.model.resunet_flow import ResUNet3D


class ResUNetFlowWrapper(torch.nn.Module):
    """Small adapter that lets ResUNet3D participate in flow matching."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.target_channels = int(getattr(cfg, 'target_channels', 1))
        self.cond_channels = int(getattr(cfg, 'cond_channels', self.target_channels))

        model_cfg = getattr(cfg, 'model', None)
        if model_cfg is None:
            self.backbone = ResUNet3D(
                in_channels=self.target_channels,
                out_channels=self.target_channels,
                n_channels=32,
                depth=2,
                n_residual_blocks=2,
                kernel_size=3,
                use_checkpoint=True,
            )
        else:
            self.backbone = instantiate(model_cfg)

        self.input_adapter = torch.nn.Conv3d(
            self.target_channels + self.cond_channels + 1,
            self.target_channels,
            kernel_size=1,
        )

    def forward(
        self,
        state: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_channel = t.view(-1, 1, 1, 1, 1).expand(-1, 1, *state.shape[2:])
        model_input = torch.cat([state, cond, t_channel], dim=1)
        return self.backbone(self.input_adapter(model_input))


class LightningFlow(LightningModule):
    """Minimal Lightning flow-matching module for ResUNet3D."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.eps = float(getattr(cfg, 'eps', 1e-3))
        self.lr = float(getattr(cfg, 'lr', 1e-4))
        self.weight_decay = float(getattr(cfg, 'weight_decay', 0.0))
        self.n_sample_steps = int(getattr(cfg, 'n_sample_steps', 10))

        self.model = ResUNetFlowWrapper(cfg)
        self.flow_loss = torch.nn.MSELoss()
        self.nmae = NormMAE()

        self.save_hyperparameters(
            {
                'eps': self.eps,
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'n_sample_steps': self.n_sample_steps,
            },
        )

    def forward(self, state: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(state, cond, t)

    def _iter_batch_samples(self, batch):
        cond = batch['data']
        target = batch['label']

        if isinstance(cond, list):
            for cond_i, target_i in zip(cond, target, strict=True):
                if cond_i.ndim == 4:
                    cond_i = cond_i.unsqueeze(0)
                if target_i.ndim == 4:
                    target_i = target_i.unsqueeze(0)
                yield cond_i, target_i
            return

        yield cond, target

    def _batch_size(self, batch) -> int:
        data = batch['data']
        if isinstance(data, list):
            return len(data)
        return int(data.shape[0])

    def _flow_loss_from_pair(self,cond: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
        batch_size = target.shape[0]
        # source = torch.randn_like(target) if I wanted noise
        source = cond

        t = torch.rand(batch_size, device=target.device) * (1.0 - self.eps) + self.eps
        t_view = t.view(batch_size, 1, 1, 1, 1)
        state_t = t_view * target + (1.0 - t_view) * source # X_1(t) + (1-t)X_0 = X_t
        target_velocity = target - source # dX/dt = X_1 - X_0

        pred_velocity = self(state_t, cond, t)
        return self.flow_loss(pred_velocity, target_velocity)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        del batch_idx
        losses = [
            self._flow_loss_from_pair(cond, target)
            for cond, target in self._iter_batch_samples(batch)
        ]
        loss = torch.stack(losses).mean()

        self.log(
            'train_loss',
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self._batch_size(batch),
        )
        return loss

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.randn_like(reference)
        t_schedule = torch.linspace(
            self.eps,
            1.0,
            self.n_sample_steps + 1,
            device=reference.device,
            dtype=reference.dtype,
        )

        model_was_training = self.model.training
        self.model.eval()

        for i in range(self.n_sample_steps):
            t_cur = t_schedule[i]
            t_next = t_schedule[i + 1]
            dt = t_next - t_cur
            t_batch = torch.full(
                (reference.shape[0],),
                t_cur,
                device=reference.device,
                dtype=reference.dtype,
            )
            velocity = self(state, cond, t_batch)
            state = state + dt * velocity

        if model_was_training:
            self.model.train()

        return state

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        del batch_idx
        flow_losses = []
        rollout_nmaes = []

        for cond, target in self._iter_batch_samples(batch):
            flow_losses.append(self._flow_loss_from_pair(cond, target))
            pred = self.sample(cond, target)
            rollout_nmaes.append(self.nmae(pred, target))

        val_loss = torch.stack(flow_losses).mean()
        val_rollout_nmae = torch.stack(rollout_nmaes).mean()
        batch_size = self._batch_size(batch)

        self.log(
            'val_loss',
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'val_rollout_nmae',
            val_rollout_nmae,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return val_loss

    def on_validation_epoch_end(self) -> None:
        metric = self.trainer.callback_metrics.get('val_rollout_nmae')
        if metric is not None:
            self.print(f'val_rollout_nmae: {float(metric):.6f}')

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


LightningFlowTest = LightningFlow


"""
PyTorch Lightning module for Rectified Flow training.

Implements the training procedure from:
  https://github.com/gnobitab/RectifiedFlow

Training:
  1. Sample x_0 ~ N(0, I)  (noise)
  2. Sample t ~ Uniform(eps, 1)
  3. Interpolate: z_t = t * x_1 + (1 - t) * x_0
  4. Target velocity: v = x_1 - x_0
  5. Loss: MSE(model(z_t, t), v)

Inference (Euler):
  Start from x_0 ~ N(0, I), step forward with predicted velocity.
"""

from __future__ import annotations

import copy
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from lightning.pytorch import LightningModule


class LightningFlowMatch(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = instantiate(cfg.model)

        # Rectified flow parameters
        self.eps = 1e-3  # small offset to avoid t=0 singularity
        self.n_sample_steps = getattr(cfg, "n_sample_steps", 10)

        # EMA
        self.ema_rate = getattr(cfg, "ema_rate", 0.9999)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)

    def training_step(self, batch):
        loss = self._flow_loss(batch)
        self.log(
            "train_loss", loss,
            prog_bar=True, on_step=True, on_epoch=True, sync_dist=False,
        )
        return loss

    def validation_step(self, batch):
        loss = self._flow_loss(batch)
        self.log(
            "val_loss", loss,
            prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,
        )
        return loss

    def _flow_loss(self, batch):
        """Rectified flow MSE loss, following the original implementation."""
        cond = batch["data"]   # LR conditioning, shape (B, 1, D, H, W)
        x_1 = batch["label"]   # HR target, shape (B, 1, D, H, W)

        B = x_1.shape[0]
        device = x_1.device

        # 1. Sample noise
        x_0 = torch.randn_like(x_1)

        # 2. Sample t ~ Uniform(eps, 1)
        t = torch.rand(B, device=device) * (1.0 - self.eps) + self.eps

        # 3. Interpolate: z_t = t * x_1 + (1 - t) * x_0
        t_expand = t.view(B, 1, 1, 1, 1)
        z_t = t_expand * x_1 + (1.0 - t_expand) * x_0

        # 4. Velocity target
        target = x_1 - x_0

        # 5. Concatenate [z_t, conditioning] along channel dim
        model_input = torch.cat([z_t, cond], dim=1)

        # 6. Predict velocity
        v_pred = self.model(model_input, t)

        # 7. MSE loss
        losses = (v_pred - target) ** 2
        loss = losses.mean()
        return loss

    @torch.no_grad()
    def sample_euler(self, cond: torch.Tensor, N: int | None = None) -> torch.Tensor:
        """Euler ODE integration from noise to data.

        Parameters
        ----------
        cond : torch.Tensor
            LR conditioning, shape (B, 1, D, H, W).
        N : int, optional
            Number of Euler steps. Defaults to self.n_sample_steps.

        Returns
        -------
        torch.Tensor
            Generated HR sample, shape (B, 1, D, H, W).
        """
        if N is None:
            N = self.n_sample_steps

        model = self.ema_model
        model.eval()
        device = cond.device

        # Start from noise
        x = torch.randn(
            cond.shape[0], 1, *cond.shape[2:],
            device=device, dtype=cond.dtype,
        )

        dt = 1.0 / N
        for i in range(N):
            t_val = i / N * (1.0 - self.eps) + self.eps
            t_batch = torch.ones(cond.shape[0], device=device) * t_val

            model_input = torch.cat([x, cond], dim=1)
            v_pred = model(model_input, t_batch)
            x = x + v_pred * dt

        return x

    def on_before_zero_grad(self, *args, **kwargs):
        """Update EMA after each optimizer step."""
        self._update_ema()

    @torch.no_grad()
    def _update_ema(self):
        for p_ema, p_model in zip(
            self.ema_model.parameters(), self.model.parameters(), strict=True
        ):
            p_ema.data.mul_(self.ema_rate).add_(p_model.data, alpha=1.0 - self.ema_rate)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
            betas=(getattr(self.cfg, "beta1", 0.9), getattr(self.cfg, "beta2", 0.999)),
        )

        linsch = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-5, end_factor=1,
            total_iters=self.cfg.warmup_length,
        )
        cossch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(self.cfg.epochs) - self.cfg.warmup_length,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [linsch, cossch], milestones=[self.cfg.warmup_length],
        )
        return [optimizer], [scheduler]

    # ------------------------------------------------------------------
    # Test / inference hooks (mirrors LightningGenerator)
    # ------------------------------------------------------------------
    def on_test_start(self):
        self.log_dir = self.test_cfg.log_dir
        self.out_dir = self.test_cfg.out_dir
        self.tmp_dir = self.test_cfg.tmp_dir
        self.save_pred = self.test_cfg.save_pred
        self.test_outputs = []

    def test_step(self, batch):
        cond = batch["data"]
        y = batch["label"]
        indices = batch["index"]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        preds = self.sample_euler(cond)
        end.record()

        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)

        # Compute NormMAE for evaluation
        from electrai.model.loss.charge import NormMAE
        loss_fn = NormMAE()
        loss = loss_fn(preds, y)

        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        out = {
            "target": y.detach().cpu(),
            "index": indices,
            "nmae": loss.detach().cpu(),
            "duration": elapsed,
        }
        if self.save_pred:
            out["pred"] = preds.detach().cpu()
        return out

    def on_test_batch_end(self, outputs, _batch, batch_idx):
        indices = outputs["index"]
        nmae = outputs["nmae"]

        if self.save_pred:
            preds = outputs["pred"]
            for i in range(len(indices)):
                idx = indices[i]
                np.save(
                    self.out_dir / f"rank_{self.global_rank}_{idx}.npy",
                    preds[i].squeeze(0).cpu().numpy(),
                )

        if isinstance(nmae, torch.Tensor) and nmae.ndim == 0:
            nmae = nmae.unsqueeze(0)
        tmp_csv = self.tmp_dir / f"metrics_rank_{self.global_rank}_batch_{batch_idx}.csv"
        with tmp_csv.open("w") as f:
            for idx, n in zip(indices, nmae, strict=True):
                f.write(f"rank_{self.global_rank},{idx},{n.item()}\n")

    def on_test_epoch_end(self):
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        local_count = len(list(self.tmp_dir.glob(f"metrics_rank_{rank}_batch_*.csv")))

        if is_dist:
            count_tensor = torch.tensor([local_count], dtype=torch.long, device=self.device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            expected_total = int(count_tensor.item())
            dist.barrier()
        else:
            expected_total = local_count

        final_csv = self.log_dir / "metrics.csv"

        if self.global_rank == 0:
            retries = 0
            all_tmp_csvs = sorted(self.tmp_dir.glob("metrics_rank_*_batch_*.csv"))
            while len(all_tmp_csvs) < expected_total and retries < 60:
                time.sleep(1)
                all_tmp_csvs = sorted(self.tmp_dir.glob("metrics_rank_*_batch_*.csv"))
                retries += 1

            if len(all_tmp_csvs) < expected_total:
                raise RuntimeError(
                    f"Expected {expected_total} CSV files but found {len(all_tmp_csvs)}."
                )

            with final_csv.open("w") as f_out:
                f_out.write("rank,index,nmae\n")
                for tmp_csv in all_tmp_csvs:
                    with tmp_csv.open() as f_in:
                        for line in f_in:
                            f_out.write(line)

            shutil.rmtree(self.tmp_dir, ignore_errors=True)

        if is_dist:
            dist.barrier()

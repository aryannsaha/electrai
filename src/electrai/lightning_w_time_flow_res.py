from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from lightning.pytorch import LightningModule

from electrai.model.loss.charge import NormMAE, SquaredFrobeniusLoss


class LightningGenerator(LightningModule):
    def __init__(self, cfg, test_cfg=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.test_cfg: Any = test_cfg
        self.model = instantiate(cfg.model)
        self.loss_fn = SquaredFrobeniusLoss()
        self.nmae_fn = NormMAE()
        self.n_inference_steps: int = getattr(cfg, "n_inference_steps", 10)
        self.eps: float = getattr(cfg, "eps", 1e-4)
        self.source_distribution: str = getattr(cfg, "source_distribution", "zero").lower()
        self.source_noise_scale: float = float(getattr(cfg, "source_noise_scale", 1.0))
        self.log_dir: Path = Path()
        self.out_dir: Path = Path()
        self.tmp_dir: Path = Path()
        self.save_pred: bool = False

    def forward(self, x, t=None, cond=None): # input, time, and optional condition
        if t is None:
            t = x.new_ones(x.shape[0])
        if cond is not None:
            return self.model(x, t, cond)
        return self.model(x, t)

    def training_step(self, batch):
        random_t_sq_fro, random_t_nmae = self._random_t_metrics(batch)
        batch_size = self._infer_batch_size(batch)

        self._log_epoch_progress(batch_size=batch_size)
        self._log_metric(
            "train_loss",
            random_t_sq_fro,
            batch_size=batch_size,
            prog_bar=True,
            sync_dist=False,
            log_step=True,
        )
        self._log_metric(
            "train_random_t_sq_fro",
            random_t_sq_fro,
            batch_size=batch_size,
            sync_dist=False,
            log_step=True,
        )
        self._log_metric(
            "train_random_t_nmae",
            random_t_nmae,
            batch_size=batch_size,
            sync_dist=False,
            log_step=True,
        )
        return random_t_sq_fro

    def validation_step(self, batch):
        # Training-style validation metrics at a random interpolation time t
        random_t_sq_fro, random_t_nmae = self._random_t_metrics(batch)
        batch_size = self._infer_batch_size(batch)

        self._log_epoch_progress(batch_size=batch_size)
        self._log_metric(
            "val_loss",
            random_t_sq_fro,
            batch_size=batch_size,
            prog_bar=True,
            sync_dist=True,
            log_step=True,
        )
        self._log_metric(
            "val_random_t_sq_fro",
            random_t_sq_fro,
            batch_size=batch_size,
            sync_dist=True,
        )
        self._log_metric(
            "val_random_t_nmae",
            random_t_nmae,
            batch_size=batch_size,
            sync_dist=True,
        )

        # Full rollout from x (low-res) to y_hat, compare against DFT ground truth
        x = batch["data"]
        y = batch["label"]
        with torch.no_grad():
            if isinstance(x, list):
                preds_list = [self._sample(x_i.unsqueeze(0)) for x_i in x]
                rollout_nmae_losses = [
                    self.nmae_fn(p, y_i.unsqueeze(0))
                    for p, y_i in zip(preds_list, y, strict=True)
                ]
                rollout_sq_fro_losses = [
                    self.loss_fn(p, y_i.unsqueeze(0))
                    for p, y_i in zip(preds_list, y, strict=True)
                ]
                rollout_nmae = torch.stack(rollout_nmae_losses).mean()
                rollout_sq_fro = torch.stack(rollout_sq_fro_losses).mean()
            else:
                preds = self._sample(x)
                rollout_nmae = self.nmae_fn(preds, y)
                rollout_sq_fro = self.loss_fn(preds, y)
        self._log_metric(
            "val_rollout_nmae",
            rollout_nmae,
            batch_size=batch_size,
            prog_bar=True,
            sync_dist=True,
        )
        self._log_metric(
            "val_rollout_sq_fro",
            rollout_sq_fro,
            batch_size=batch_size,
            sync_dist=True,
        )
        return random_t_sq_fro

    def _infer_batch_size(self, batch) -> int:
        x = batch["data"]
        if isinstance(x, list):
            return len(x)
        return int(x.shape[0])

    def _log_metric(
        self,
        name: str,
        value: torch.Tensor,
        *,
        batch_size: int,
        prog_bar: bool = False,
        sync_dist: bool,
        log_step: bool = False,
    ) -> None:
        if log_step:
            self.log(
                f"{name}_step",
                value,
                prog_bar=prog_bar,
                on_step=True,
                on_epoch=False,
                sync_dist=sync_dist,
                batch_size=batch_size,
            )

        self.log(
            name,
            value,
            prog_bar=prog_bar,
            on_step=False,
            on_epoch=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )

    def _log_epoch_progress(self, *, batch_size: int) -> None:
        trainer = getattr(self, "_trainer", None)
        current_epoch = float(getattr(trainer, "current_epoch", 0.0))
        self.log(
            "epoch",
            current_epoch,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=batch_size,
        )

    def _loss_calculation(self, batch):
        loss, _ = self._random_t_metrics(batch)
        return loss

    def _random_t_metrics(self, batch):
        x = batch["data"]
        y = batch["label"]
        if isinstance(x, list): #never used, unless batch >1
            metrics = [
                self._flow_metrics(
                    self._sample_source_state(x_i.unsqueeze(0)),
                    (y_i - x_i).unsqueeze(0),
                    cond=x_i.unsqueeze(0),
                )
                for x_i, y_i in zip(x, y, strict=True)
            ]
            sq_fro_losses, nmae_losses = zip(*metrics, strict=True)
            return torch.stack(sq_fro_losses).mean(), torch.stack(nmae_losses).mean()
        return self._flow_metrics(self._sample_source_state(x), y - x, cond=x)

    def _sample_source_state(self, reference: torch.Tensor) -> torch.Tensor:
        if self.source_distribution in {"zero", "zeros", "deterministic_zero"}:
            return torch.zeros_like(reference)
        if self.source_distribution in {
            "gaussian",
            "standard_gaussian",
            "zero_mean_gaussian",
        }:
            noise = torch.randn_like(reference)
            noise = noise - noise.mean(dim=tuple(range(1, noise.ndim)), keepdim=True)
            return self.source_noise_scale * noise
        raise ValueError(
            "Unknown source_distribution="
            f"{self.source_distribution!r}. Expected 'zero' or 'zero_mean_gaussian'."
        )

    def _zero_charge_residual(self, residual: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, residual.ndim))
        return residual - residual.mean(dim=dims, keepdim=True)

    def _flow_loss(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """X-prediction flow loss. Interpolates x_t between x (low-res) and y (high-res),
        then regresses the model output directly against y."""
        loss, _ = self._flow_metrics(x_0, x_1)
        return loss

    def _flow_metrics(
        self, x_0: torch.Tensor, x_1: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = x_0.shape[0] #batch size, shoudl be 1
        # Sample t ~ Uniform[eps, 1-eps] per sample
        t = torch.rand(bsz, device=x_0.device) * (1 - 2 * self.eps) + self.eps #(1,)
        t_e = t.view(bsz, *([1] * (x_0.ndim - 1)))  # broadcast to match input tensor dims (B, 1, 1, 1,..)
        x_t = (1 - t_e) * x_0 + t_e * x_1  # linear interpolation low→high res
        y_hat = self(x_t, t, cond=cond)  # t is 1-dim tensor
        y_hat = self._zero_charge_residual(y_hat)
        return self.loss_fn(y_hat, x_1), self.nmae_fn(y_hat, x_1)

    @torch.no_grad()
    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        """Euler ODE integration from x (low-res) toward predicted high-res output.

        Velocity is implied from x-prediction: v_t = (y_hat - x_t) / (1 - t).
        At the last step this reduces to x_t = y_hat exactly.
        """
        bsz = x.shape[0] #batch size, should be 1
        x_t = self._sample_source_state(x)
        t_steps = torch.linspace(
            0.0, 1.0, self.n_inference_steps + 1, device=x.device, dtype=x.dtype
        )
        for i in range(self.n_inference_steps):
            t_cur = t_steps[i]
            dt = t_steps[i + 1] - t_cur
            t_batch = t_cur.expand(bsz)
            y_hat = self(x_t, t_batch, cond=x)
            y_hat = self._zero_charge_residual(y_hat)
            denom = (1.0 - t_cur).clamp(min=self.eps)
            x_t = x_t + dt * (y_hat - x_t) / denom
        return x + self._zero_charge_residual(x_t)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
            betas=(getattr(self.cfg, "beta1", 0.9), getattr(self.cfg, "beta2", 0.999)),
        )

        linsch = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-5,
            end_factor=1,
            total_iters=self.cfg.warmup_length,
        )
        cossch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(self.cfg.epochs) - self.cfg.warmup_length
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [linsch, cossch], milestones=[self.cfg.warmup_length]
        )
        return [optimizer], [scheduler]

    def on_test_start(self):
        self.log_dir = self.test_cfg.log_dir
        self.out_dir = self.test_cfg.out_dir
        self.tmp_dir = self.test_cfg.tmp_dir
        self.save_pred = self.test_cfg.save_pred

    def test_step(self, batch):
        x = batch["data"]
        y = batch["label"]
        indices = batch["index"]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        if isinstance(x, list):
            preds_list = [self._sample(x_i.unsqueeze(0)) for x_i in x]
            losses = [
                self.loss_fn(p, y_i.unsqueeze(0))
                for p, y_i in zip(preds_list, y, strict=True)
            ]
            preds = torch.cat(preds_list, dim=0)
            loss = torch.stack(losses).mean()
        else:
            preds = self._sample(x)
            loss = self.loss_fn(preds, y)
        end.record()

        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)

        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        out = {
            "target": y.detach().cpu()
            if not isinstance(y, list)
            else [yi.cpu() for yi in y],
            "index": indices,
            "nmae": loss.detach().cpu(),
            "duration": elapsed,
        }
        if self.save_pred:
            out["pred"] = preds.detach().cpu()
        return out

    def on_test_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        _ = batch, dataloader_idx
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
        tmp_csv = (
            self.tmp_dir / f"metrics_rank_{self.global_rank}_batch_{batch_idx}.csv"
        )
        with tmp_csv.open("w") as f:
            for idx, n in zip(indices, nmae, strict=True):
                f.write(f"rank_{self.global_rank},{idx},{n.item()}\n")

    def on_test_epoch_end(self):
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        # Count only files written by THIS rank
        local_count = len(list(self.tmp_dir.glob(f"metrics_rank_{rank}_batch_*.csv")))

        if is_dist:
            count_tensor = torch.tensor(
                [local_count], dtype=torch.long, device=self.device
            )
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

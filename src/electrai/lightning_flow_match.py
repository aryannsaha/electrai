"""
PyTorch Lightning module for Flow Matching training.

=== OVERVIEW ===

This module implements the Rectified Flow / Conditional Flow Matching training
loop as a PyTorch Lightning module. It replaces the direct regression approach
(LR -> HR) with a flow-based generative approach (noise -> HR, conditioned on LR).

=== TRAINING LOOP (what happens each step) ===

1. Get a batch: {LR: (B, 1, Nx_lr, Ny_lr, Nz_lr), HR: (B, 1, Nx_hr, Ny_hr, Nz_hr)}

2. Sample random noise at HR resolution:
   noise ~ N(0, I), shape: (B, 1, Nx_hr, Ny_hr, Nz_hr)

3. Sample random timestep with logit-normal distribution:
   t = sigmoid(N(0, 1)), shape: (B,)  — focuses learning on mid-range timesteps

4. Create the interpolant at HR resolution:
   x_t = (1 - t) * noise + t * HR

5. Downsample x_t to LR resolution and concatenate with LR:
   model_input = [downsample(x_t), LR]: (B, 2, Nx_lr, Ny_lr, Nz_lr)

6. Model predicts velocity at HR resolution (via PixelShuffle):
   v_pred = model(model_input, t): (B, 1, Nx_hr, Ny_hr, Nz_hr)

7. Loss = MSE(v_pred, HR - noise)

=== INFERENCE ===

1. Start from pure noise at HR resolution: x_0 ~ N(0, I)
2. For each Euler step: downsample x to LR, model predicts HR velocity
3. Apply charge conservation normalization to final output

References:
  - Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
  - Liu et al., "Rectified Flow" (ICLR 2023)
  - Esser et al., "Scaling Rectified Flow Transformers" (2024)
"""

from __future__ import annotations

import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from hydra.utils import instantiate
from lightning.pytorch import LightningModule

from electrai.model.loss.charge import NormMAE


class LightningFlowMatch(LightningModule):
    """
    PyTorch Lightning module for flow matching training.

    This replaces LightningGenerator for the flow matching paradigm.
    The key differences are:
      - training_step: implements the flow matching loss (MSE on velocity)
      - sample(): new method for ODE-based inference
      - No NormMAE loss during training (we use MSE on velocities instead)
      - NormMAE is still used for validation metrics to compare with the
        original SRGAN approach on the same scale
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # --- Model instantiation ---
        # The config should point to FlowMatchGeneratorResNet
        # which accepts (B, 2, Nx, Ny, Nz) + timestep t
        self.model = instantiate(cfg.model)

        # --- Loss functions ---
        # MSE for training (velocity prediction loss)
        self.mse_loss = torch.nn.MSELoss()
        # NormMAE for validation metrics only — this lets us compare
        # flow matching results with the original SRGAN on the same metric
        self.nmae_loss = NormMAE()

        # --- Inference parameters ---
        # Number of Euler steps during inference (more steps = better quality, slower)
        # This can be overridden at inference time
        self.n_sample_steps = getattr(cfg, "n_sample_steps", 10)

    def forward(self, x, t):
        """Forward pass — delegates to the model.
        x: (B, 2, Nx, Ny, Nz) concatenated [x_t, lr_up]
        t: (B,) timestep
        """
        return self.model(x, t)

    # =========================================================================
    # TRAINING STEP — The core flow matching algorithm
    # =========================================================================
    def training_step(self, batch):
        """
        One training step of the flow matching algorithm.

        This is where the magic happens. We:
        1. Sample noise and a random timestep
        2. Create the interpolant x_t between noise and HR data
        3. Have the model predict the velocity field
        4. Compute MSE loss between predicted and target velocity
        """
        # --- Unpack batch ---
        lr = batch["data"]  # Low-res input: (B, 1, Nx_lr, Ny_lr, Nz_lr)
        hr = batch["label"]  # High-res target: (B, 1, Nx_hr, Ny_hr, Nz_hr)

        loss = self._flow_match_loss(lr, hr)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        return loss

    # =========================================================================
    # VALIDATION STEP — Computes both flow matching loss and NormMAE
    # =========================================================================
    def validation_step(self, batch):
        """
        Validation step.

        We compute two metrics:
        1. Flow matching loss (MSE on velocity) — for training monitoring
        2. NormMAE on a generated sample — for comparison with SRGAN baseline

        The NormMAE requires running full inference (ODE integration), which is
        expensive, so we only do it during validation (not every training step).
        """
        lr = batch["data"]
        hr = batch["label"]

        # Metric 1: Flow matching velocity loss (same as training)
        fm_loss = self._flow_match_loss(lr, hr)
        self.log(
            "val_fm_loss",
            fm_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # Metric 2: Generate a sample via ODE integration and compute NormMAE
        # This tells us how good the actual generated HR is (not just the velocity)
        with torch.no_grad():
            # Handle both tensor batches and list batches (variable-sized grids)
            if isinstance(hr, list):
                nmaes = []
                for lr_i, hr_i in zip(lr, hr, strict=True):
                    pred_i = self.sample(
                        lr_i.unsqueeze(0), target_shape=hr_i.shape[-3:]
                    )
                    nmaes.append(self.nmae_loss(pred_i, hr_i.unsqueeze(0)))
                nmae = torch.stack(nmaes).mean()
            else:
                hr_pred = self.sample(lr, target_shape=hr.shape[2:])
                nmae = self.nmae_loss(hr_pred, hr)
        self.log(
            "val_loss", nmae, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )

        return nmae

    # =========================================================================
    # _flow_match_loss — The actual flow matching loss computation
    # =========================================================================
    def _flow_match_loss(self, lr, hr):
        """
        Compute the flow matching (rectified flow) loss.

        Parameters
        ----------
        lr : torch.Tensor
            Low-res input, shape (B, 1, Nx_lr, Ny_lr, Nz_lr)
        hr : torch.Tensor
            High-res target, shape (B, 1, Nx_hr, Ny_hr, Nz_hr)

        Returns
        -------
        torch.Tensor
            Scalar MSE loss between predicted and target velocity
        """
        # Handle list inputs (variable-size grids that couldn't be batched)
        if isinstance(lr, list):
            losses = []
            for lr_i, hr_i in zip(lr, hr, strict=True):
                losses.append(
                    self._flow_match_loss_single(lr_i.unsqueeze(0), hr_i.unsqueeze(0))
                )
            return torch.stack(losses).mean()
        return self._flow_match_loss_single(lr, hr)

    def _flow_match_loss_single(self, lr, hr):
        """
        Flow matching loss for a single (possibly batched) tensor pair.

        The model operates at LR resolution with PixelShuffle upscaling to HR.
        Timesteps use logit-normal sampling to focus on mid-range values.
        """
        B = hr.shape[0]

        # --- Sample noise at HR resolution ---
        noise = torch.randn_like(hr)

        # --- Sample timestep with logit-normal distribution ---
        # t = sigmoid(N(0, 1)) concentrates around t=0.5 where learning
        # is hardest, and puts less weight on trivial endpoints.
        # (Esser et al., "Scaling Rectified Flow Transformers", 2024)
        u = torch.randn(B, device=hr.device)
        t = torch.sigmoid(u)
        t_expand = t.view(B, 1, 1, 1, 1)

        # --- Create interpolant at HR resolution ---
        x_t = (1 - t_expand) * noise + t_expand * hr

        # --- Downsample x_t to LR resolution for model input ---
        # The model processes at LR resolution (8x fewer voxels) and
        # upscales to HR via PixelShuffle at the end.
        x_t_lr = F.interpolate(
            x_t, size=lr.shape[2:], mode="trilinear", align_corners=False
        )

        # --- Model input: [x_t_downsampled, LR] at LR resolution ---
        model_input = torch.cat([x_t_lr, lr], dim=1)

        # --- Model predicts velocity at HR resolution (via PixelShuffle) ---
        v_pred = self.model(model_input, t)

        # --- Target velocity at HR resolution ---
        v_target = hr - noise

        return self.mse_loss(v_pred, v_target)

    # =========================================================================
    # SAMPLING — Generate HR from LR via ODE integration
    # =========================================================================
    @torch.no_grad()
    def sample(
        self,
        lr: torch.Tensor,
        target_shape: tuple[int, ...] | None = None,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Generate a high-resolution sample from a low-resolution input
        by integrating the learned velocity field from t=0 (noise) to t=1 (data).

        Uses Euler integration at HR resolution. At each step, the current state
        is downsampled to LR and fed to the model (which upscales back to HR via
        PixelShuffle). Charge conservation is applied to the final output.

        Parameters
        ----------
        lr : torch.Tensor
            Low-res input, shape (B, 1, Nx_lr, Ny_lr, Nz_lr)
        target_shape : tuple of int, optional
            Target spatial dimensions (Nx_hr, Ny_hr, Nz_hr).
            If None, uses 2x the LR spatial dims.
        n_steps : int, optional
            Number of Euler integration steps. More steps = better quality.
            Default: self.n_sample_steps (from config, default 10).

        Returns
        -------
        torch.Tensor
            Generated HR output, shape (B, 1, Nx_hr, Ny_hr, Nz_hr)
        """
        if n_steps is None:
            n_steps = self.n_sample_steps

        if target_shape is None:
            target_shape = tuple(s * 2 for s in lr.shape[2:])

        # Start from pure noise at HR resolution
        x = torch.randn(lr.shape[0], 1, *target_shape, device=lr.device, dtype=lr.dtype)

        # Euler integration from t=0 to t=1
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((lr.shape[0],), i * dt, device=lr.device, dtype=lr.dtype)

            # Downsample current state to LR for model input
            x_lr = F.interpolate(
                x, size=lr.shape[2:], mode="trilinear", align_corners=False
            )

            # Model input at LR: [x_downsampled, lr]
            model_input = torch.cat([x_lr, lr], dim=1)

            # Model predicts velocity at HR (via PixelShuffle)
            v = self.model(model_input, t)

            # Euler step at HR resolution
            x = x + v * dt

        # Post-processing: clamp + charge conservation
        x = torch.clamp(x, min=0.0)

        # Charge conservation: total charge in HR = total charge in LR * volume_ratio
        # Same normalization as GeneratorResNet (srgan_layernorm_pbc.py)
        upscale_factor = 1
        for s_hr, s_lr in zip(target_shape, lr.shape[2:], strict=True):
            upscale_factor *= s_hr / s_lr
        upscale_factor = round(upscale_factor)
        hr_sum = torch.sum(x, dim=(-3, -2, -1), keepdim=True).clamp(min=1e-10)
        lr_sum = torch.sum(lr, dim=(-3, -2, -1), keepdim=True)
        x = x / hr_sum * lr_sum * upscale_factor

        return x

    # =========================================================================
    # OPTIMIZER & LR SCHEDULE — Same as the original LightningGenerator
    # =========================================================================
    def configure_optimizers(self):
        """
        Adam optimizer with linear warmup + cosine annealing schedule.
        Identical to the original LightningGenerator.
        """
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

    # =========================================================================
    # TEST STEP — For final evaluation
    # =========================================================================
    def on_test_start(self):
        self.log_dir = self.test_cfg.log_dir
        self.out_dir = self.test_cfg.out_dir
        self.tmp_dir = self.test_cfg.tmp_dir
        self.save_pred = self.test_cfg.save_pred
        self.test_outputs = []

    def test_step(self, batch):
        """
        Test step: generate HR from LR and compute NormMAE against ground truth.
        """
        lr = batch["data"]
        hr = batch["label"]
        indices = batch["index"]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Generate HR via ODE integration
        start.record()
        if isinstance(hr, list):
            pred_list, nmaes = [], []
            for lr_i, hr_i in zip(lr, hr, strict=True):
                p = self.sample(lr_i.unsqueeze(0), target_shape=hr_i.shape[-3:])
                pred_list.append(p)
                nmaes.append(self.nmae_loss(p, hr_i.unsqueeze(0)))
            preds = pred_list
            loss = torch.stack(nmaes).mean()
        else:
            preds = self.sample(lr, target_shape=hr.shape[2:])
            loss = self.nmae_loss(preds, hr)
        end.record()

        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)

        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        out = {
            "target": hr.detach().cpu(),
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
        tmp_csv = (
            self.tmp_dir / f"metrics_rank_{self.global_rank}_batch_{batch_idx}.csv"
        )
        with tmp_csv.open("w") as f:
            for idx, n in zip(indices, nmae, strict=True):
                f.write(f"rank_{self.global_rank},{idx},{n.item()}\n")

    def on_test_epoch_end(self):
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

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

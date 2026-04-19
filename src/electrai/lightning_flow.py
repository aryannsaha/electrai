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

Inference:
  Start from x_0 ~ N(0, I), step forward with predicted velocity.
"""

from __future__ import annotations

import copy
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from hydra.utils import instantiate
from lightning.pytorch import LightningModule

from electrai.model.loss.charge import ElectronCountLoss, NormMAE


class LightningFlowMatch(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = instantiate(cfg.model)

        self.eps = 1e-3
        self.n_sample_steps = getattr(cfg, 'n_sample_steps', 10)
        self.val_rollout_steps = getattr(cfg, 'val_rollout_steps', self.n_sample_steps)
        self.sample_solver = getattr(cfg, 'sample_solver', 'euler').lower()
        self.val_use_ema = getattr(cfg, 'val_use_ema', True)

        self.flow_loss_weight = float(getattr(cfg, 'flow_loss_weight', 1.0))
        self.endpoint_nmae_weight = float(getattr(cfg, 'endpoint_nmae_weight', 0.0))
        self.endpoint_mass_weight = float(getattr(cfg, 'endpoint_mass_weight', 0.0))

        self.nmae_loss = NormMAE()
        self.mass_loss = ElectronCountLoss()

        self.ema_rate = getattr(cfg, 'ema_rate', 0.9999)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)

        # The baseline flow model still defaults to a standard Gaussian source.
        # The ablation experiments can opt into a condition-informed source by
        # switching these config values in their experiment-specific YAML files.
        self.source_distribution = getattr(
            cfg, 'source_distribution', 'standard_gaussian',
        ).lower()
        self.source_mean_scale = float(getattr(cfg, 'source_mean_scale', 0.0))
        self.source_noise_base_std = float(getattr(cfg, 'source_noise_base_std', 1.0))
        self.source_noise_cond_scale = float(getattr(cfg, 'source_noise_cond_scale', 0.0))
        self.source_noise_power = float(getattr(cfg, 'source_noise_power', 0.5))
        self.source_noise_blur_kernel = int(getattr(cfg, 'source_noise_blur_kernel', 1))
        self.source_noise_eps = float(getattr(cfg, 'source_noise_eps', 1e-8))
        self.density_projection = getattr(
            cfg, 'density_projection', 'none',
        ).lower()
        self.density_floor = float(getattr(cfg, 'density_floor', 0.0))
        self.density_softplus_beta = float(
            getattr(cfg, 'density_softplus_beta', 4.0),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)

    # ---------------------------------------------------------------------
    # Extensibility hooks
    #
    # The three ablations in this repo all share the same optimizer, logging,
    # EMA handling, and test loop. To keep those pieces aligned, the base flow
    # module exposes a few focused hooks that child classes can override.
    # ---------------------------------------------------------------------

    def _condition_for_model(self, cond: torch.Tensor, *, stage: str) -> torch.Tensor:
        """Return the conditioning tensor presented to the model.

        The baseline model always uses the clean SAD guess. The conditioning
        augmentation ablation overrides this hook during training.
        """
        return cond

    def _target_state(self, cond: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Return the state variable whose flow we model.

        The default formulation models the full density directly. The residual
        ablation overrides this to model `(target - condition)` instead.
        """
        return x_1

    def _prediction_from_state(
        self, state: torch.Tensor, cond: torch.Tensor,
    ) -> torch.Tensor:
        """Map the flow state back to a physical density prediction."""
        del cond
        if self.density_projection in {'identity', 'none'}:
            return state
        if self.density_projection == 'clamp':
            return state.clamp_min(self.density_floor)
        if self.density_projection == 'softplus':
            shifted = state - self.density_floor
            return (
                F.softplus(shifted, beta=self.density_softplus_beta)
                + self.density_floor
            )

        raise ValueError(
            f'Unknown density_projection={self.density_projection!r}. '
            "Expected 'none', 'clamp', or 'softplus'.",
        )

    def _reference_state_from_condition(self, cond: torch.Tensor) -> torch.Tensor:
        """Return a tensor with the right shape for source-state sampling.

        In the default formulation the source state lives in the same space as
        the target density, which already matches the conditioning tensor shape
        for the current QM9 flow setups.
        """
        return cond

    def _default_sampling_model(self, *, stage: str):
        """Choose the rollout network for inference-like stages.

        Validation already has an explicit `val_use_ema` switch. Test-time and
        ad hoc sampling should default to the same choice unless the config
        overrides them separately.
        """
        stage = stage.lower()
        if stage == 'validation_rollout':
            use_ema = self.val_use_ema
        elif stage == 'test':
            use_ema = bool(getattr(self.cfg, 'test_use_ema', self.val_use_ema))
        else:
            use_ema = bool(getattr(self.cfg, 'sample_use_ema', self.val_use_ema))
        return self.ema_model if use_ema else self.model

    # ---------------------------------------------------------------------
    # Source distribution helpers
    # ---------------------------------------------------------------------

    def _normalized_guidance_map(
        self,
        cond: torch.Tensor,
        *,
        blur_kernel: int,
        eps: float,
    ) -> torch.Tensor:
        """Build a smooth per-voxel guidance map from the conditioning density.

        This map is intentionally normalized per-sample so the scale controls in
        the config remain interpretable across molecules with different total
        charge magnitudes.
        """
        guide = cond.clamp_min(0.0)

        kernel = max(1, int(blur_kernel))
        if kernel % 2 == 0:
            kernel += 1
        if kernel > 1:
            guide = torch.nn.functional.avg_pool3d(
                guide,
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            )

        flat = guide.flatten(start_dim=1)
        scale = flat.amax(dim=1, keepdim=True).clamp_min(eps)
        return guide / scale.view(-1, 1, 1, 1, 1)

    def _density_guided_std_map(
        self,
        cond: torch.Tensor,
        *,
        base_std: float,
        guided_scale: float,
        power: float,
        blur_kernel: int,
        eps: float,
    ) -> torch.Tensor:
        """Convert a conditioning density into a spatially varying std map."""
        guidance = self._normalized_guidance_map(
            cond,
            blur_kernel=blur_kernel,
            eps=eps,
        )
        std_map = base_std + guided_scale * guidance.pow(power)
        return std_map.to(device=cond.device, dtype=cond.dtype)

    def _sample_source_state(
        self,
        cond: torch.Tensor,
        *,
        reference_state: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the source state used by flow matching.

        By default this is the usual standard Gaussian source. The change-1
        ablation switches `source_distribution` to `conditioned_gaussian`, which
        recent conditional diffusion/flow papers motivate as a better aligned
        source when the condition already contains meaningful low-frequency
        structure.
        """
        if self.source_distribution in {'standard', 'gaussian', 'standard_gaussian'}:
            return torch.randn_like(reference_state)

        if self.source_distribution in {
            'condition',
            'condition_only',
            'deterministic_condition',
            'sad',
            'sad_guess',
        }:
            if cond.shape != reference_state.shape:
                raise ValueError(
                    'Condition-only source sampling requires the condition and '
                    f'state to share a shape, got {tuple(cond.shape)} and '
                    f'{tuple(reference_state.shape)}.',
                )
            return self.source_mean_scale * cond

        if self.source_distribution in {'conditioned', 'conditioned_gaussian'}:
            if cond.shape != reference_state.shape:
                raise ValueError(
                    'Condition-informed source sampling requires the condition and '
                    f'state to share a shape, got {tuple(cond.shape)} and '
                    f'{tuple(reference_state.shape)}.',
                )
            std_map = self._density_guided_std_map(
                cond,
                base_std=self.source_noise_base_std,
                guided_scale=self.source_noise_cond_scale,
                power=self.source_noise_power,
                blur_kernel=self.source_noise_blur_kernel,
                eps=self.source_noise_eps,
            )
            noise = torch.randn_like(reference_state)
            return self.source_mean_scale * cond + std_map * noise

        raise ValueError(
            f'Unknown source_distribution={self.source_distribution!r}. '
            "Expected 'standard_gaussian', 'condition_only', or "
            "'conditioned_gaussian'.",
        )

    def _infer_batch_size(self, batch) -> int:
        data = batch['data']
        if isinstance(data, list):
            return len(data)
        return int(data.shape[0])

    def _iter_batch_samples(self, batch):
        cond = batch['data']
        x_1 = batch['label']
        indices = batch.get('index')

        if isinstance(cond, list):
            if indices is None:
                indices = [None] * len(cond)
            for cond_i, target_i, index_i in zip(cond, x_1, indices, strict=True):
                if cond_i.ndim == 4:
                    cond_i = cond_i.unsqueeze(0)
                if target_i.ndim == 4:
                    target_i = target_i.unsqueeze(0)
                yield cond_i, target_i, index_i
            return

        yield cond, x_1, indices

    def _mean_metrics(self, metrics_list):
        return {
            key: torch.stack([metrics[key] for metrics in metrics_list]).mean()
            for key in metrics_list[0]
        }

    def _log_extra_metrics(
        self,
        prefix: str,
        metrics: dict[str, torch.Tensor],
        *,
        skip: set[str],
        batch_size: int,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        """Log any subclass-provided metrics without hard-coding their names."""
        for name, value in metrics.items():
            if name in skip:
                continue
            self.log(
                f'{prefix}_{name}',
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                batch_size=batch_size,
            )

    def _objective_terms(
        self,
        cond: torch.Tensor,
        x_1: torch.Tensor,
        *,
        stage: str,
    ):
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
        x_1_hat = self._prediction_from_state(state_hat, cond)
        endpoint_nmae = self.nmae_loss(x_1_hat, x_1)
        endpoint_mass = self.mass_loss(x_1_hat, x_1)

        loss = (
            self.flow_loss_weight * flow_mse
            + self.endpoint_nmae_weight * endpoint_nmae
            + self.endpoint_mass_weight * endpoint_mass
        )
        return {
            'loss': loss,
            'flow_mse': flow_mse,
            'endpoint_nmae': endpoint_nmae,
            'endpoint_mass': endpoint_mass,
        }

    def _objective_from_batch(self, batch, *, stage: str):
        metrics = [
            self._objective_terms(cond, target, stage=stage)
            for cond, target, _index in self._iter_batch_samples(batch)
        ]
        return self._mean_metrics(metrics)

    def training_step(self, batch):
        metrics = self._objective_from_batch(batch, stage='train')
        batch_size = self._infer_batch_size(batch)
        self.log(
            'train_loss', metrics['loss'],
            prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_flow_mse', metrics['flow_mse'],
            on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size,
        )
        self.log(
            'train_endpoint_nmae', metrics['endpoint_nmae'],
            on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size,
        )
        self.log(
            'train_endpoint_mass', metrics['endpoint_mass'],
            on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size,
        )
        self._log_extra_metrics(
            'train',
            metrics,
            skip={'loss', 'flow_mse', 'endpoint_nmae', 'endpoint_mass'},
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
        )
        return metrics['loss']

    def validation_step(self, batch):
        objective_metrics = self._objective_from_batch(batch, stage='validation')
        rollout_metrics = self._rollout_metrics_from_batch(
            batch,
            model=self.ema_model if self.val_use_ema else self.model,
            n_steps=self.val_rollout_steps,
            solver=self.sample_solver,
            stage='validation_rollout',
        )
        batch_size = self._infer_batch_size(batch)

        self.log(
            'val_loss', objective_metrics['loss'],
            prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'val_flow_mse', objective_metrics['flow_mse'],
            on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size,
        )
        self.log(
            'val_endpoint_nmae', objective_metrics['endpoint_nmae'],
            on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size,
        )
        self.log(
            'val_endpoint_mass', objective_metrics['endpoint_mass'],
            on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size,
        )
        self.log(
            'val_rollout_nmae', rollout_metrics['rollout_nmae'],
            prog_bar=True, on_step=False, on_epoch=True, sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'val_rollout_mass', rollout_metrics['rollout_mass'],
            on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size,
        )
        self._log_extra_metrics(
            'val',
            objective_metrics,
            skip={'loss', 'flow_mse', 'endpoint_nmae', 'endpoint_mass'},
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        return objective_metrics['loss']

    def _predict_velocity(self, model, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        model_input = torch.cat([x, cond], dim=1)
        return model(model_input, t)

    @torch.no_grad()
    def _sample_state_impl(
        self,
        cond: torch.Tensor,
        *,
        model,
        n_steps: int,
        solver: str,
        stage: str,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        solver = solver.lower()
        if solver not in {'euler', 'heun'}:
            raise ValueError(f'Unknown solver: {solver}')

        model_was_training = model.training
        model.eval()

        model_cond = self._condition_for_model(cond, stage=stage)

        if initial_state is None:
            x = self._sample_source_state(
                cond,
                reference_state=self._reference_state_from_condition(cond),
            )
        else:
            x = initial_state.to(device=cond.device, dtype=cond.dtype)
        t_schedule = torch.linspace(
            self.eps, 1.0, n_steps + 1, device=cond.device, dtype=cond.dtype,
        )

        for i in range(n_steps):
            t_cur = t_schedule[i]
            t_next = t_schedule[i + 1]
            dt = t_next - t_cur
            t_batch = torch.full(
                (cond.shape[0],), t_cur, device=cond.device, dtype=cond.dtype,
            )
            v_cur = self._predict_velocity(model, x, model_cond, t_batch)

            if solver == 'euler':
                x = x + v_cur * dt
                continue

            x_euler = x + v_cur * dt
            t_next_batch = torch.full(
                (cond.shape[0],), t_next, device=cond.device, dtype=cond.dtype,
            )
            v_next = self._predict_velocity(model, x_euler, model_cond, t_next_batch)
            x = x + 0.5 * dt * (v_cur + v_next)

        if model_was_training:
            model.train()
        return x

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        n_steps: int | None = None,
        *,
        model=None,
        solver: str | None = None,
        stage: str = 'sample',
    ) -> torch.Tensor:
        if n_steps is None:
            n_steps = self.n_sample_steps
        if model is None:
            model = self._default_sampling_model(stage=stage)
        if solver is None:
            solver = self.sample_solver
        state = self._sample_state_impl(
            cond,
            model=model,
            n_steps=n_steps,
            solver=solver,
            stage=stage,
        )
        return self._prediction_from_state(state, cond)

    @torch.no_grad()
    def sample_state_from_source(
        self,
        cond: torch.Tensor,
        source_state: torch.Tensor,
        n_steps: int | None = None,
        *,
        model=None,
        solver: str | None = None,
        stage: str = 'sample',
    ) -> torch.Tensor:
        if n_steps is None:
            n_steps = self.n_sample_steps
        if model is None:
            model = self._default_sampling_model(stage=stage)
        if solver is None:
            solver = self.sample_solver
        return self._sample_state_impl(
            cond,
            model=model,
            n_steps=n_steps,
            solver=solver,
            stage=stage,
            initial_state=source_state,
        )

    @torch.no_grad()
    def sample_from_source(
        self,
        cond: torch.Tensor,
        source_state: torch.Tensor,
        n_steps: int | None = None,
        *,
        model=None,
        solver: str | None = None,
        stage: str = 'sample',
    ) -> torch.Tensor:
        state = self.sample_state_from_source(
            cond,
            source_state,
            n_steps=n_steps,
            model=model,
            solver=solver,
            stage=stage,
        )
        return self._prediction_from_state(state, cond)

    @torch.no_grad()
    def sample_euler(self, cond: torch.Tensor, N: int | None = None) -> torch.Tensor:
        return self.sample(
            cond, n_steps=N, model=self.ema_model, solver='euler', stage='sample',
        )

    @torch.no_grad()
    def _rollout_metrics(
        self,
        cond: torch.Tensor,
        x_1: torch.Tensor,
        *,
        model,
        n_steps: int,
        solver: str,
        stage: str,
    ):
        preds = self.sample(
            cond, n_steps=n_steps, model=model, solver=solver, stage=stage,
        )
        return {
            'rollout_nmae': self.nmae_loss(preds, x_1),
            'rollout_mass': self.mass_loss(preds, x_1),
        }

    @torch.no_grad()
    def _rollout_metrics_from_batch(
        self,
        batch,
        *,
        model,
        n_steps: int,
        solver: str,
        stage: str,
    ):
        metrics = [
            self._rollout_metrics(
                cond,
                target,
                model=model,
                n_steps=n_steps,
                solver=solver,
                stage=stage,
            )
            for cond, target, _index in self._iter_batch_samples(batch)
        ]
        return self._mean_metrics(metrics)

    def on_before_zero_grad(self, *args, **kwargs):
        self._update_ema()

    @torch.no_grad()
    def _update_ema(self):
        for p_ema, p_model in zip(
            self.ema_model.parameters(), self.model.parameters(), strict=True,
        ):
            p_ema.data.mul_(self.ema_rate).add_(p_model.data, alpha=1.0 - self.ema_rate)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
            betas=(getattr(self.cfg, 'beta1', 0.9), getattr(self.cfg, 'beta2', 0.999)),
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

    def on_test_start(self):
        self.log_dir = self.test_cfg.log_dir
        self.out_dir = self.test_cfg.out_dir
        self.tmp_dir = self.test_cfg.tmp_dir
        self.save_pred = self.test_cfg.save_pred
        self.test_outputs = []

    def _timed_sample(self, cond: torch.Tensor):
        if torch.cuda.is_available() and cond.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            preds = self.sample(cond, stage='test')
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
        else:
            start_time = time.perf_counter()
            preds = self.sample(cond, stage='test')
            elapsed = (time.perf_counter() - start_time) * 1000.0
        return preds, elapsed

    def test_step(self, batch):
        cond = batch['data']
        y = batch['label']
        indices = batch['index']

        if isinstance(cond, list):
            preds_cpu = []
            targets_cpu = []
            index_list = []
            nmae_values = []
            mass_values = []
            total_elapsed = 0.0

            for cond_i, y_i, idx_i in self._iter_batch_samples(batch):
                preds_i, elapsed_i = self._timed_sample(cond_i)
                total_elapsed += elapsed_i
                preds_cpu.append(preds_i.detach().cpu())
                targets_cpu.append(y_i.detach().cpu())
                nmae_values.append(self.nmae_loss(preds_i, y_i).detach().cpu())
                mass_values.append(self.mass_loss(preds_i, y_i).detach().cpu())
                if isinstance(idx_i, torch.Tensor):
                    index_list.append(int(idx_i.item()))
                else:
                    index_list.append(idx_i)

            nmae = torch.stack(nmae_values)
            mass_error = torch.stack(mass_values)
            self.log('test_loss', nmae.mean(), prog_bar=True, sync_dist=True)
            self.log('test_mass_error', mass_error.mean(), sync_dist=True)

            out = {
                'target': targets_cpu,
                'index': index_list,
                'nmae': nmae,
                'mass_error': mass_error,
                'duration': total_elapsed,
            }
            if self.save_pred:
                out['pred'] = preds_cpu
            return out

        preds, elapsed = self._timed_sample(cond)
        nmae = self.nmae_loss(preds, y)
        mass_error = self.mass_loss(preds, y)

        self.log('test_loss', nmae, prog_bar=True, sync_dist=True)
        self.log('test_mass_error', mass_error, sync_dist=True)

        out = {
            'target': y.detach().cpu(),
            'index': indices,
            'nmae': nmae.detach().cpu(),
            'mass_error': mass_error.detach().cpu(),
            'duration': elapsed,
        }
        if self.save_pred:
            out['pred'] = preds.detach().cpu()
        return out

    def on_test_batch_end(self, outputs, _batch, batch_idx):
        indices = outputs['index']
        nmae = outputs['nmae']
        mass_error = outputs['mass_error']

        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        elif not isinstance(indices, list):
            indices = [indices]

        if self.save_pred:
            preds = outputs['pred']
            pred_items = preds if isinstance(preds, list) else [preds[i] for i in range(len(indices))]
            for idx, pred in zip(indices, pred_items, strict=True):
                pred_to_save = pred
                if pred_to_save.ndim == 5 and pred_to_save.shape[0] == 1:
                    pred_to_save = pred_to_save.squeeze(0)
                if pred_to_save.ndim == 4 and pred_to_save.shape[0] == 1:
                    pred_to_save = pred_to_save.squeeze(0)
                np.save(
                    self.out_dir / f'rank_{self.global_rank}_{idx}.npy',
                    pred_to_save.cpu().numpy(),
                )

        if isinstance(nmae, torch.Tensor) and nmae.ndim == 0:
            nmae = nmae.unsqueeze(0)
        if isinstance(mass_error, torch.Tensor) and mass_error.ndim == 0:
            mass_error = mass_error.unsqueeze(0)
        tmp_csv = self.tmp_dir / f'metrics_rank_{self.global_rank}_batch_{batch_idx}.csv'
        with tmp_csv.open('w') as f:
            for idx, n, m in zip(indices, nmae, mass_error, strict=True):
                f.write(f'rank_{self.global_rank},{idx},{n.item()},{m.item()}\n')

    def on_test_epoch_end(self):
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        local_count = len(list(self.tmp_dir.glob(f'metrics_rank_{rank}_batch_*.csv')))

        if is_dist:
            count_tensor = torch.tensor([local_count], dtype=torch.long, device=self.device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            expected_total = int(count_tensor.item())
            dist.barrier()
        else:
            expected_total = local_count

        final_csv = self.log_dir / 'metrics.csv'

        if self.global_rank == 0:
            retries = 0
            all_tmp_csvs = sorted(self.tmp_dir.glob('metrics_rank_*_batch_*.csv'))
            while len(all_tmp_csvs) < expected_total and retries < 60:
                time.sleep(1)
                all_tmp_csvs = sorted(self.tmp_dir.glob('metrics_rank_*_batch_*.csv'))
                retries += 1

            if len(all_tmp_csvs) < expected_total:
                raise RuntimeError(
                    f'Expected {expected_total} CSV files but found {len(all_tmp_csvs)}.',
                )

            with final_csv.open('w') as f_out:
                f_out.write('rank,index,nmae,mass_error\n')
                for tmp_csv in all_tmp_csvs:
                    with tmp_csv.open() as f_in:
                        for line in f_in:
                            f_out.write(line)

            shutil.rmtree(self.tmp_dir, ignore_errors=True)

        if is_dist:
            dist.barrier()

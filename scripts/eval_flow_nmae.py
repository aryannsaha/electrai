"""Evaluate a flow matching checkpoint with NMAE on the validation set.

Uses the exact same data split as training (same random_seed, max_samples,
val_frac) so the validation samples match what the model saw during training.

Usage:
    uv run python scripts/eval_flow_nmae.py \
        --config examples/QM9/experiment_2/config_flow_match.yaml \
        --ckpt examples/QM9/experiment_2/checkpoints_flow_match/last.ckpt \
        [--n_steps 10] \
        [--solver heun]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import wandb
import yaml
from hydra.utils import instantiate

from electrai.lightning_flow import LightningFlowMatch
from electrai.model.loss.charge import NormMAE


def main():
    parser = argparse.ArgumentParser(description="Evaluate flow matching model with NMAE")
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML config")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.ckpt)")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=None,
        help="Integration steps (default: from config)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=("euler", "heun"),
        default=None,
        help="Sampler override (default: from config)",
    )
    args = parser.parse_args()

    # Load config
    with Path(args.config).open() as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)

    # Override n_sample_steps if provided
    if args.n_steps is not None:
        cfg.n_sample_steps = args.n_steps
    if args.solver is not None:
        cfg.sample_solver = args.solver

    # Setup datamodule with stage="fit" to get the validation set
    datamodule = instantiate(cfg.data)
    datamodule.setup(stage="fit")
    val_loader = datamodule.val_dataloader()
    print(f"Validation set size: {len(datamodule.val_set)}")

    # Load model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = LightningFlowMatch.load_from_checkpoint(args.ckpt, cfg=cfg)
    lit_model = lit_model.to(device)
    lit_model.eval()

    loss_fn = NormMAE()
    n_steps = lit_model.n_sample_steps
    solver = lit_model.sample_solver

    # Initialize W&B
    wandb.init(
        project=getattr(cfg, "wb_pname", "electrai"),
        entity=getattr(cfg, "entity", None),
        name=f"eval_nmae_{getattr(cfg, 'wandb_run_id', 'flow')}",
        config={
            "checkpoint": args.ckpt,
            "n_sample_steps": n_steps,
            "solver": solver,
            "val_size": len(datamodule.val_set),
            "config_file": args.config,
        },
        tags=["eval", "nmae", "flow_match"],
    )

    print(f"Checkpoint: {args.ckpt}")
    print(f"Solver: {solver}")
    print(f"Integration steps: {n_steps}")
    print(f"Device: {device}")
    print("-" * 60)

    all_nmae = []
    all_indices = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            cond = batch["data"].to(device)
            target = batch["label"].to(device)
            indices = batch["index"]

            preds = lit_model.sample(cond, n_steps=n_steps, solver=solver)
            nmae = loss_fn(preds, target)

            all_nmae.append(nmae.item())
            all_indices.append(indices)

            wandb.log({"batch_nmae": nmae.item(), "batch_idx": i})
            print(f"  Batch {i:4d} | NMAE: {nmae.item():.6f} | indices: {indices}")

    mean_nmae = sum(all_nmae) / len(all_nmae)
    wandb.log({"mean_nmae": mean_nmae})
    wandb.finish()

    print("-" * 60)
    print(f"Mean NMAE over {len(all_nmae)} batches: {mean_nmae:.6f}")


if __name__ == "__main__":
    main()

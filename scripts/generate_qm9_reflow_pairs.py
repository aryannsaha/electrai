"""Generate one-round reflow pairs for conditional QM9 flow matching."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from electrai.dataloader.collate import collate_fn
from electrai.lightning_flow import LightningFlowMatch
from electrai.lightning_flow_cond_aug import LightningFlowMatchCondAug
from electrai.lightning_flow_pretrained_cond import LightningFlowMatchPretrainedCond
from electrai.lightning_flow_residual import LightningFlowMatchResidual

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate fixed (x0, z1) reflow pairs from a trained flow model.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to the Stage-1 training YAML config.',
    )
    parser.add_argument(
        '--ckpt',
        type=Path,
        required=True,
        help='Path to the Stage-1 checkpoint (.ckpt).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory where the reflow pair dataset will be written.',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'validation'],
        help='Dataset splits to export. Defaults to train validation.',
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=None,
        help='Sampling step override. Defaults to the training config value.',
    )
    parser.add_argument(
        '--solver',
        type=str,
        choices=('euler', 'heun'),
        default=None,
        help='Sampler override. Defaults to the training config value.',
    )
    parser.add_argument(
        '--sample-model',
        type=str,
        choices=('ema', 'raw'),
        default='ema',
        help='Use the EMA network or the raw network when generating z1.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base seed for deterministic source sampling.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use for generation: auto, cpu, cuda, cuda:0, ...',
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    requested = str(device).strip().lower()
    if requested in {'', 'auto'}:
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    return torch.device(device)


def load_cfg(config_path: Path) -> SimpleNamespace:
    with config_path.open() as fp:
        cfg_dict = yaml.safe_load(fp)
    return SimpleNamespace(**cfg_dict)


def cfg_value(node, key: str, default):
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def load_module(cfg: SimpleNamespace, ckpt_path: Path, device: torch.device):
    training_mode = str(getattr(cfg, 'training_mode', 'flow_match')).lower()
    if training_mode == 'flow_match':
        model_cls = LightningFlowMatch
    elif training_mode == 'flow_match_cond_aug':
        model_cls = LightningFlowMatchCondAug
    elif training_mode == 'flow_match_residual':
        model_cls = LightningFlowMatchResidual
    elif training_mode == 'flow_match_pretrained_cond':
        model_cls = LightningFlowMatchPretrainedCond
    else:
        raise ValueError(
            'Reflow-pair generation currently supports only flow checkpoints, '
            f'got training_mode={training_mode!r}.',
        )

    module = model_cls.load_from_checkpoint(ckpt_path, map_location="cpu", cfg=cfg)
    module.requires_grad_(False)
    return module.to(device).eval()


def scalar_index(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    return value


def save_sample(
    output_dir: Path,
    split: str,
    sample_number: int,
    *,
    cond: torch.Tensor,
    label: torch.Tensor,
    source: torch.Tensor,
    reflow_target: torch.Tensor,
    index,
    noise_seed: int,
) -> dict[str, str | int]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    sample_path = split_dir / f'sample_{sample_number:06d}.pt'
    torch.save(
        {
            'data': cond.squeeze(0).cpu(),
            'label': label.squeeze(0).cpu(),
            'source': source.squeeze(0).cpu(),
            'reflow_target': reflow_target.squeeze(0).cpu(),
            'index': scalar_index(index),
            'noise_seed': int(noise_seed),
        },
        sample_path,
    )
    return {
        'path': str(sample_path.relative_to(output_dir)),
        'index': scalar_index(index),
        'noise_seed': int(noise_seed),
    }


def iter_subset_batches(dataset, *, num_workers: int, pin_memory: bool):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    yield from loader


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    args = parse_args()

    cfg = load_cfg(args.config)
    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = instantiate(cfg.data)
    datamodule.setup(stage=None)
    if not hasattr(datamodule, 'subsets'):
        raise ValueError('Expected datamodule.setup(stage=None) to populate datamodule.subsets.')

    module = load_module(cfg, args.ckpt, device)
    n_steps = module.n_sample_steps if args.n_steps is None else int(args.n_steps)
    solver = module.sample_solver if args.solver is None else str(args.solver).lower()
    sample_network = module.ema_model if args.sample_model == 'ema' else module.model

    manifest = {
        'metadata': {
            'created_at_utc': datetime.now(UTC).isoformat(),
            'source_config': str(args.config),
            'source_checkpoint': str(args.ckpt),
            'source_training_mode': str(getattr(cfg, 'training_mode', 'flow_match')),
            'n_steps': n_steps,
            'solver': solver,
            'sample_model': args.sample_model,
            'seed': int(args.seed),
        },
        'splits': {
            'train': [],
            'validation': [],
            'test': [],
        },
    }

    split_offsets = {'train': 0, 'validation': 1_000_000, 'test': 2_000_000}
    pin_memory = bool(cfg_value(cfg.data, 'pin_memory', False))
    num_workers = int(cfg_value(cfg.data, 'val_workers', 1))

    for split in args.splits:
        if split not in datamodule.subsets:
            raise KeyError(
                f'Split {split!r} is not available in the source datamodule. '
                f'Available splits: {sorted(datamodule.subsets)}.',
            )

        subset = datamodule.subsets[split]
        LOGGER.info('Generating reflow pairs for split=%s (%d samples)', split, len(subset))

        for sample_number, batch in enumerate(
            iter_subset_batches(subset, num_workers=num_workers, pin_memory=pin_memory),
        ):
            cond = batch['data'].to(device)
            label = batch['label'].to(device)
            index = batch.get('index')
            if isinstance(index, list):
                index = index[0]
            elif isinstance(index, torch.Tensor) and index.ndim > 0:
                index = index[0]

            noise_seed = int(args.seed) + split_offsets.get(split, 0) + sample_number
            with torch.inference_mode():
                fork_devices = list(range(torch.cuda.device_count())) if device.type == 'cuda' else []
                with torch.random.fork_rng(devices=fork_devices):
                    torch.manual_seed(noise_seed)
                    if device.type == 'cuda':
                        torch.cuda.manual_seed_all(noise_seed)
                    source = module._sample_source_state(
                        cond,
                        reference_state=module._reference_state_from_condition(cond),
                    )

                reflow_target = module.sample_from_source(
                    cond,
                    source,
                    n_steps=n_steps,
                    model=sample_network,
                    solver=solver,
                    stage='reflow_generation',
                )
            entry = save_sample(
                output_dir,
                split,
                sample_number,
                cond=cond,
                label=label,
                source=source,
                reflow_target=reflow_target,
                index=index,
                noise_seed=noise_seed,
            )
            manifest['splits'][split].append(entry)

    manifest_path = output_dir / 'manifest.json'
    with manifest_path.open('w') as fp:
        json.dump(manifest, fp, indent=2)
    LOGGER.info('Wrote reflow manifest to %s', manifest_path)


if __name__ == '__main__':
    main()

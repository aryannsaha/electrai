from __future__ import annotations

import json

import torch

from electrai.dataloader.reflow_pairs import ReflowPairDataModule, ReflowPairDataset


def write_sample(path, value: float, index: int):
    torch.save(
        {
            'data': torch.full((1, 3, 3, 3), value),
            'label': torch.full((1, 3, 3, 3), value + 1.0),
            'source': torch.full((1, 3, 3, 3), value - 1.0),
            'reflow_target': torch.full((1, 3, 3, 3), value + 2.0),
            'index': index,
        },
        path,
    )


def test_reflow_pair_dataset_loads_manifest_entries(tmp_path):
    train_path = tmp_path / 'train_sample.pt'
    write_sample(train_path, 1.0, 7)
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(
        json.dumps(
            {
                'splits': {
                    'train': [{'path': 'train_sample.pt'}],
                    'validation': [],
                    'test': [],
                },
            },
        ),
    )

    dataset = ReflowPairDataset(manifest_path, 'train')
    sample = dataset[0]

    assert sample['data'].shape == (1, 3, 3, 3)
    assert sample['index'] == 7


def test_reflow_pair_datamodule_builds_train_and_val_loaders(tmp_path):
    write_sample(tmp_path / 'train_sample.pt', 1.0, 1)
    write_sample(tmp_path / 'val_sample.pt', 2.0, 2)
    write_sample(tmp_path / 'test_sample.pt', 3.0, 3)
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(
        json.dumps(
            {
                'splits': {
                    'train': [{'path': 'train_sample.pt'}],
                    'validation': [{'path': 'val_sample.pt'}],
                    'test': [{'path': 'test_sample.pt'}],
                },
            },
        ),
    )

    datamodule = ReflowPairDataModule(manifest_path, batch_size=1)
    datamodule.setup(stage='fit')

    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))

    assert train_batch['source'].shape == (1, 1, 3, 3, 3)
    assert val_batch['reflow_target'].shape == (1, 1, 3, 3, 3)

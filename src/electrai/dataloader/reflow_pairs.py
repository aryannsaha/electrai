from __future__ import annotations

import json
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from electrai.dataloader.collate import collate_fn


class ReflowPairDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str):
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open() as fp:
            manifest = json.load(fp)

        splits = manifest.get('splits')
        if not isinstance(splits, dict):
            raise ValueError(
                f'Manifest at {self.manifest_path} must define a top-level "splits" mapping.',
            )
        if split not in splits:
            raise KeyError(
                f'Split {split!r} not found in {self.manifest_path}. '
                f'Available splits: {sorted(splits)}.',
            )

        self.root = self.manifest_path.parent
        self.entries = list(splits[split])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        if isinstance(entry, dict):
            path_value = entry.get('path')
        else:
            path_value = entry
        if path_value is None:
            raise ValueError(
                f'Manifest entry #{index} in {self.manifest_path} is missing a path.',
            )

        sample_path = self.root / Path(path_value)
        sample = torch.load(sample_path, map_location='cpu', weights_only=False)
        if not isinstance(sample, dict):
            raise TypeError(
                f'Reflow sample at {sample_path} must be a dict, got {type(sample)!r}.',
            )
        return sample


class ReflowPairDataModule(LightningDataModule):
    def __init__(
        self,
        manifest_path: str | Path,
        batch_size: int = 1,
        train_workers: int = 2,
        val_workers: int = 1,
        pin_memory: bool = False,
        drop_last: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.manifest_path = Path(manifest_path)
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def setup(self, stage=None):
        if stage in {None, 'fit'}:
            self.train_set = ReflowPairDataset(self.manifest_path, 'train')
            self.val_set = ReflowPairDataset(self.manifest_path, 'validation')
        if stage in {None, 'test'}:
            self.test_set = ReflowPairDataset(self.manifest_path, 'test')

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            self.batch_size,
            num_workers=self.train_workers,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            self.batch_size,
            num_workers=self.val_workers,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=1,
            num_workers=self.val_workers,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

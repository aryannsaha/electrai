from __future__ import annotations

import torch

from electrai.dataloader.collate import collate_fn


def test_collate_fn_uses_default_collate_for_uniform_dict_batches():
    batch = [
        {
            'data': torch.ones(1, 2, 2, 2),
            'label': torch.zeros(1, 2, 2, 2),
            'index': 0,
        },
        {
            'data': torch.zeros(1, 2, 2, 2),
            'label': torch.ones(1, 2, 2, 2),
            'index': 1,
        },
    ]

    collated = collate_fn(batch)

    assert isinstance(collated['data'], torch.Tensor)
    assert collated['data'].shape == (2, 1, 2, 2, 2)
    assert collated['index'].tolist() == [0, 1]


def test_collate_fn_falls_back_to_lists_for_variable_sized_dict_batches():
    batch = [
        {
            'data': torch.ones(1, 2, 2, 2),
            'label': torch.zeros(1, 2, 2, 2),
            'index': 0,
        },
        {
            'data': torch.zeros(1, 3, 3, 3),
            'label': torch.ones(1, 3, 3, 3),
            'index': 1,
        },
    ]

    collated = collate_fn(batch)

    assert isinstance(collated['data'], list)
    assert isinstance(collated['label'], list)
    assert collated['index'] == [0, 1]
    assert collated['data'][0].shape == (1, 2, 2, 2)
    assert collated['data'][1].shape == (1, 3, 3, 3)

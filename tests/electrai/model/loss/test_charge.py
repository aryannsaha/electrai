from __future__ import annotations

import torch

from electrai.model.loss.charge import ElectronCountLoss, NormMAE


def test_norm_mae_is_averaged_per_sample_after_electron_normalization():
    loss = NormMAE()

    output = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    ).reshape(2, 1, 1, 1, 2)
    target = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    ).reshape(2, 1, 1, 1, 2)

    actual = loss(output, target)

    expected = torch.tensor(0.75)
    torch.testing.assert_close(actual, expected)


def test_norm_mae_supports_variable_sized_list_inputs():
    loss = NormMAE()

    output = [
        torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 2),
        torch.tensor([0.0, 0.0, 0.0]).reshape(1, 1, 1, 3),
    ]
    target = [
        torch.tensor([1.0, 1.0]).reshape(1, 1, 1, 2),
        torch.tensor([1.0, 1.0, 1.0]).reshape(1, 1, 1, 3),
    ]

    actual = loss(output, target)

    expected = torch.tensor((0.5 + 1.0) / 2.0)
    torch.testing.assert_close(actual, expected)


def test_electron_count_loss_matches_relative_charge_error():
    loss = ElectronCountLoss()

    output = torch.tensor([1.0, 2.0]).reshape(1, 1, 1, 1, 2)
    target = torch.tensor([1.0, 1.0]).reshape(1, 1, 1, 1, 2)

    actual = loss(output, target)

    expected = torch.tensor(0.5)
    torch.testing.assert_close(actual, expected)

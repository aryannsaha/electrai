from __future__ import annotations

from torch.utils.data import default_collate


def collate_fn(batch):
    try:
        return default_collate(batch)
    except RuntimeError:
        # Fallback for variable-sized tensors that can't be stacked.
        # Each sample may be a dict {"data": ..., "label": ..., "index": ...}
        # or a tuple (data, label, index).
        if isinstance(batch[0], dict):
            # Dict-returning datasets: collect each key's values into a list
            return {key: [sample[key] for sample in batch] for key in batch[0]}
        else:
            # Tuple-returning datasets (legacy): unzip into separate lists
            x, y, index = zip(*batch, strict=True)
            return list(x), list(y), list(index)

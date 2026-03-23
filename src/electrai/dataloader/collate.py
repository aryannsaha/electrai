from __future__ import annotations

from torch.utils.data import default_collate


def collate_fn(batch):
    try:
        return default_collate(batch)
    except RuntimeError:
        if batch and isinstance(batch[0], dict):
            return {key: [sample[key] for sample in batch] for key in batch[0]}

        x, y, index = zip(*batch, strict=True)
        return {"data": list(x), "label": list(y), "index": list(index)}

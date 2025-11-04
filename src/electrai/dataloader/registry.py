from __future__ import annotations

DATASET_REGISTRY = {}


def register_data(name):
    """Decorator to register new datasets."""

    def decorator(fn):
        DATASET_REGISTRY[name] = fn
        return fn

    return decorator


def get_data(cfg):
    name = cfg.dataset_name
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name](cfg)

from .registry import DATASET_REGISTRY, register_dataset, get_dataset
import importlib

for module in ["chgcar_read"]:  
    importlib.import_module(f"{__name__}.{module}")
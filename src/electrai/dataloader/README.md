# Dataloader Documentation

The dataloader module uses a registry pattern to manage dataset implementations.
Each dataset is registered under a unique key via the `@register_data` decorator, enabling dynamic dataset loading based on configuration parameters.

## 📚 Registered Datasets

| **Key** | **Dataset**           | **Description** |
|----------|-----------------------|-----------------|
| `mp`     | Materials Project     | Loads charge density data from the Materials Project dataset. |

When a configuration specifies `cfg.dataset_name = "mp"`, the corresponding dataset loader is automatically retrieved from the registry and initialized.

This design allows new datasets to be integrated without modifying core dataloader logic. To implement a loader function and register it:

```
@register_data("my_dataset")
def load_my_dataset(cfg):
    # Define how to prepare and return your dataset
    return train_data, test_data
```

## 📁 Materials Project Data

Charge densities are provided in the `task-id.json.gz` format, with instructions for S3 access and download available [here](../../../data/MP/). Crude densities were generated using a minimal DFT calculation in VASP via the [QuAcc](https://github.com/Quantum-Accelerators/quacc) platform and are loade in `CHGCAR` format.

If the data has been downloaded via S3, it is already filtered by `exchange-correlation functional`. Otherwise, a mapping dictionary is required, similar to [map_sample.json.gz](../../../data/MP/map/map_sample.json.gz).

The final dimensions of the input/output charge densities can be adjusted via `cfg.data_size` and `cfg.label_size`, which are processed by the dataloader.

## ⚡ Zarr conversion

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_render_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "render_qm9_xz_slices.py"
    spec = importlib.util.spec_from_file_location("render_qm9_xz_slices", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_slices_can_include_condition_residual_and_label_condition_nmae():
    render_qm9_xz_slices = _load_render_module()
    volume = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
    volumes = {
        "source": volume,
        "condition": volume + 1.0,
        "label": volume + 2.0,
        "output": volume + 3.0,
        "residual": np.ones_like(volume),
        "condition_residual": np.full_like(volume, 2.0),
    }

    figure, slice_meta = render_qm9_xz_slices.plot_slices(
        volumes,
        source_title="Source",
        sample_id="sample-1",
        sample_idx=7,
        plane="xz",
        slice_index=2,
        slice_frac=0.5,
        nmae=0.4321,
        condition_nmae=0.1234,
        sampler_label="heun, 10 steps",
        plot_condition_residual=True,
        output_path=None,
        show=False,
    )

    titles = [axis.get_title() for axis in figure.axes if axis.get_title()]

    assert "condition_residual" in slice_meta
    assert any("Condition NMAE=0.1234" in title for title in titles)
    assert any("Label - Condition" in title for title in titles)

    plt.close(figure)


def test_load_lightning_module_from_checkpoint_restores_on_cpu_first():
    render_qm9_xz_slices = _load_render_module()
    calls = {}

    class DummyModel:
        def requires_grad_(self, enabled):
            calls["requires_grad"] = enabled
            return self

        def to(self, device):
            calls["device"] = str(device)
            return self

        def eval(self):
            calls["eval"] = True
            return self

    class DummyModelCls:
        @staticmethod
        def load_from_checkpoint(checkpoint_path, map_location=None, cfg=None):
            calls["checkpoint_path"] = checkpoint_path
            calls["map_location"] = map_location
            calls["cfg"] = cfg
            return DummyModel()

    cfg = object()
    model = render_qm9_xz_slices.load_lightning_module_from_checkpoint(
        DummyModelCls,
        Path("dummy.ckpt"),
        cfg=cfg,
        device=torch.device("cpu"),
    )

    assert isinstance(model, DummyModel)
    assert calls["checkpoint_path"] == Path("dummy.ckpt")
    assert calls["map_location"] == "cpu"
    assert calls["cfg"] is cfg
    assert calls["requires_grad"] is False
    assert calls["device"] == "cpu"
    assert calls["eval"] is True


def test_load_checkpoint_model_for_training_mode_uses_time_flow_module(monkeypatch):
    render_qm9_xz_slices = _load_render_module()
    calls = {}

    def fake_load(model_cls, checkpoint_path, *, cfg, device):
        calls["model_cls_module"] = model_cls.__module__
        calls["model_cls_name"] = model_cls.__name__
        calls["checkpoint_path"] = checkpoint_path
        calls["cfg"] = cfg
        calls["device"] = device
        return "loaded-model"

    monkeypatch.setattr(
        render_qm9_xz_slices,
        "load_lightning_module_from_checkpoint",
        fake_load,
    )
    cfg = object()

    model, module_type = render_qm9_xz_slices.load_checkpoint_model_for_training_mode(
        "flow_match_with_time",
        Path("dummy.ckpt"),
        cfg=cfg,
        device=torch.device("cpu"),
    )

    assert model == "loaded-model"
    assert module_type == "flow_with_time"
    assert calls["model_cls_module"] == "electrai.lightning_w_time_flow"
    assert calls["model_cls_name"] == "LightningGenerator"
    assert calls["checkpoint_path"] == Path("dummy.ckpt")
    assert calls["cfg"] is cfg
    assert calls["device"] == torch.device("cpu")


def test_visualize_time_flow_checkpoint_uses_condition_rollout(monkeypatch):
    render_qm9_xz_slices = _load_render_module()

    class DummyDataModule:
        def setup(self, stage=None):
            self.val_set = [
                {
                    "data": torch.zeros(2, 2, 2),
                    "label": torch.ones(2, 2, 2),
                    "index": "sample-0",
                }
            ]

    class DummyModel:
        def __init__(self):
            self.n_inference_steps = 9

        def _sample(self, x):
            return x + 1.0

    dummy_model = DummyModel()
    cfg = SimpleNamespace(training_mode="flow_match_with_time", data={})

    monkeypatch.setattr(
        render_qm9_xz_slices,
        "load_cfg_from_checkpoint",
        lambda _checkpoint_path: cfg,
    )
    monkeypatch.setattr(
        render_qm9_xz_slices,
        "instantiate",
        lambda _data_cfg: DummyDataModule(),
    )
    monkeypatch.setattr(
        render_qm9_xz_slices,
        "load_checkpoint_model_for_training_mode",
        lambda *args, **kwargs: (dummy_model, "flow_with_time"),
    )

    result = render_qm9_xz_slices.visualize_qm9_checkpoint_sample(
        Path("dummy.ckpt"),
        sample_idx=0,
        n_steps=3,
        solver="euler",
        device="cpu",
        show=False,
    )

    assert result["module_type"] == "flow_with_time"
    assert result["n_steps"] == 3
    assert result["solver"] == "euler"
    assert torch.equal(result["source"], result["condition"])
    assert torch.equal(result["initial_state"], result["condition"])
    assert torch.equal(result["initial_model_input"], result["condition"])
    assert torch.equal(result["output"], result["condition"] + 1.0)

    plt.close(result["figure"])

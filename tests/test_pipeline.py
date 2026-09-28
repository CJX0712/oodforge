"""tests/test_pipeline.py — 端到端流水线与离线兜底。"""

import numpy as np

from oodforge.core.config import Config, load_config
from oodforge.pipeline.pipeline import OODPipeline


def _cfg():
    return Config(
        random_state=42,
        n_classes=3,
        n_features=10,
        id_train_size=300,
        id_val_size=100,
        id_test_size=150,
        ood_val_size=100,
        ood_test_size=150,
    )


def test_run_returns_summary():
    pipe = OODPipeline(_cfg())
    run = pipe.run("synthetic_far", backend="auto")
    assert "detectors" in run and "router" in run and "calibration" in run
    names = [r["detector"] for r in run["detectors"]]
    assert any(n.startswith("ccor") for n in names)
    assert run["router"]["selected_detector"]


def test_run_numpy_backend_offline():
    pipe = OODPipeline(_cfg())
    run = pipe.run("synthetic_far", backend="numpy")
    # numpy 后端应仍产出有效 AUROC
    ccor = [r for r in run["detectors"] if r["detector"].startswith("ccor")][0]
    assert ccor["auroc"] is not None and ccor["auroc"] > 0.6


def test_benchmark_aggregates():
    pipe = OODPipeline(_cfg())
    res = pipe.benchmark(["synthetic_far", "synthetic_near"])
    assert len(res["runs"]) == 2
    assert isinstance(res["aggregate"], dict)
    assert len(res["aggregate"]) > 0


def test_calibration_reduces_ece():
    pipe = OODPipeline(_cfg())
    run = pipe.run("synthetic_far")
    cal = run["calibration"]
    assert cal["ece_calibrated"] <= cal["ece_uncalibrated"] + 1e-6

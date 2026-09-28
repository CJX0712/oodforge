"""tests/test_core.py — 核心类型/配置/错误校验。"""

import numpy as np
import pytest

from oodforge.core.config import Config, load_config
from oodforge.core.errors import OODForgeError, err
from oodforge.core.types import Dataset, OODSplit, RouterReport


def test_dataset_requires_2d():
    with pytest.raises(ValueError):
        Dataset(X=np.zeros((5,)), name="bad")


def test_dataset_shapes():
    d = Dataset(X=np.zeros((10, 4)), y=np.zeros(10, dtype=int))
    assert d.n_samples == 10 and d.n_features == 4


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("OODFORGE_RANDOM_STATE", "7")
    monkeypatch.setenv("OODFORGE_KNN_K", "3")
    cfg = load_config()
    assert cfg.random_state == 7 and cfg.knn_k == 3


def test_config_explicit_override():
    cfg = load_config(random_state=99, n_classes=2)
    assert cfg.random_state == 99 and cfg.n_classes == 2


def test_error_registry():
    e = err("E100", "boom")
    assert isinstance(e, OODForgeError)
    assert e.code == "E100"
    assert "boom" in str(e)


def test_ood_split_n_classes():
    rng = np.random.default_rng(0)
    id_tr = Dataset(X=rng.standard_normal((20, 4)), y=rng.integers(0, 3, 20))
    id_va = Dataset(X=rng.standard_normal((10, 4)), y=rng.integers(0, 3, 10))
    id_te = Dataset(X=rng.standard_normal((10, 4)), y=rng.integers(0, 3, 10))
    ood_va = Dataset(X=rng.standard_normal((10, 4)))
    ood_te = Dataset(X=rng.standard_normal((10, 4)))
    split = OODSplit(
        id_train=id_tr, id_val=id_va, id_test=id_te, ood_val=ood_va, ood_test=ood_te
    )
    assert split.n_classes == 3


def test_router_report_row():
    r = RouterReport("knn", "matrix", 0.95, 0.3, runner_up="msp")
    row = r.as_row()
    assert row["selected_detector"] == "knn"
    assert row["threshold_fpr95"] == 0.3

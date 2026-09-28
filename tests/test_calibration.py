"""tests/test_calibration.py — 校准器与 ECE 度量。"""

import numpy as np

from oodforge.calibration.metrics import ece, reliability_curve
from oodforge.calibration.scalers import (
    MatrixScaler,
    TemperatureScaler,
    VectorScaler,
    _softmax,
    build_calibrator,
)
from oodforge.core.config import Config


def _logits_labels(seed=0):
    rng = np.random.default_rng(seed)
    n, k = 400, 3
    logits = rng.standard_normal((n, k)) * 2.0
    y = logits.argmax(1)
    return logits, y


def test_temperature_reduces_nll():
    logits, y = _logits_labels()
    t = TemperatureScaler(Config(use_netcal=False))
    t.fit(logits, y)
    # 温度缩放不应使 NLL 变差 (总存在一个 T 不比 1.0 差)
    from oodforge.calibration.scalers import _nll, _softmax

    nll_base = _nll(logits, y)
    nll_cal = _nll(logits / t.T, y)
    assert nll_cal <= nll_base + 1e-6


def test_matrix_scaling_ece_not_worse():
    logits, y = _logits_labels()
    from oodforge.calibration.scalers import _softmax

    ece0 = ece(_softmax(logits), y, 15)
    m = MatrixScaler(Config(use_netcal=False))
    m.fit(logits, y)
    ece1 = ece(m.calibrate(logits), y, 15)
    assert ece1 <= ece0 + 1e-6  # 在训练集上校准应不劣化 ECE


def test_vector_scaling_shapes():
    logits, y = _logits_labels()
    v = VectorScaler(Config(use_netcal=False))
    v.fit(logits, y)
    p = v.calibrate(logits)
    assert p.shape == logits.shape
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)


def test_build_calibrator_unknown():
    try:
        build_calibrator("nope", Config())
        raise AssertionError("应拒绝未知校准器")
    except ValueError:
        pass


def test_reliability_curve_counts():
    logits, y = _logits_labels()
    p = _softmax(logits)
    rc = reliability_curve(p, y, n_bins=10)
    assert sum(rc["counts"]) == len(y)
    assert len(rc["confidence"]) == 10

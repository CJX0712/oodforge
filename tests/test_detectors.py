"""tests/test_detectors.py — 各检测器输出有限 + 方向正确 (越远越 OOD)。"""

import numpy as np

from oodforge.core.config import Config
from oodforge.data.synthetic import make_synthetic_split
from oodforge.detectors.base import knn_distance, logsumexp
from oodforge.detectors.energy import EnergyDetector
from oodforge.detectors.knn import KNNDetector
from oodforge.detectors.mahalanobis import MahalanobisDetector
from oodforge.detectors.msp import MSPDetector
from oodforge.detectors.ocsvm import OneClassSVMDetector
from oodforge.detectors.registry import available_detectors, build_detector
from oodforge.detectors.tsmsp import TemperatureMSPDetector
from oodforge.train.classifiers import build_classifier


def _setup():
    cfg = Config(
        random_state=42,
        n_classes=3,
        n_features=10,
        id_train_size=300,
        id_val_size=100,
        id_test_size=150,
        ood_val_size=100,
        ood_test_size=150,
    )
    split = make_synthetic_split(cfg, ood_kind="far")
    clf = build_classifier("auto", cfg)
    clf.fit(split.id_train.X, split.id_train.y)
    return cfg, split, clf


def test_base_utils():
    z = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    ls = logsumexp(z)
    assert ls.shape == (2,)
    assert np.all(np.isfinite(ls))
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    Xtr = np.array([[0.0, 0.0]])
    d = knn_distance(X, Xtr, k=1)
    assert abs(d[0] - 0.0) < 1e-6
    assert abs(d[1] - 5.0) < 1e-6


def test_msp_finite_in_range():
    cfg, split, clf = _setup()
    det = MSPDetector()
    det.fit(split, clf)
    s_id = det.score(split.id_test.X)
    s_ood = det.score(split.ood_test.X)
    assert np.all(np.isfinite(s_id)) and np.all(np.isfinite(s_ood))
    # ood_score = 1 - max_softmax ∈ [0,1]
    assert s_id.min() >= -1e-9 and s_id.max() <= 1.0 + 1e-9


def test_energy_finite():
    cfg, split, clf = _setup()
    det = EnergyDetector()
    det.fit(split, clf)
    s_ood = det.score(split.ood_test.X)
    s_id = det.score(split.id_test.X)
    assert np.all(np.isfinite(s_ood)) and np.all(np.isfinite(s_id))


def test_mahalanobis_orientation():
    cfg, split, clf = _setup()
    det = MahalanobisDetector(cfg)
    det.fit(split)
    s_ood = det.score(split.ood_test.X)
    s_id = det.score(split.id_test.X)
    assert s_ood.mean() > s_id.mean()


def test_knn_orientation():
    cfg, split, clf = _setup()
    det = KNNDetector(cfg)
    det.fit(split)
    s_ood = det.score(split.ood_test.X)
    s_id = det.score(split.id_test.X)
    assert s_ood.mean() > s_id.mean()


def test_ocsvm_finite():
    cfg, split, clf = _setup()
    det = OneClassSVMDetector(cfg)
    det.fit(split)
    s = det.score(split.ood_test.X)
    assert np.all(np.isfinite(s))


def test_tsmsp_finite_and_tempered():
    cfg, split, clf = _setup()
    det = TemperatureMSPDetector()
    det.fit(split, clf)
    assert det.T > 0
    s_ood = det.score(split.ood_test.X)
    s_id = det.score(split.id_test.X)
    assert np.all(np.isfinite(s_ood)) and np.all(np.isfinite(s_id))
    # 温度缩放后 ood_score 仍在 [0,1]
    assert s_ood.min() >= -1e-9 and s_ood.max() <= 1.0 + 1e-9


def test_registry_all_buildable():
    for name in available_detectors():
        det = build_detector(name, Config())
        assert hasattr(det, "name")

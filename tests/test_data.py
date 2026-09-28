"""tests/test_data.py — 数据生成器正确性 (维度/可区分性/无泄漏)。"""

import numpy as np

from oodforge.core.config import Config
from oodforge.data.loaders import build_split, make_digits_split
from oodforge.data.synthetic import make_synthetic_split


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


def test_synthetic_shapes():
    split = make_synthetic_split(_cfg(), ood_kind="far")
    assert split.id_train.X.shape == (300, 10)
    assert split.id_val.X.shape == (100, 10)
    assert split.ood_test.X.shape == (150, 10)
    assert split.ood_test.y is None  # OOD 无标签


def test_synthetic_far_separable():
    split = make_synthetic_split(_cfg(), ood_kind="far")
    # 远 OOD 应远离 ID 支撑: 用 KNN 距离的均值差验证可区分性
    from oodforge.detectors.knn import KNNDetector

    det = KNNDetector(_cfg())
    det.fit(split)
    id_d = det.score(split.id_test.X).mean()
    ood_d = det.score(split.ood_test.X).mean()
    assert ood_d > id_d  # OOD 更远


def test_synthetic_near_not_trivial():
    split = make_synthetic_split(_cfg(), ood_kind="near")
    from oodforge.detectors.knn import KNNDetector

    det = KNNDetector(_cfg())
    det.fit(split)
    id_d = det.score(split.id_test.X).mean()
    ood_d = det.score(split.ood_test.X).mean()
    # 近 OOD 仍应略远 (存在梯度, 但不应像 far 那样悬殊)
    assert ood_d >= id_d


def test_digits_split_unseen_classes():
    cfg = Config(random_state=0)
    split = make_digits_split(cfg, n_id_classes=5)
    # ID 仅含 0..4, OOD 仅含 5..9
    assert set(np.unique(split.id_train.y)).issubset({0, 1, 2, 3, 4})
    assert split.ood_test.X.shape[1] == 64
    assert split.id_train.X.shape[1] == split.ood_test.X.shape[1]


def test_build_split_factory():
    split = build_split("synthetic_mixed", _cfg())
    assert split.id_test.n_samples == 150

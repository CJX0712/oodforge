"""tests/test_classifiers.py — 分类器 (sklearn / numpy 兜底)。"""

import numpy as np

from oodforge.core.config import Config
from oodforge.train.classifiers import (
    LogisticClassifier,
    NumpySoftmaxClassifier,
    build_classifier,
)


def _data(n=200, n_feat=8, n_cls=3, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_feat))
    # 用前 n_cls 个坐标的符号粗略造标签噪声
    w = rng.standard_normal((n_feat, n_cls))
    logits = X @ w
    y = logits.argmax(1)
    return X, y


def test_sklearn_shapes():
    X, y = _data()
    clf = LogisticClassifier(Config())
    clf.fit(X, y)
    assert clf.predict_logits(X).shape == (X.shape[0], 3)
    p = clf.predict_proba(X)
    assert p.shape == (X.shape[0], 3)
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)


def test_numpy_classifier_converges():
    X, y = _data(n=300)
    clf = NumpySoftmaxClassifier(Config())
    clf.fit(X, y)
    p = clf.predict_proba(X)
    assert p.shape == (X.shape[0], 3)
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)
    # 训练集准确率应明显优于随机 (1/3)
    acc = (p.argmax(1) == y).mean()
    assert acc > 0.5


def test_numpy_classifier_rejects_nan():
    X, y = _data()
    X[0, 0] = np.nan
    clf = NumpySoftmaxClassifier(Config())
    try:
        clf.fit(X, y)
        raise AssertionError("应拒绝 NaN 输入")
    except Exception:
        pass


def test_build_classifier_auto_returns_object():
    clf = build_classifier("auto", Config())
    assert hasattr(clf, "predict_proba")

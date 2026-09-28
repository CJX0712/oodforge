"""train/classifiers.py — ID 分类器 (提供 logits 给能量/Mahalanobis 检测器).

后端:
- sklearn LogisticRegression(solver='lbfgs', 多分类自动 multinomial) — 默认
- 纯 numpy 多分类 softmax 回归 (离线兜底, 无外部依赖)
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.errors import err


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LogisticClassifier:
    """sklearn 多分类逻辑回归封装, 输出 logits 与概率。"""

    name = "logreg"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticClassifier":
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception as exc:
            raise err("E300", f"sklearn 不可用: {exc}")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
            random_state=self.cfg.random_state,
        )
        self._model.fit(X, y)
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise err("E301", "分类器未拟合")
        X = np.asarray(X, dtype=float)
        return np.asarray(self._model.decision_function(X), dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise err("E301", "分类器未拟合")
        X = np.asarray(X, dtype=float)
        return np.asarray(self._model.predict_proba(X), dtype=float)


class NumpySoftmaxClassifier:
    """纯 numpy 多分类 softmax 回归 (离线兜底)。内部标准化特征以保证收敛。"""

    name = "numpy_softmax"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.W = None
        self.b = None
        self.mean_ = None
        self.std_ = None
        self.n_classes_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpySoftmaxClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if not np.all(np.isfinite(X)):
            raise err("E300", "输入含 NaN/Inf")
        self.n_classes_ = int(len(np.unique(y)))
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = (X - self.mean_) / self.std_
        n_feat = Xs.shape[1]
        rng = np.random.default_rng(self.cfg.random_state)
        self.W = rng.standard_normal((n_feat, self.n_classes_)) * 0.01
        self.b = np.zeros(self.n_classes_)
        # 标签 one-hot
        onehot = np.zeros((Xs.shape[0], self.n_classes_))
        onehot[np.arange(Xs.shape[0]), y.astype(int)] = 1.0
        lr = 0.1
        reg = 1e-3
        n_iter = 600
        for _ in range(n_iter):
            logits = Xs @ self.W + self.b
            p = _softmax(logits)
            grad_W = Xs.T @ (p - onehot) / Xs.shape[0] + reg * self.W
            grad_b = (p - onehot).mean(axis=0)
            self.W -= lr * grad_W
            self.b -= lr * grad_b
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise err("E301", "分类器未拟合")
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.std_
        return Xs @ self.W + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax(self.predict_logits(X))


def build_classifier(backend: str = "auto", cfg: Config | None = None) -> object:
    """工厂: backend='auto' 优先 sklearn, 缺失则纯 numpy 兜底。"""
    cfg = cfg or Config()
    if backend == "numpy":
        return NumpySoftmaxClassifier(cfg)
    if backend == "sklearn":
        return LogisticClassifier(cfg)
    # auto
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: F401

        return LogisticClassifier(cfg)
    except Exception:
        return NumpySoftmaxClassifier(cfg)

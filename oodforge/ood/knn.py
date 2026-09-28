"""OODForge · ood/knn — k 近邻距离打分器（sklearn）。

作者: 晨星 (CJX0712)
语义: 分数 = 到第 k 近邻的距离（越大越 OOD）。sklearn 不可用时降级为 numpy 暴力近邻。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E101DataError, E200FitError
from .base import BaseScorer


class KNNScorer(BaseScorer):
    name = "knn"

    def __init__(self, k: int = 5) -> None:
        super().__init__()
        self.k = k
        self._X_train = None

    def fit(self, X_id, probabilities=None, logits=None) -> KNNScorer:
        X = np.asarray(X_id, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise E101DataError("knn 需要 2D 且样本数>=2")
        self._X_train = X
        self._fitted = True
        self._fit_norm(self._raw(X))
        return self

    def _raw(self, X: np.ndarray, probabilities=None, logits=None) -> np.ndarray:
        try:
            from sklearn.neighbors import NearestNeighbors  # type: ignore

            nn = NearestNeighbors(
                n_neighbors=min(self.k, self._X_train.shape[0]), n_jobs=1
            )
            nn.fit(self._X_train)
            d, _ = nn.kneighbors(X)
            return d[:, -1]
        except Exception:  # noqa: BLE001 — 暴力降级
            kk = min(self.k, self._X_train.shape[0])
            diff = X[:, None, :] - self._X_train[None, :, :]
            dist = np.sqrt((diff**2).sum(axis=2))
            return np.sort(dist, axis=1)[:, kk - 1]

    def score(self, X, probabilities=None, logits=None) -> np.ndarray:
        if not self._fitted or self._X_train is None:
            raise E200FitError("knn 未拟合")
        return self._norm(self._raw(np.asarray(X, dtype=np.float64)))

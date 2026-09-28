"""OODForge · ood/iforest — 隔离森林异常打分器（sklearn）。

作者: 晨星 (CJX0712)
语义: 分数 = -score_samples（越大越异常 = 越 OOD）。sklearn 不可用时降级为马氏距离近似。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E101DataError, E200FitError
from .base import BaseScorer


class IsolationForestScorer(BaseScorer):
    name = "iforest"

    def __init__(self, n_estimators: int = 100, seed: int = 42) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.seed = seed
        self._use_sklearn = True
        self._sk_model = None
        self._mean_ = None
        self._prec_ = None

    def fit(self, X_id, probabilities=None, logits=None) -> IsolationForestScorer:
        X = np.asarray(X_id, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise E101DataError("iforest 需要 2D 且样本数>=2")
        try:
            from sklearn.ensemble import IsolationForest  # type: ignore

            self._sk_model = IsolationForest(
                n_estimators=self.n_estimators,
                random_state=self.seed,
                n_jobs=1,
            )
            self._sk_model.fit(X)
            self._use_sklearn = True
        except Exception:  # noqa: BLE001 — 降级
            self._use_sklearn = False
            self._mean_ = X.mean(axis=0)
            cov = np.cov(X, rowvar=False) + 1e-3 * np.eye(X.shape[1])
            self._prec_ = np.linalg.inv(cov)
        self._fitted = True
        self._fit_norm(self._raw(X))
        return self

    def _raw(self, X: np.ndarray, probabilities=None, logits=None) -> np.ndarray:
        if self._use_sklearn and self._sk_model is not None:
            return -self._sk_model.score_samples(X)
        d = X - self._mean_
        return np.einsum("ij,jk,ik->i", d, self._prec_, d)

    def score(self, X, probabilities=None, logits=None) -> np.ndarray:
        if not self._fitted:
            raise E200FitError("iforest 未拟合")
        return self._norm(self._raw(np.asarray(X, dtype=np.float64)))

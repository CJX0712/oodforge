"""OODForge · ood/mahalanobis — 类条件马氏距离打分器（纯 numpy 离线可用）。

作者: 晨星 (CJX0712)
语义: 分数 = 到最近类中心（共享协方差下）的马氏距离，越大越 OOD。
不变量: 拟合后 ID 样本自身距离应远小于一个远高斯 OOD 样本 → 单测交叉验证。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E101DataError, E200FitError
from .base import BaseScorer


class MahalanobisScorer(BaseScorer):
    name = "mahalanobis"

    def __init__(self, reg: float = 1e-3) -> None:
        super().__init__()
        self.reg = reg
        self.means_: list[np.ndarray] = []
        self.prec_: np.ndarray | None = None

    def _raw(self, X: np.ndarray, probabilities=None, logits=None) -> np.ndarray:
        if self.prec_ is None:
            raise E200FitError("mahalanobis 未拟合")
        means = np.asarray(self.means_)  # (C, d)
        d = X[None, :, :] - means[:, None, :]  # (C, n, d)
        # (n, C)：共享协方差下的类条件马氏距离
        maha = np.einsum("cnd,dk,cnk->nc", d, self.prec_, d)
        return maha.min(axis=1)

    def fit(self, X_id, probabilities=None, logits=None) -> MahalanobisScorer:
        X = np.asarray(X_id, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise E101DataError("mahalanobis 需要 2D 且样本数>=2")
        probs = self._to_probs(logits, probabilities)
        if probs is not None:
            labels = probs.argmax(axis=1)
            self.means_ = [X[labels == c].mean(axis=0) for c in np.unique(labels)]
        else:
            self.means_ = [X.mean(axis=0)]
        cov = np.cov(X, rowvar=False)
        cov = np.atleast_2d(cov) + self.reg * np.eye(X.shape[1])
        try:
            self.prec_ = np.linalg.inv(cov)
        except np.linalg.LinAlgError as exc:
            raise E200FitError(f"协方差不可逆: {exc}")
        self._fitted = True
        self._fit_norm(self._raw(X))
        return self

    def score(self, X, probabilities=None, logits=None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self._norm(self._raw(X))

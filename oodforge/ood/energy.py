"""OODForge · ood/energy — 能量基 OOD 打分器（纯 numpy）。

作者: 晨星 (CJX0712)
语义: Energy = -T * logsumexp(logits / T)。ID 置信样本能量低（更负），OOD 能量高（更接近 0）。
      分数 = -Energy（即 logsumexp）越大越 OOD。无 logits 时用 log(probs) 作代理。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E101DataError, E200FitError
from .base import BaseScorer


class EnergyScorer(BaseScorer):
    name = "energy"

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def _raw(self, X: np.ndarray, probabilities=None, logits=None) -> np.ndarray:
        logits = self._to_logits(probabilities, logits)
        if logits is None:
            raise E101DataError("energy 需要 logits 或 probabilities")
        z = logits / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        # 分数 = logsumexp（越大越 OOD）
        return np.log(np.exp(z).sum(axis=1) + 1e-12)

    def fit(self, X_id, probabilities=None, logits=None) -> EnergyScorer:
        if logits is None and probabilities is None:
            raise E101DataError("energy 需要 logits 或 probabilities 以拟合归一化")
        self._fitted = True
        self._fit_norm(
            self._raw(np.asarray(X_id, dtype=np.float64), probabilities, logits)
        )
        return self

    def score(self, X, probabilities=None, logits=None) -> np.ndarray:
        if not self._fitted:
            raise E200FitError("energy 未拟合")
        return self._norm(
            self._raw(np.asarray(X, dtype=np.float64), probabilities, logits)
        )

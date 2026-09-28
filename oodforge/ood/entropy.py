"""OODForge · ood/entropy — 最大概率 / 熵 不确定性打分器（纯 numpy）。

作者: 晨星 (CJX0712)
语义: 分数 = 预测熵（越大越不确定 = 越 OOD）；等价于 1 - max_prob。无 logits 时由 probs 计算。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E101DataError, E200FitError
from .base import BaseScorer


class EntropyScorer(BaseScorer):
    name = "entropy"

    def __init__(self, use_margin: bool = False) -> None:
        super().__init__()
        self.use_margin = use_margin  # True 时用「最小两类概率差」(越大越确定)

    def _raw(self, X: np.ndarray, probabilities=None, logits=None) -> np.ndarray:
        probs = self._to_probs(logits, probabilities)
        if probs is None:
            raise E101DataError("entropy 需要 probabilities 或 logits")
        eps = 1e-12
        p = np.clip(probs, eps, 1.0)
        ent = -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])
        if self.use_margin:
            sp = np.sort(p, axis=1)
            # margin 越大越确定；转为「越大越 OOD」：1 - margin
            return 1.0 - (sp[:, -1] - sp[:, -2])
        return ent

    def fit(self, X_id, probabilities=None, logits=None) -> EntropyScorer:
        if probabilities is None and logits is None:
            raise E101DataError("entropy 需要 probabilities 或 logits")
        self._fitted = True
        self._fit_norm(
            self._raw(np.asarray(X_id, dtype=np.float64), probabilities, logits)
        )
        return self

    def score(self, X, probabilities=None, logits=None) -> np.ndarray:
        if not self._fitted:
            raise E200FitError("entropy 未拟合")
        return self._norm(
            self._raw(np.asarray(X, dtype=np.float64), probabilities, logits)
        )

"""OODForge · ood/base — 打分器基类与归一化工具。

作者: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E201ScoreError


def robust_normalize(scores: np.ndarray) -> np.ndarray:
    """稳健 min-max 归一化到 [0,1]（基于 5%~95% 分位，抗离群点）。

    返回分数越大越 OOD 的归一化值；训练时记录的 weibull/分位参数用于推理时对齐。
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    lo, hi = np.quantile(scores, [0.05, 0.95])
    if hi - lo < 1e-12:
        return np.clip(scores - scores.min(), 0.0, 1.0)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


class BaseScorer:
    """打分器基类：统一拟合/打分骨架 + 推理时归一化（用训练集分位）。"""

    name = "base"

    def __init__(self) -> None:
        self._fitted = False
        self._norm_lo: float | None = None
        self._norm_hi: float | None = None

    def _fit_norm(self, id_scores: np.ndarray) -> None:
        s = np.asarray(id_scores, dtype=np.float64).ravel()
        self._norm_lo, self._norm_hi = np.quantile(s, [0.05, 0.95])
        if self._norm_hi - self._norm_lo < 1e-12:
            self._norm_hi = self._norm_lo + 1e-12

    def _norm(self, scores: np.ndarray) -> np.ndarray:
        if self._norm_lo is None:
            raise E201ScoreError(f"{self.name} 未拟合归一化参数")
        s = np.asarray(scores, dtype=np.float64).ravel()
        return np.clip((s - self._norm_lo) / (self._norm_hi - self._norm_lo), 0.0, 1.0)

    def fit(self, X_id, probabilities=None, logits=None) -> BaseScorer:
        raise NotImplementedError

    def score(self, X, probabilities=None, logits=None) -> np.ndarray:
        raise NotImplementedError

    # 便捷：softmax / logits 派生
    @staticmethod
    def _to_probs(
        logits: np.ndarray | None, probabilities: np.ndarray | None
    ) -> np.ndarray | None:
        if probabilities is not None:
            return np.asarray(probabilities, dtype=np.float64)
        if logits is not None:
            z = np.asarray(logits, dtype=np.float64)
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)
        return None

    @staticmethod
    def _to_logits(
        probabilities: np.ndarray | None, logits: np.ndarray | None
    ) -> np.ndarray | None:
        if logits is not None:
            return np.asarray(logits, dtype=np.float64)
        if probabilities is not None:
            p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
            return np.log(p)
        return None

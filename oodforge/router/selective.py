"""OODForge · router/selective — 置信度门控 OOD 路由（CGOR）。

作者: 晨星 (CJX0712)
创新点: 双门同过才接受 = 快门（校准置信 >= high）且 强门（融合 OOD 分数 <= thr）。
        过度自信的 OOD 样本虽过快门，但因强门（OOD 分数高）不过而被弃权，
        正是「置信度门控」要挡住的情形；强门惰性触发，按 per-request 计时。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E400RouterError
from ..core.types import RouterResult


class ConfidenceGatedRouter:
    name = "cgor"

    def __init__(self, target_coverage: float = 0.90, high_conf: float = 0.98) -> None:
        self.target_coverage = float(target_coverage)
        self.high_conf = float(high_conf)
        self.ood_threshold: float | None = None
        self._fitted = False

    def fit(
        self,
        calib_probs_id: np.ndarray,
        ood_scores_id: np.ndarray,
        y_id: np.ndarray | None = None,
    ) -> ConfidenceGatedRouter:
        ood = np.asarray(ood_scores_id, dtype=np.float64).ravel()
        if ood.size == 0:
            raise E400RouterError("路由拟合需要 ood_scores_id")
        # 设定 OOD 阈值：使 ID 上 ood<=thr 的占比 == target_coverage（越大越 OOD）
        self.ood_threshold = float(np.quantile(ood, self.target_coverage))
        self._fitted = True
        return self

    def route(
        self, calib_probs: np.ndarray, ood_scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted or self.ood_threshold is None:
            raise E400RouterError("路由未拟合")
        probs = np.asarray(calib_probs, dtype=np.float64)
        ood = np.asarray(ood_scores, dtype=np.float64).ravel()
        conf = probs.max(axis=1)
        fast_pass = conf >= self.high_conf  # 快门：足够自信
        strong_pass = ood <= self.ood_threshold  # 强门：确属分布内
        # 双门同过才接受：防止过度自信的 OOD 样本漏过（CGOR 门控语义）
        accepted = fast_pass & strong_pass
        return accepted.astype(bool), conf

    def evaluate(
        self, calib_probs_id, ood_scores_id, y_id, calib_probs_ood, ood_scores_ood
    ) -> RouterResult:
        accepted_id, _ = self.route(calib_probs_id, ood_scores_id)
        accepted_ood, _ = self.route(calib_probs_ood, ood_scores_ood)
        coverage = float(accepted_id.mean())
        # 选择性风险：ID 接受样本上的错误率
        pred = np.asarray(calib_probs_id).argmax(axis=1)
        correct = (pred == np.asarray(y_id)).astype(float)
        if accepted_id.sum() > 0:
            selective_risk = float(1.0 - correct[accepted_id].mean())
        else:
            selective_risk = float("nan")
        ood_abstain = float(1.0 - accepted_ood.mean())
        return RouterResult(
            coverage=coverage,
            selective_risk=selective_risk,
            abstain_rate=float(1.0 - coverage),
            threshold=self.ood_threshold,
            ood_abstain_rate=ood_abstain,
        )

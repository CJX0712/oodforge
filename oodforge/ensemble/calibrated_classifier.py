"""ensemble/calibrated_classifier.py — 校准后的分类器包装。

把基础分类器的 logits 经校准器映射为校准概率, 供 MSP 检测器使用。
仅替换 predict_proba, 保留 predict_logits (能量/Mahalanobis 用原始 logits)。
"""

from __future__ import annotations

import numpy as np


class CalibratedClassifier:
    name = "calibrated_classifier"

    def __init__(self, base, calibrator) -> None:
        self.base = base
        self.calibrator = calibrator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CalibratedClassifier":
        self.base.fit(X, y)
        logits = self.base.predict_logits(np.asarray(X, dtype=float))
        self.calibrator.fit(logits, np.asarray(y))
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        return self.base.predict_logits(np.asarray(X, dtype=float))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.base.predict_logits(np.asarray(X, dtype=float))
        return self.calibrator.calibrate(logits)

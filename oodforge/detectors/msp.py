"""detectors/msp.py — 最大 softmax 概率 (MSP) 基线检测器。

Hendrycks & Gimpel 2016: 置信度 = max softmax。OOD 通常置信度更低。
ood_score = 1 - max_softmax (越大越 OOD)。
"""

from __future__ import annotations

import numpy as np

from ..core.interfaces import ClassifierProtocol
from ..core.types import OODSplit


class MSPDetector:
    name = "msp"

    def __init__(self) -> None:
        self.clf = None

    def fit(
        self, split: OODSplit, classifier: ClassifierProtocol | None = None
    ) -> "MSPDetector":
        if classifier is None:
            raise ValueError("MSPDetector 需要分类器")
        self.clf = classifier
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        p = self.clf.predict_proba(np.asarray(X, dtype=float))
        return 1.0 - p.max(axis=1)

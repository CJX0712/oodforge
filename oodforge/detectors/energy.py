"""detectors/energy.py — 能量基线 (Energy-Based OOD)。

Liu et al. 2020: 能量 E(x) = -logsumexp(f(x))。OOD 样本能量更高。
ood_score = E = -logsumexp(logits) (越大越 OOD)。
"""

from __future__ import annotations

import numpy as np

from ..core.interfaces import ClassifierProtocol
from ..core.types import OODSplit
from .base import logsumexp


class EnergyDetector:
    name = "energy"

    def __init__(self) -> None:
        self.clf = None

    def fit(
        self, split: OODSplit, classifier: ClassifierProtocol | None = None
    ) -> "EnergyDetector":
        if classifier is None:
            raise ValueError("EnergyDetector 需要分类器")
        self.clf = classifier
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        logits = self.clf.predict_logits(np.asarray(X, dtype=float))
        # 能量 E = -logsumexp(f); OOD 能量更高 → 直接以 E 为 ood_score
        return -logsumexp(logits)

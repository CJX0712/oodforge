"""detectors/knn.py — k 近邻距离检测器。

特征空间标准化后, 取样本到 ID 训练集第 k 近邻的欧氏距离。
OOD 远离 ID 支撑 → 距离大。ood_score = kNN 距离 (越大越 OOD)。
依赖: 仅 numpy。
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.types import OODSplit
from .base import knn_distance


class KNNDetector:
    name = "knn"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.X_train_ = None

    def fit(self, split: OODSplit, classifier=None) -> "KNNDetector":
        self.X_train_ = np.asarray(split.id_train.X, dtype=float)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None:
            raise ValueError("KNNDetector 未拟合")
        return knn_distance(np.asarray(X, dtype=float), self.X_train_, k=self.cfg.knn_k)

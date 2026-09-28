"""detectors/oneclass.py — 单类异常检测 (单类 SVM / 高斯密度兜底)。

sklearn OneClassSVM(nu=0.05, novelty=True): decision_function 正=inlier。
ood_score = -decision_function (越大越 OOD)。
后端缺失时降级为单高斯对数密度 (ood_score = 0.5 * Mahalanobis 距离平方)。
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.types import OODSplit


class OneClassSVMDetector:
    name = "ocsvm"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._model = None
        self._mean = None
        self._prec = None
        self.used_fallback = False

    def fit(self, split: OODSplit, classifier=None) -> "OneClassSVMDetector":
        X = np.asarray(split.id_train.X, dtype=float)
        try:
            from sklearn.svm import OneClassSVM

            self._model = OneClassSVM(nu=0.05, gamma="scale", kernel="rbf")
            self._model.fit(X)
            self.used_fallback = False
        except Exception:
            mu = X.mean(axis=0)
            cov = np.cov(X, rowvar=False)
            cov = cov + self.cfg.mahalanobis_reg * np.eye(cov.shape[0])
            try:
                prec = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                prec = np.linalg.pinv(cov)
            self._mean = mu
            self._prec = prec
            self._model = None
            self.used_fallback = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._model is not None:
            return -np.asarray(self._model.decision_function(X), dtype=float)
        diff = X - self._mean
        d2 = np.einsum("ij,jk,ik->i", diff, self._prec, diff)
        return 0.5 * d2

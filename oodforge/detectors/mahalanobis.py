"""detectors/mahalanobis.py — 类条件马氏距离 (Mahalanobis) 检测器。

Lee et al. 2018: 在特征空间拟合类条件高斯, OOD = 到最近类中心马氏距离大。
ood_score = min_c d_Maha(x; mu_c, Sigma) (距离越大越 OOD)。
依赖: 仅 numpy (纯离线可跑)。
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.types import OODSplit


class MahalanobisDetector:
    name = "mahalanobis"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.means_ = None
        self.prec_ = None

    def fit(self, split: OODSplit, classifier=None) -> "MahalanobisDetector":
        X = np.asarray(split.id_train.X, dtype=float)
        y = np.asarray(split.id_train.y)
        classes = np.unique(y)
        means = []
        covs = []
        for c in classes:
            Xc = X[y == c]
            if Xc.shape[0] < 2:
                continue
            means.append(Xc.mean(axis=0))
            covs.append(np.cov(Xc, rowvar=False))
        if not means:
            raise ValueError("无法拟合 Mahalanobis: 类别样本不足")
        means = np.asarray(means)
        # 共享 (tied) 协方差 + 正则化, 保证可逆
        pooled = np.mean(np.asarray(covs), axis=0)
        pooled = pooled + self.cfg.mahalanobis_reg * np.eye(pooled.shape[0])
        try:
            prec = np.linalg.inv(pooled)
        except np.linalg.LinAlgError:
            prec = np.linalg.pinv(pooled)
        self.means_ = means
        self.prec_ = prec
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None:
            raise ValueError("Mahalanobis 未拟合")
        X = np.asarray(X, dtype=float)
        # 到每个类中心的马氏距离平方
        d2 = []
        for mu in self.means_:
            diff = X - mu
            md = np.einsum("ij,jk,ik->i", diff, self.prec_, diff)
            d2.append(md)
        d2 = np.stack(d2, axis=1)  # (n, n_classes)
        min_d = np.sqrt(d2.min(axis=1))
        return min_d  # 距离越大越 OOD

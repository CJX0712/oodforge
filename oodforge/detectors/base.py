"""detectors/base.py — 检测器公共工具。

约定: 所有检测器输出 ood_score, **越大越 OOD** (与 AUROC 正类方向一致)。
"""

from __future__ import annotations

import numpy as np


def logsumexp(z: np.ndarray) -> np.ndarray:
    """数值稳定 logsumexp, 沿 axis=1。"""
    z = np.asarray(z, dtype=float)
    m = z.max(axis=1, keepdims=True)
    s = np.exp(z - m).sum(axis=1, keepdims=True)
    return (m + np.log(s)).ravel()


def standardize(
    X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score 标准化; 若未给均值/标准差则用自身统计 (测试期用训练统计更佳)。"""
    X = np.asarray(X, dtype=float)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.asarray(std, dtype=float)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def knn_distance(X: np.ndarray, X_train: np.ndarray, k: int) -> np.ndarray:
    """每个 X 样本到训练集第 k 近邻的欧氏距离 (标准化空间)。

    返回 (n,) 。对中小数据直接广播, 避免引入 scipy/sklearn 依赖。
    """
    X = np.asarray(X, dtype=float)
    X_train = np.asarray(X_train, dtype=float)
    # 用训练集统计标准化, 保证训练/测试同尺度
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    Xtr = (X_train - mean) / std
    # 块化处理显存
    n = Xs.shape[0]
    kk = min(k, Xtr.shape[0])
    dists = np.empty(n, dtype=float)
    block = 512
    for i in range(0, n, block):
        Xb = Xs[i : i + block]
        d2 = ((Xb[:, None, :] - Xtr[None, :, :]) ** 2).sum(axis=2)  # (b, Ntr)
        kn = np.sort(d2, axis=1)[:, kk - 1]
        dists[i : i + block] = np.sqrt(kn)
    return dists

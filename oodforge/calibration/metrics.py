"""calibration/metrics.py — 概率校准度量 (ECE / 可靠性曲线)。"""

from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """期望校准误差 (Expected Calibration Error)。

    probs: (n, k) 预测概率; y: (n,) 整型标签。
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    total = conf.shape[0]
    ece_val = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = correct[mask].mean()
        conf_b = conf[mask].mean()
        ece_val += n_b / total * abs(acc_b - conf_b)
    return float(ece_val)


def reliability_curve(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    """返回可靠性曲线数据: 各 bin 的平均置信度 / 准确率 / 样本数。"""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    confs, accs, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        n_b = int(mask.sum())
        counts.append(n_b)
        confs.append(float(conf[mask].mean()) if n_b else 0.0)
        accs.append(float(correct[mask].mean()) if n_b else 0.0)
    return {"confidence": confs, "accuracy": accs, "counts": counts, "n_bins": n_bins}

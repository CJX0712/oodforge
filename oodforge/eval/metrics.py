"""eval/metrics.py — OOD 评测指标 (AUROC / AUPR / FPR@95TPR)。

约定: 输入 ood_score (越大越 OOD), y 二值标签 (1=OOD, 0=ID)。
sklearn 为首选后端; 缺失时退化为纯 numpy (Mann-Whitney AUROC + 线性插值 AUPR)。
"""

from __future__ import annotations

import numpy as np


def _try_sklearn():
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return roc_auc_score, average_precision_score
    except Exception:
        return None, None


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y).ravel().astype(int)
    ra, pa = _try_sklearn()
    if ra is not None:
        return float(ra(y, scores))
    # 纯 numpy Mann-Whitney AUROC
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 平均秩处理并列
    sa = np.sort(scores, kind="mergesort")
    dup = np.concatenate([[False], sa[1:] == sa[:-1]])
    if dup.any():
        sorted_ranks = ranks[order]
        start = 0
        for i in range(1, len(sa)):
            if sa[i] != sa[i - 1]:
                if i - start > 1:
                    blk = sorted_ranks[start:i]
                    sorted_ranks[start:i] = blk.mean()
                start = i
        if len(sa) - start > 1:
            blk = sorted_ranks[start:]
            sorted_ranks[start:] = blk.mean()
        ranks = np.empty_like(ranks)
        ranks[order] = sorted_ranks
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def aupr(scores: np.ndarray, y: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y).ravel().astype(int)
    _, pa = _try_sklearn()
    if pa is not None:
        return float(pa(y, scores))
    # 纯 numpy: 按阈值扫描计算 PR 曲线 (线性插值近似)
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(y.sum()), 1)
    # 去重 recall 点
    idx = np.concatenate([[True], rec[1:] != rec[:-1]])
    return float(
        np.trapezoid(prec[idx], rec[idx])
        if hasattr(np, "trapezoid")
        else np.trapz(prec[idx], rec[idx])
    )


def fpr95(scores: np.ndarray, y: np.ndarray) -> float:
    """FPR@95TPR: 在 TPR=0.95 处 ID 被误判为 OOD 的比例。"""
    scores = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y).ravel().astype(int)
    ood = scores[y == 1]
    idn = scores[y == 0]
    if len(ood) == 0 or len(idn) == 0:
        return float("nan")
    thr = np.percentile(ood, 5.0)  # 95% OOD 高于该阈值 → TPR=0.95
    return float(np.mean(idn >= thr))


def evaluate(scores: np.ndarray, y: np.ndarray) -> dict:
    return {
        "auroc": round(auroc(scores, y), 4),
        "aupr": round(aupr(scores, y), 4),
        "fpr95": round(fpr95(scores, y), 4),
    }

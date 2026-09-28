"""OODForge · eval/metrics — 评测指标（AUROC/AUPRC/FPR95/ECE/选择性风险）。

作者: 晨星 (CJX0712)
注意: sklearn 指标导入改名 `_sk_*`，避免与作用域内同名函数递归调用（踩坑）。
约定: y_true: 1=OOD, 0=ID。score 越大越 OOD。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E101DataError

try:  # 可选后端
    from sklearn import metrics as _sk_metrics

    _HAVE_SK = True
except Exception:  # noqa: BLE001
    _sk_metrics = None
    _HAVE_SK = False


def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    if _HAVE_SK:
        return float(_sk_metrics.roc_auc_score(y_true, score))
    # 纯 numpy 实现（梯形法，按阈值排序）
    order = np.argsort(score)
    s_sorted = score[order]
    # 秩平均处理并列
    ranks = np.empty_like(score, dtype=float)
    idx = 0
    while idx < len(score):
        j = idx
        while j + 1 < len(score) and s_sorted[j + 1] == s_sorted[idx]:
            j += 1
        ranks[order[idx : j + 1]] = (idx + j) / 2.0 + 0.5
        idx = j + 1
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float(
        (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    if _HAVE_SK:
        return float(_sk_metrics.average_precision_score(y_true, score))
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (y.sum() + 1e-12)
    return (
        float(np.trapz(prec, rec))
        if hasattr(np, "trapz")
        else float(np.trapezoid(prec, rec))
    )


def fpr_at_tpr(
    y_true: np.ndarray, score: np.ndarray, tpr_target: float = 0.95
) -> float:
    """在给定 TPR 下的假阳性率（越小越好）。"""
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    if _HAVE_SK:
        fpr, tpr, _ = _sk_metrics.roc_curve(y_true, score)
        idx = np.searchsorted(tpr, tpr_target)
        idx = min(idx, len(fpr) - 1)
        return float(fpr[idx])
    # numpy 实现
    order = np.argsort(-score)
    y = y_true[order]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / (n_pos + 1e-12)
    fpr = fp / (n_neg + 1e-12)
    idx = np.searchsorted(tpr, tpr_target)
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """期望校准误差（等宽分箱）。probs: (n, k) 校准概率；y_true: (n,) 真实标签。"""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true)
    if probs.ndim != 2 or len(y_true) != probs.shape[0]:
        raise E101DataError("ece 需要 (n,k) 概率与 (n,) 标签")
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    total = len(y_true)
    score = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf_mean = conf[mask].mean()
        score += mask.sum() / total * abs(acc - conf_mean)
    return float(score)


def selective_risk_curve(
    calib_conf: np.ndarray, is_correct: np.ndarray, coverages: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """给定校准置信与是否正确(1/0)，返回各覆盖率下的选择性风险与覆盖率。

    return: (coverages, selective_risks)。接受「置信最高」的 1-coverage 部分。
    """
    conf = np.asarray(calib_conf, dtype=np.float64)
    correct = np.asarray(is_correct, dtype=float)
    order = np.argsort(-conf)
    correct_s = correct[order]
    cum_correct = np.cumsum(correct_s)
    n = len(conf)
    if coverages is None:
        coverages = np.linspace(0.05, 1.0, 20)
    risks = []
    for c in coverages:
        k = max(1, round(c * n))
        k = min(k, n)
        # 接受前 k 个（最高置信）
        risks.append(1.0 - cum_correct[k - 1] / k)
    return np.asarray(coverages), np.asarray(risks)


def coverage_at_selective_risk(
    calib_conf: np.ndarray, is_correct: np.ndarray, target_risk: float
) -> float:
    """达到目标选择性风险所需的最小覆盖率。"""
    cov, risk = selective_risk_curve(calib_conf, is_correct)
    idx = np.argmax(risk <= target_risk)
    return float(cov[idx]) if risk[idx] <= target_risk else float(cov[-1])

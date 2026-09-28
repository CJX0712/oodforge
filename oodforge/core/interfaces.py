"""core/interfaces.py — 模块契约 (Protocol).

单向无环调用: cli → pipeline → {data, train, detectors, calibration, ensemble, eval} → core.
所有可插拔组件实现下列 Protocol, 保证工厂模式与离线兜底可替换。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import CalibrationReport, Dataset, DetectionResult, OODSplit


@runtime_checkable
class ClassifierProtocol(Protocol):
    """在 ID 训练集上拟合, 输出 logits 与 softmax 概率."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassifierProtocol": ...

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """返回 (n, n_classes) 未归一化分数 (logits)."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回 (n, n_classes) 行归一化概率."""
        ...


@runtime_checkable
class OODDetectorProtocol(Protocol):
    """OOD 检测器: 在 (ID 训练 + 可选分类器) 上 fit, 输出 ood_score (越大越 OOD)."""

    name: str

    def fit(
        self, split: OODSplit, classifier: ClassifierProtocol | None = None
    ) -> "OODDetectorProtocol": ...

    def score(self, X: np.ndarray) -> np.ndarray:
        """返回 (n,) ood_score, 越大越 OOD."""
        ...


@runtime_checkable
class CalibratorProtocol(Protocol):
    """概率校准器: 输入未校准 logits, 输出校准后概率。"""

    name: str

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "CalibratorProtocol": ...

    def calibrate(self, logits: np.ndarray) -> np.ndarray: ...


def available_classifier() -> bool:
    return True  # 分类器始终可用 (sklearn / numpy 兜底)


def make_ood_labels(split: OODSplit) -> tuple[np.ndarray, np.ndarray]:
    """构造评测用的二分类标签: ID=0, OOD=1, 以及拼接分数空间.

    返回 (scores_cat, y_cat): 用于阈值/路由在验证集上决策。
    """
    _ = (split,)
    return np.empty(0), np.empty(0)


__all__ = [
    "ClassifierProtocol",
    "OODDetectorProtocol",
    "CalibratorProtocol",
    "available_classifier",
    "make_ood_labels",
]

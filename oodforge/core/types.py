"""core/types.py — 全局数据类型 (dataclass 契约).

所有模块间流转的数据结构在此声明，保证跨模块公平评测与单一职责。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dataset:
    """一个简单数据集 (特征矩阵 + 可选标签)."""

    X: np.ndarray
    y: np.ndarray | None = None
    name: str = "dataset"

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        if self.y is not None:
            self.y = np.asarray(self.y)
        if self.X.ndim != 2:
            raise ValueError("Dataset.X 必须是二维 (n_samples, n_features)")

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


@dataclass
class OODSplit:
    """OOD 评测标准划分: ID 训练/验证/测试 + OOD 验证/测试.

    约定: ood_* 样本标签在评测时统一视为正类 (y=1)，ID 为 y=0。
    """

    id_train: Dataset
    id_val: Dataset
    id_test: Dataset
    ood_val: Dataset
    ood_test: Dataset

    @property
    def n_classes(self) -> int:
        return int(len(np.unique(self.id_train.y)))


@dataclass
class DetectionResult:
    """单个检测器在一批样本上的输出.

    ood_score 语义统一: **越大越 OOD** (与 AUROC 正类方向一致)。
    """

    name: str
    ood_score: np.ndarray

    def __post_init__(self) -> None:
        self.ood_score = np.asarray(self.ood_score, dtype=float).ravel()


@dataclass
class DetectorReport:
    """检测器在 OOD 测试集上的评测指标."""

    name: str
    auroc: float
    aupr: float
    fpr95: float  # FPR@95TPR

    def as_row(self) -> dict:
        return {
            "detector": self.name,
            "auroc": round(self.auroc, 4),
            "aupr": round(self.aupr, 4),
            "fpr95": round(self.fpr95, 4),
        }


@dataclass
class CalibrationReport:
    """校准器在 ID 验证/测试集上的指标."""

    method: str
    ece: float
    n_bins: int
    available: bool = True

    def as_row(self) -> dict:
        return {
            "method": self.method,
            "ece": round(self.ece, 4),
            "n_bins": self.n_bins,
            "available": self.available,
        }


@dataclass
class RouterReport:
    """CCOR 路由器的选型与阈值决策."""

    selected_detector: str
    selected_calibrator: str
    val_auroc_of_selected: float
    threshold_fpr95: float
    runner_up: str = ""
    fallback_used: bool = False
    notes: str = ""

    def as_row(self) -> dict:
        return {
            "selected_detector": self.selected_detector,
            "selected_calibrator": self.selected_calibrator,
            "val_auroc": round(self.val_auroc_of_selected, 4),
            "threshold_fpr95": round(self.threshold_fpr95, 6),
            "runner_up": self.runner_up,
            "fallback_used": self.fallback_used,
            "notes": self.notes,
        }

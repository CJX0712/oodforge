"""OODForge · core/types — 领域数据类型（dataclass 契约先行）。

作者: 晨星 (CJX0712)
约定: 所有 OOD 打分器的 `score` 语义统一为「分数越大越 OOD」(越大越异常/越远离分布)。
      calibrate 返回的是「越大越可信」的校准后概率（与 OOD 分数量纲相反，路由层负责对齐）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    """最小数据集容器。y 可为 None（纯无标签场景）。"""

    X: np.ndarray
    y: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.y is not None:
            self.y = np.asarray(self.y)


@dataclass
class OODScores:
    """单个打分器在某批样本上的原始分数（越大越 OOD）。"""

    name: str
    scores: np.ndarray  # shape (n_samples,)

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=np.float64).ravel()


@dataclass
class OODResult:
    """一个打分器的 OOD 检测评测结果（ID vs OOD）。"""

    name: str
    auroc: float
    auprc: float
    fpr95: float  # 在 95% TPR 处的假阳性率（越小越好）
    mean_score_id: float
    mean_score_ood: float


@dataclass
class CalibResult:
    """校准器评测：校准前/后 ECE。"""

    method: str
    n_bins: int
    ece_before: float
    ece_after: float
    nll_before: float | None = None
    nll_after: float | None = None


@dataclass
class RouterResult:
    """选择性预测路由结果。"""

    coverage: float  # ID 接受样本比例
    selective_risk: float  # ID 接受样本上的错误率（越低越好）
    abstain_rate: float  # ID 弃权比例 = 1 - coverage
    threshold: float  # 触发弃权的 OOD 分数阈值
    ood_abstain_rate: float = float("nan")  # OOD 样本被弃权比例（越高越好）


@dataclass
class BenchmarkRecord:
    """单条基准记录，便于落盘为 benchmark.json。"""

    name: str
    metric: str
    value: float
    note: str = ""


@dataclass
class PipelineConfig:
    """端到端流水线配置。支持 ENV_OODFORGE_* 覆盖（见 core/config）。"""

    random_state: int = 42
    base_classifier: str = "logreg"  # logreg | rf
    n_classes: int = 3
    ensemble_trials: int = 24
    calib_method: str = (
        "temperature"  # temperature | vector | matrix | isotonic | platt
    )
    target_coverage: float = 0.90
    use_optional_backends: bool = True

    def to_dict(self) -> dict:
        return {
            "random_state": self.random_state,
            "base_classifier": self.base_classifier,
            "n_classes": self.n_classes,
            "ensemble_trials": self.ensemble_trials,
            "calib_method": self.calib_method,
            "target_coverage": self.target_coverage,
            "use_optional_backends": self.use_optional_backends,
        }

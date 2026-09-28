"""OODForge — 分布外检测 (OOD) 与概率校准系统.

模块化、CPU/离线可运行、一键可复现。复用顶级开源 (scikit-learn / scipy /
numpy / netcal)，含纯 numpy·sklearn 离线兜底链路。

作者: 晨星 (CJX0712)
"""

from .core.config import Config, load_config
from .core.types import (
    CalibrationReport,
    Dataset,
    DetectionResult,
    DetectorReport,
    OODSplit,
    RouterReport,
)

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = [
    "Config",
    "load_config",
    "Dataset",
    "OODSplit",
    "DetectionResult",
    "DetectorReport",
    "CalibrationReport",
    "RouterReport",
    "__version__",
    "__author__",
]

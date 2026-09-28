"""core — OODForge 核心层: 类型 / 错误 / 配置 / 接口契约."""

from .config import Config, dataclass_replace, load_config
from .errors import OODForgeError, err
from .interfaces import (
    CalibratorProtocol,
    ClassifierProtocol,
    OODDetectorProtocol,
    available_classifier,
)
from .types import (
    CalibrationReport,
    Dataset,
    DetectionResult,
    DetectorReport,
    OODSplit,
    RouterReport,
)

__all__ = [
    "Config",
    "load_config",
    "dataclass_replace",
    "OODForgeError",
    "err",
    "ClassifierProtocol",
    "OODDetectorProtocol",
    "CalibratorProtocol",
    "available_classifier",
    "Dataset",
    "OODSplit",
    "DetectionResult",
    "DetectorReport",
    "CalibrationReport",
    "RouterReport",
]

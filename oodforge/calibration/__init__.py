"""calibration — 概率校准层 (温度/向量/矩阵缩放 + ECE 度量)。"""

from .metrics import ece, reliability_curve
from .scalers import (
    CALIBRATORS,
    MatrixScaler,
    TemperatureScaler,
    VectorScaler,
    build_calibrator,
)

__all__ = [
    "TemperatureScaler",
    "VectorScaler",
    "MatrixScaler",
    "build_calibrator",
    "CALIBRATORS",
    "ece",
    "reliability_curve",
]

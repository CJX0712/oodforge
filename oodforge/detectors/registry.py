"""detectors/registry.py — 检测器注册表与工厂。"""

from __future__ import annotations

from ..core.config import Config
from ..core.errors import err
from .base import knn_distance, logsumexp, standardize
from .energy import EnergyDetector
from .knn import KNNDetector
from .mahalanobis import MahalanobisDetector
from .msp import MSPDetector
from .ocsvm import OneClassSVMDetector
from .tsmsp import TemperatureMSPDetector

DETECTORS = {
    "msp": MSPDetector,
    "energy": EnergyDetector,
    "mahalanobis": MahalanobisDetector,
    "knn": KNNDetector,
    "ocsvm": OneClassSVMDetector,
    "tsmsp": TemperatureMSPDetector,
}


def build_detector(name: str, cfg: Config | None = None):
    if name not in DETECTORS:
        raise err("E400", f"未知检测器: {name}")
    det = DETECTORS[name]
    try:
        return (
            det(cfg) if cfg is not None and name in ("mahalanobis", "knn", "ocsvm") else det()
        )
    except TypeError:
        return det()


def available_detectors() -> list[str]:
    return list(DETECTORS)


__all__ = [
    "DETECTORS",
    "build_detector",
    "available_detectors",
    "logsumexp",
    "standardize",
    "knn_distance",
    "MSPDetector",
    "EnergyDetector",
    "MahalanobisDetector",
    "KNNDetector",
    "OneClassSVMDetector",
    "TemperatureMSPDetector",
]

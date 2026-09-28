"""OODForge · ood — OOD 打分器与集成路由（置信度门控 OOD 路由 CGOR）。

作者: 晨星 (CJX0712)
约定: 所有 scorer.score 返回「越大越 OOD」。
"""

from .energy import EnergyScorer
from .ensemble import EnsembleRouter
from .entropy import EntropyScorer
from .gmm import GMMScorer
from .iforest import IsolationForestScorer
from .knn import KNNScorer
from .mahalanobis import MahalanobisScorer

__all__ = [
    "EnergyScorer",
    "EnsembleRouter",
    "EntropyScorer",
    "GMMScorer",
    "IsolationForestScorer",
    "KNNScorer",
    "MahalanobisScorer",
]

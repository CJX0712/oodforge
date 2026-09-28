"""OODForge · core/interfaces — 协议层（Protocol / ABC）。

作者: 晨星 (CJX0712)
单向依赖: cli -> pipeline -> {data, ood, calib, router, eval} -> core。
接口语义：
  - OODScorer.score 越大越 OOD
  - Calibrator.calibrate 返回校准后「越大越可信」概率（行和=1）
  - Router.route 输入 (校准概率, ood分数)，输出是否接受 + 置信
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OODScorer(Protocol):
    name: str

    def fit(
        self,
        X_id: np.ndarray,
        probabilities: np.ndarray | None = None,
        logits: np.ndarray | None = None,
    ) -> OODScorer:
        """在 ID 训练集上拟合（可借助分类器输出）。"""
        ...

    def score(
        self,
        X: np.ndarray,
        probabilities: np.ndarray | None = None,
        logits: np.ndarray | None = None,
    ) -> np.ndarray:
        """返回 (n_samples,) 越大越 OOD 的分数。"""
        ...


@runtime_checkable
class Calibrator(Protocol):
    name: str

    def fit(self, probs: np.ndarray, y: np.ndarray) -> Calibrator: ...

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """返回校准后概率，每行和为 1。"""
        ...


@runtime_checkable
class Router(Protocol):
    name: str

    def fit(
        self, calib_probs_id: np.ndarray, ood_scores_id: np.ndarray, y_id: np.ndarray
    ) -> Router: ...

    def route(
        self, calib_probs: np.ndarray, ood_scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 (accepted: bool array, confidence: float array)。"""
        ...

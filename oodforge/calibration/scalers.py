"""calibration/scalers.py — 概率校准器 (Temperature / Vector / Matrix Scaling)。

SOTA 后端: netcal (可选, 优先)。离线兜底: 纯 numpy (网格搜索 / 梯度下降)。
约定: fit(logits, y) → calibrate(logits) 返回校准后 (n,k) 概率。
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, y: np.ndarray) -> float:
    p = _softmax(logits)
    idx = np.arange(logits.shape[0])
    return float(-np.mean(np.log(p[idx, y.astype(int)] + 1e-12)))


class TemperatureScaler:
    name = "temperature"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.T = 1.0
        self._netcal = None

    def _use_netcal(self) -> bool:
        if not self.cfg.use_netcal:
            return False
        try:
            import netcal  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float)
        if self._use_netcal():
            try:
                from netcal import TemperatureScaling

                self._netcal = TemperatureScaling()
                self._netcal.fit(_softmax(logits), np.asarray(y))
                return self
            except Exception:
                self._netcal = None
        grid = np.unique(
            np.concatenate([np.linspace(0.2, 2.0, 10), np.linspace(2.0, 20.0, 10)])
        )
        best_T, best = 1.0, float("inf")
        for T in grid:
            nll = _nll(logits / T, y)
            if nll < best:
                best, best_T = nll, T
        self.T = float(best_T)
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float)
        if self._netcal is not None:
            return np.asarray(self._netcal.calibrate(_softmax(logits)), dtype=float)
        return _softmax(logits / self.T)


class VectorScaler:
    name = "vector"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.w = None
        self.b = None
        self._netcal = None

    def _use_netcal(self) -> bool:
        if not self.cfg.use_netcal:
            return False
        try:
            import netcal  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "VectorScaler":
        logits = np.asarray(logits, dtype=float)
        y = np.asarray(y).astype(int)
        if self._use_netcal():
            try:
                from netcal import VectorScaling

                self._netcal = VectorScaling()
                self._netcal.fit(_softmax(logits), y)
                return self
            except Exception:
                self._netcal = None
        n, k = logits.shape
        w = np.ones(k)
        b = np.zeros(k)
        onehot = np.zeros((n, k))
        onehot[np.arange(n), y] = 1.0
        lr, reg, iters = 0.1, 1e-3, 400
        for _ in range(iters):
            ls = logits * w + b
            p = _softmax(ls)
            d = (p - onehot) / n
            gw = (logits * d).mean(axis=0) + reg * w
            gb = d.mean(axis=0)
            w -= lr * gw
            b -= lr * gb
        self.w, self.b = w, b
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float)
        if self._netcal is not None:
            return np.asarray(self._netcal.calibrate(_softmax(logits)), dtype=float)
        return _softmax(logits * self.w + self.b)


class MatrixScaler:
    name = "matrix"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.W = None
        self.b = None
        self._netcal = None

    def _use_netcal(self) -> bool:
        if not self.cfg.use_netcal:
            return False
        try:
            import netcal  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "MatrixScaler":
        logits = np.asarray(logits, dtype=float)
        y = np.asarray(y).astype(int)
        if self._use_netcal():
            try:
                from netcal import LogisticCalibration

                self._netcal = LogisticCalibration()
                self._netcal.fit(_softmax(logits), y)
                return self
            except Exception:
                self._netcal = None
        n, k = logits.shape
        W = np.eye(k)
        b = np.zeros(k)
        onehot = np.zeros((n, k))
        onehot[np.arange(n), y] = 1.0
        lr, reg, iters = 0.1, 1e-3, 400
        for _ in range(iters):
            ls = logits @ W + b
            p = _softmax(ls)
            d = (p - onehot) / n
            gW = (logits.T @ d) + reg * W
            gb = d.mean(axis=0)
            W -= lr * gW
            b -= lr * gb
        self.W, self.b = W, b
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float)
        if self._netcal is not None:
            return np.asarray(self._netcal.calibrate(_softmax(logits)), dtype=float)
        return _softmax(logits @ self.W + self.b)


CALIBRATORS = {
    "temperature": TemperatureScaler,
    "vector": VectorScaler,
    "matrix": MatrixScaler,
}


def build_calibrator(name: str, cfg: Config | None = None):
    if name not in CALIBRATORS:
        raise ValueError(f"未知校准器: {name}")
    return CALIBRATORS[name](cfg or Config())

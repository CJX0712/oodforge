"""OODForge · calib/calibration — 概率校准器（温度/向量/矩阵/等渗）。

作者: 晨星 (CJX0712)
语义: calibrate 返回「越大越可信」的校准概率（每行和为 1）。
后端: scipy L-BFGS 调参；sklearn IsotonicRegression 作等渗后端（不可用时退化为温度缩放）。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _logits_from(probs: np.ndarray, logits: np.ndarray | None) -> np.ndarray:
    if logits is not None:
        return np.asarray(logits, dtype=np.float64)
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TemperatureScaler:
    name = "temperature"

    def __init__(self, seed: int = 42) -> None:
        self.T = 1.0
        self.seed = seed

    def fit(self, probs, y, logits=None) -> TemperatureScaler:
        z = _logits_from(probs, logits)
        y = np.asarray(y, dtype=int)
        n = z.shape[0]

        def nll(T):
            T = float(T[0])
            if T <= 1e-3:
                return 1e6
            p = _softmax(z / T)
            return float(-np.log(p[np.arange(n), y] + 1e-12).mean())

        res = minimize(nll, x0=[1.0], bounds=[(1e-3, 100.0)])
        self.T = float(res.x[0]) if res.success else 1.0
        return self

    def calibrate(self, probs, logits=None) -> np.ndarray:
        z = _logits_from(probs, logits)
        return _softmax(z / self.T)


class VectorScaler:
    name = "vector"

    def __init__(self, seed: int = 42) -> None:
        self.w = None
        self.b = None

    def fit(self, probs, y, logits=None) -> VectorScaler:
        z = _logits_from(probs, logits)
        y = np.asarray(y, dtype=int)
        k = z.shape[1]
        n = z.shape[0]
        theta = np.concatenate([np.ones(k), np.zeros(k)])

        def nll(t):
            w = t[:k]
            b = t[k:]
            p = _softmax(z * w + b)
            return float(-np.log(p[np.arange(n), y] + 1e-12).mean())

        res = minimize(nll, theta, method="L-BFGS-B")
        self.w = res.x[:k]
        self.b = res.x[k:]
        return self

    def calibrate(self, probs, logits=None) -> np.ndarray:
        z = _logits_from(probs, logits)
        return _softmax(z * self.w + self.b)


class MatrixScaler:
    name = "matrix"

    def __init__(self, seed: int = 42) -> None:
        self.W = None
        self.b = None

    def fit(self, probs, y, logits=None) -> MatrixScaler:
        z = _logits_from(probs, logits)
        y = np.asarray(y, dtype=int)
        k = z.shape[1]
        n = z.shape[0]
        theta = np.concatenate([np.eye(k).ravel(), np.zeros(k)])

        def nll(t):
            W = t[: k * k].reshape(k, k)
            b = t[k * k :]
            p = _softmax(z @ W + b)
            return float(-np.log(p[np.arange(n), y] + 1e-12).mean())

        res = minimize(nll, theta, method="L-BFGS-B")
        self.W = res.x[: k * k].reshape(k, k)
        self.b = res.x[k * k :]
        return self

    def calibrate(self, probs, logits=None) -> np.ndarray:
        z = _logits_from(probs, logits)
        return _softmax(z @ self.W + self.b)


class IsotonicCalibrator:
    name = "isotonic"

    def __init__(self, seed: int = 42) -> None:
        self._models = []
        self._use_sk = True

    def fit(self, probs, y, logits=None) -> IsotonicCalibrator:
        probs = np.asarray(probs, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        k = probs.shape[1]
        try:
            from sklearn.isotonic import IsotonicRegression  # type: ignore

            self._use_sk = True
            self._models = []
            for c in range(k):
                ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                ir.fit(probs[:, c], (y == c).astype(float))
                self._models.append(ir)
        except Exception:  # noqa: BLE001 — 等渗不可用 → 退化为温度缩放
            self._use_sk = False
            from .calibration import TemperatureScaler

            self._fallback = TemperatureScaler().fit(probs, y, logits)
        return self

    def calibrate(self, probs, logits=None) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        if not self._use_sk:
            return self._fallback.calibrate(probs, logits)
        out = np.column_stack(
            [m.predict(probs[:, c]) for c, m in enumerate(self._models)]
        )
        out = np.clip(out, 1e-6, 1.0)
        return out / out.sum(axis=1, keepdims=True)


def make_calibrator(method: str):
    method = (method or "temperature").lower()
    if method == "vector":
        return VectorScaler()
    if method == "matrix":
        return MatrixScaler()
    if method == "isotonic":
        return IsotonicCalibrator()
    return TemperatureScaler()

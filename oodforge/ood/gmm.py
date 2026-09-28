"""OODForge · ood/gmm — 高斯混合似然打分器。

作者: 晨星 (CJX0712)
语义: 分数 = -log p(x)（GMM 负对数似然），越大越 OOD。
后端: sklearn.GaussianMixture（SOTA）；不可用时降级为纯 numpy 对角 GMM EM（离线兜底）。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E101DataError, E200FitError
from .base import BaseScorer


def _numpy_gmm_negll(
    X: np.ndarray, n_comp: int, seed: int, n_iter: int = 60
) -> tuple[np.ndarray, callable]:
    """纯 numpy 对角协方差 GMM EM，返回 (拟合后的负对数似然函数)。"""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # k-means++ 风格初始化均值
    means = X[rng.choice(n, size=n_comp, replace=False)]
    covs = np.tile(np.var(X, axis=0) + 1e-6, (n_comp, 1))
    weights = np.full(n_comp, 1.0 / n_comp)

    for _ in range(n_iter):
        # E 步
        diff = X[:, None, :] - means[None, :, :]  # (n, k, d)
        prec = 1.0 / (covs + 1e-9)  # (k, d)
        log_det = np.sum(np.log(covs + 1e-9), axis=1)  # (k,)
        quad = np.einsum("nkd,kd,nkd->nk", diff, prec, diff)  # (n, k)
        log_resp = (
            np.log(weights + 1e-12)[None, :]
            - 0.5 * (log_det[None, :] + quad)
            - 0.5 * d * np.log(2 * np.pi)
        )
        ll = np.logaddexp.reduce(log_resp, axis=1)
        resp = np.exp(log_resp - ll[:, None])
        # M 步
        Nk = resp.sum(axis=0) + 1e-9
        weights = Nk / n
        means = (resp.T @ X) / Nk[:, None]
        for k in range(n_comp):
            diff = X - means[k]
            covs[k] = (resp[:, k, None] * diff * diff).sum(axis=0) / Nk[k] + 1e-6

    def neg_ll(Xq: np.ndarray) -> np.ndarray:
        diff = Xq[:, None, :] - means[None, :, :]
        prec = 1.0 / (covs + 1e-9)
        quad = np.einsum("nkd,kd,nkd->nk", diff, prec, diff)
        log_resp = (
            np.log(weights + 1e-12)[None, :]
            - 0.5 * (log_det[None, :] + quad)
            - 0.5 * d * np.log(2 * np.pi)
        )
        return -np.logaddexp.reduce(log_resp, axis=1)

    return np.full(n, 0.0), neg_ll


class GMMScorer(BaseScorer):
    name = "gmm"

    def __init__(self, n_components: int | None = None, seed: int = 42) -> None:
        super().__init__()
        self.n_components = n_components
        self.seed = seed
        self._use_sklearn = True
        self._sk_model = None
        self._np_negll = None

    def fit(self, X_id, probabilities=None, logits=None) -> GMMScorer:
        X = np.asarray(X_id, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise E101DataError("gmm 需要 2D 且样本数>=2")
        k = self.n_components or max(1, min(X.shape[0] // 5, 5))
        try:
            from sklearn.mixture import GaussianMixture  # type: ignore

            self._sk_model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=1e-3,
                max_iter=100,
                random_state=self.seed,
            )
            self._sk_model.fit(X)
            self._use_sklearn = True
        except Exception:  # noqa: BLE001 — 离线兜底
            self._use_sklearn = False
            _, self._np_negll = _numpy_gmm_negll(X, k, self.seed)
        self._fitted = True
        self._fit_norm(self._raw(X))
        return self

    def _raw(self, X: np.ndarray, probabilities=None, logits=None) -> np.ndarray:
        if self._use_sklearn and self._sk_model is not None:
            return -self._sk_model.score_samples(X)
        if self._np_negll is not None:
            return self._np_negll(X)
        raise E200FitError("gmm 未拟合")

    def score(self, X, probabilities=None, logits=None) -> np.ndarray:
        if not self._fitted:
            raise E200FitError("gmm 未拟合")
        return self._norm(self._raw(np.asarray(X, dtype=np.float64)))

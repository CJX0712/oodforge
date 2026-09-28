"""OODForge · ood/ensemble — 置信度门控 OOD 路由（CGOR）集成融合。

作者: 晨星 (CJX0712)
创新点: 在多个异质 OOD 信号上学习凸组合权重（Optuna TPE 最大化 AUROC），
        得到比任一单信号更稳的融合分数；可作为级联路由的「强信号」。
离线与厚后端: Optuna 不可用时退化为等权平均（仍可用，仅少一次调权）。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E200FitError, E400RouterError
from ..eval.metrics import auroc
from .base import BaseScorer


class EnsembleRouter(BaseScorer):
    name = "ensemble"

    def __init__(self, scorers: list, n_trials: int = 24, seed: int = 42) -> None:
        super().__init__()
        self.scorers = list(scorers)
        self.n_trials = n_trials
        self.seed = seed
        self.weights_: np.ndarray | None = None

    def fit(
        self,
        X_id,
        probs_id=None,
        logits_id=None,
        val_id=None,
        val_id_probs=None,
        val_id_logits=None,
        val_ood=None,
        val_ood_probs=None,
        val_ood_logits=None,
    ) -> EnsembleRouter:
        if val_id is None or val_ood is None:
            raise E400RouterError("ensemble 需要 val_id + val_ood 以调权")
        # 1) 在 ID 训练集上拟合每个子打分器（post-hoc 需基分类器输出）
        for s in self.scorers:
            s.fit(X_id, probabilities=probs_id, logits=logits_id)
        # 2) 在已知 val 上取归一化子分数（val_id / val_ood 各自用对应分类器输出）
        cols_id = [s.score(val_id, val_id_probs, val_id_logits) for s in self.scorers]
        cols_ood = [
            s.score(val_ood, val_ood_probs, val_ood_logits) for s in self.scorers
        ]
        S_id = np.column_stack(cols_id)  # (n_id, m)
        S_ood = np.column_stack(cols_ood)  # (n_ood, m)
        S = np.vstack([S_id, S_ood])
        y = np.concatenate([np.zeros(S_id.shape[0]), np.ones(S_ood.shape[0])])
        self.weights_ = self._tune_weights(S, y)
        # 用 ID 自身分数拟合最终融合的归一化参数
        fused_id = S_id @ self.weights_
        self._fitted = True
        self._fit_norm(fused_id)
        return self

    def _tune_weights(self, S: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = S.shape[1]
        try:
            import optuna  # type: ignore

            def objective(trial):
                raw = [trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(m)]
                w = np.array(raw)
                tot = w.sum()
                if tot <= 0:
                    return 0.0
                w = w / tot
                fused = S @ w
                return float(auroc(y, fused))

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.seed),
            )
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            best = np.array([study.best_params[f"w{i}"] for i in range(m)])
            tot = best.sum()
            return best / tot if tot > 0 else np.full(m, 1.0 / m)
        except Exception:  # noqa: BLE001 — Optuna 不可用 → 等权
            return np.full(m, 1.0 / m)

    def _raw(self, X: np.ndarray, probabilities=None, logits=None) -> np.ndarray:
        if self.weights_ is None:
            raise E200FitError("ensemble 未拟合")
        cols = [
            s.score(X, probabilities=probabilities, logits=logits) for s in self.scorers
        ]
        S = np.column_stack(cols)
        return S @ self.weights_

    def score(self, X, probabilities=None, logits=None) -> np.ndarray:
        if not self._fitted or self.weights_ is None:
            raise E200FitError("ensemble 未拟合")
        return self._norm(
            self._raw(np.asarray(X, dtype=np.float64), probabilities, logits)
        )

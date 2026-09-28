"""data/loaders.py — 真实数据 OOD 基准 (离线, 仅依赖 scikit-learn).

旗艦真实基准: digits 数据集, 训练于类 0..k-1, 以类 k..9 作为**未见类 OOD**
(真正的分布外: 训练时从未见过这些类别)。同 64 维特征空间, 检测器可直接工作。
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.errors import err
from ..core.types import Dataset, OODSplit
from .synthetic import make_synthetic_split


def _split_by_mask(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    fracs: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[Dataset, Dataset, Dataset]:
    Xs = X[mask]
    ys = y[mask]
    idx = rng.permutation(Xs.shape[0])
    Xs, ys = Xs[idx], ys[idx]
    n = Xs.shape[0]
    n_train = int(n * fracs[0])
    n_val = int(n * fracs[1])
    tr = Dataset(X=Xs[:n_train], y=ys[:n_train], name="id_train")
    va = Dataset(
        X=Xs[n_train : n_train + n_val], y=ys[n_train : n_train + n_val], name="id_val"
    )
    te = Dataset(X=Xs[n_train + n_val :], y=ys[n_train + n_val :], name="id_test")
    return tr, va, te


def make_digits_split(cfg: Config, n_id_classes: int = 5, seed: int | None = None) -> OODSplit:
    """digits 未见类 OOD 基准。

    ID = 类 0..n_id_classes-1; OOD = 类 n_id_classes..9。
    """
    try:
        from sklearn.datasets import load_digits
    except Exception as exc:  # pragma: no cover
        raise err("E201", f"sklearn.datasets 不可达: {exc}")
    seed = seed if seed is not None else cfg.random_state
    rng = np.random.default_rng(seed)
    data = load_digits()
    X = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target, dtype=int)
    if n_id_classes < 2 or n_id_classes > 9:
        raise err("E200", "n_id_classes 需在 [2,9]")
    id_mask = y < n_id_classes
    ood_mask = y >= n_id_classes
    if id_mask.sum() < 50 or ood_mask.sum() < 50:
        raise err("E200", "digits 样本不足以划分 ID/OOD")

    id_tr, id_va, id_te = _split_by_mask(X, y, id_mask, (0.6, 0.2, 0.2), rng)
    ood_va, ood_te, _ = _split_by_mask(X, y, ood_mask, (0.5, 0.5, 0.0), rng)
    ood_va = Dataset(X=ood_va.X, y=None, name="ood_val")
    ood_te = Dataset(X=ood_te.X, y=None, name="ood_test")
    return OODSplit(
        id_train=id_tr, id_val=id_va, id_test=id_te, ood_val=ood_va, ood_test=ood_te
    )


def build_split(name: str, cfg: Config) -> OODSplit:
    """按名称构造 OODSplit 的工厂。

    name: 'synthetic_far' | 'synthetic_near' | 'synthetic_mixed' | 'digits'
    """
    if name.startswith("synthetic"):
        kind = name.split("_", 1)[1] if "_" in name else "far"
        return make_synthetic_split(cfg, ood_kind=kind)
    if name == "digits":
        return make_digits_split(cfg)
    raise err("E200", f"未知 split 名称: {name}")

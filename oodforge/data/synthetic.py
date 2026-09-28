"""data/synthetic.py — 合成 ID/OOD 数据生成.

设计要点 (踩坑: 数据泄漏/无难度梯度):
- ID 与 OOD 同特征维度, 但 OOD 来自与 ID 类中心相距很远 (far) 或夹在类间 (near) 的分布,
  保证检测器存在可学习的难度梯度, 而非字面重复 → 避免准确率虚高/路由永不降级。
- 全部固定 random_state, 可复现。
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.errors import err
from ..core.types import Dataset, OODSplit


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _class_centers(
    n_classes: int, n_features: int, rng: np.random.Generator, scale: float = 5.0
) -> np.ndarray:
    """在特征空间放 n_classes 个互相分离的类中心。"""
    raw = rng.standard_normal((n_classes, n_features))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return raw / norms * scale


def _sample_gmm(
    centers: np.ndarray,
    cov_scale: float,
    n_per_class: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_classes, n_features = centers.shape
    X_parts: list[np.ndarray] = []
    y_parts: list[int] = []
    cov = np.eye(n_features) * cov_scale
    for c, n in enumerate(n_per_class):
        Xc = rng.multivariate_normal(centers[c], cov, size=n)
        X_parts.append(Xc)
        y_parts.extend([c] * n)
    X = np.vstack(X_parts)
    y = np.asarray(y_parts, dtype=int)
    # 打乱
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


def _make_id(cfg: Config, seed: int, centers: np.ndarray) -> Dataset:
    rng = _rng(seed)
    per = _balanced_sizes(cfg.id_train_size, cfg.n_classes)
    X, y = _sample_gmm(centers, cov_scale=0.7, n_per_class=per, rng=rng)
    return Dataset(X=X, y=y, name="id_train")


def _make_ood_far(cfg: Config, seed: int, centers: np.ndarray, n: int | None = None) -> Dataset:
    """远 OOD: 取一个远离所有 ID 中心的单高斯。"""
    rng = _rng(seed)
    n_features = cfg.n_features
    n = n or cfg.ood_test_size
    # 远离方向: 类中心法向方向的加权和 → 偏离 ID 支撑
    direction = rng.standard_normal(n_features)
    direction /= np.linalg.norm(direction) + 1e-12
    far_center = direction * 16.0
    cov = np.eye(n_features) * 2.5
    X = rng.multivariate_normal(far_center, cov, size=n)
    return Dataset(X=X, y=None, name="ood_far")


def _make_ood_near(
    cfg: Config, seed: int, centers: np.ndarray, n: int | None = None
) -> Dataset:
    """近 OOD: 夹在 ID 类中心之间的模糊混合。"""
    rng = _rng(seed)
    n = n or cfg.ood_test_size
    mid = centers.mean(axis=0)
    spread = centers.std(axis=0) * 0.6
    X_parts = []
    for _ in range(n):
        base = mid + rng.standard_normal(cfg.n_features) * spread
        X_parts.append(base)
    X = np.asarray(X_parts)
    return Dataset(X=X, y=None, name="ood_near")


def _balanced_sizes(total: int, n_classes: int) -> list[int]:
    base, rem = divmod(total, n_classes)
    return [base + (1 if i < rem else 0) for i in range(n_classes)]


def make_synthetic_split(cfg: Config, ood_kind: str = "far") -> OODSplit:
    """构造合成 OODSplit。

    ood_kind: 'far' | 'near' | 'mixed'。
    """
    if cfg.n_classes < 2:
        raise err("E200", "n_classes 必须 >= 2")
    rng = _rng(cfg.random_state)
    centers = _class_centers(cfg.n_classes, cfg.n_features, rng)

    id_train = _make_id(cfg, cfg.random_state + 1, centers)
    id_val = Dataset(
        *_split_pair(
            _sample_gmm(
                centers,
                0.7,
                _balanced_sizes(cfg.id_val_size, cfg.n_classes),
                _rng(cfg.random_state + 2),
            ),
            "id_val",
        )
    )
    id_test = Dataset(
        *_split_pair(
            _sample_gmm(
                centers,
                0.7,
                _balanced_sizes(cfg.id_test_size, cfg.n_classes),
                _rng(cfg.random_state + 3),
            ),
            "id_test",
        )
    )

    if ood_kind == "far":
        ood_val = _make_ood_far(cfg, cfg.random_state + 4, centers, cfg.ood_val_size)
        ood_test = _make_ood_far(cfg, cfg.random_state + 5, centers, cfg.ood_test_size)
    elif ood_kind == "near":
        ood_val = _make_ood_near(cfg, cfg.random_state + 4, centers, cfg.ood_val_size)
        ood_test = _make_ood_near(cfg, cfg.random_state + 5, centers, cfg.ood_test_size)
    elif ood_kind == "mixed":
        ood_val = _mix(cfg, cfg.random_state + 4, centers, cfg.ood_val_size)
        ood_test = _mix(cfg, cfg.random_state + 5, centers, cfg.ood_test_size)
    else:
        raise err("E200", f"未知 ood_kind={ood_kind}")
    return OODSplit(
        id_train=id_train, id_val=id_val, id_test=id_test, ood_val=ood_val, ood_test=ood_test
    )


def _mix(cfg: Config, seed: int, centers: np.ndarray, n: int | None = None) -> Dataset:
    n = n or cfg.ood_test_size
    half = n // 2
    far = _make_ood_far(cfg, seed, centers, half)
    near = _make_ood_near(cfg, seed + 100, centers, n - half)
    X = np.vstack([far.X, near.X])
    return Dataset(X=X, y=None, name="ood_mixed")


def _split_pair(
    xy: tuple[np.ndarray, np.ndarray], name: str
) -> tuple[np.ndarray, np.ndarray, str]:
    X, y = xy
    return X, y, name

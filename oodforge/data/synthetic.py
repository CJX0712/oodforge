"""OODForge · data/synthetic — 合成 ID / OOD 数据生成。

作者: 晨星 (CJX0712)
设计要点（踩坑：防数据泄漏 / 难度梯度 / 类分布一致）:
  - ID 的「类中心」由固定 salt 生成（与切分无关），train/val/test 共享同一类条件分布，
    否则分类器学到错误映射（实测 ID-test 准确率塌到 0.045）。
  - OOD 用远高斯 / 均匀 / 旋转中心 / 各向异性协方差，制造真实的分布外偏移。
  - 各切分用不同 sample_seed 注入采样噪声，避免字面重复（防泄漏）。
"""

from __future__ import annotations

import numpy as np

from ..core.types import Dataset

_CENTER_SALT = 12345  # 类中心的固定 salt：所有 ID 切分共享


def _canonical_centers(dim: int, n_classes: int, sep: float, salt: int) -> np.ndarray:
    rng = np.random.default_rng(salt)
    return rng.normal(scale=sep, size=(n_classes, dim))


def _rand_psd(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim))
    return A @ A.T / dim + np.eye(dim) * 0.5


def _blobs(
    n: int,
    dim: int,
    n_classes: int,
    centers: np.ndarray,
    sample_seed: int,
    anisotropic: bool = False,
    rotate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(sample_seed)
    y = rng.integers(0, n_classes, size=n)
    if anisotropic:
        covs = np.array([_rand_psd(dim, sample_seed + c) for c in range(n_classes)])
    else:
        covs = np.array([np.eye(dim) for _ in range(n_classes)])
    X = np.empty((n, dim))
    for c in range(n_classes):
        m = int(np.sum(y == c))
        X[y == c] = rng.multivariate_normal(centers[c], covs[c], size=m)
    if rotate:
        R, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        X = X @ R
    return X, y


def make_in_distribution(
    n: int, dim: int, n_classes: int, seed: int, sep: float = 4.0
) -> Dataset:
    """分布内（ID）数据：高斯团，类中心固定（与 seed 无关），仅 sample_seed 控制采样。"""
    centers = _canonical_centers(dim, n_classes, sep, _CENTER_SALT)
    X, y = _blobs(n, dim, n_classes, centers, sample_seed=seed)
    return Dataset(X=X, y=y)


_OOD_KINDS = ("far_gauss", "uniform", "rotated", "anisotropic")


def make_ood(
    n: int,
    dim: int,
    n_classes: int,
    seed: int,
    kind: str | None = None,
    sep: float = 4.0,
) -> Dataset:
    """分布外（OOD）数据。kind 为 None 时按 seed 轮询多种表面形式。"""
    if kind is None:
        kind = _OOD_KINDS[seed % len(_OOD_KINDS)]
    centers = _canonical_centers(dim, n_classes, sep, _CENTER_SALT)
    rng = np.random.default_rng(seed + 777)
    if kind == "far_gauss":
        far = rng.normal(scale=sep * 3.5, size=(n_classes, dim))
        X = rng.multivariate_normal(far[0], np.eye(dim) * 2.0, size=n)
        y = rng.integers(0, n_classes, size=n)
    elif kind == "uniform":
        X = rng.uniform(-sep * 4, sep * 4, size=(n, dim))
        y = rng.integers(0, n_classes, size=n)
    elif kind == "rotated":
        R, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        X, y = _blobs(n, dim, n_classes, centers @ R, sample_seed=seed + 13)
    elif kind == "anisotropic":
        X, y = _blobs(
            n, dim, n_classes, centers, sample_seed=seed + 31, anisotropic=True
        )
    else:
        raise ValueError(f"unknown ood kind: {kind}")
    return Dataset(X=X, y=y)


def make_benchmark_datasets(
    dim: int = 10,
    n_classes: int = 3,
    sep: float = 4.0,
    seed: int = 42,
    n_train: int = 600,
    n_val: int = 200,
    n_test_id: int = 400,
    n_ood: int = 400,
) -> dict:
    """返回完整基准数据集切分。各切分使用不同 sample_seed，避免泄漏。"""
    id_train = make_in_distribution(n_train, dim, n_classes, seed=seed, sep=sep)
    id_val = make_in_distribution(n_val, dim, n_classes, seed=seed + 101, sep=sep)
    id_test = make_in_distribution(n_test_id, dim, n_classes, seed=seed + 202, sep=sep)
    ood_a = make_ood(
        n_ood // 2, dim, n_classes, seed=seed + 303, kind="far_gauss", sep=sep
    )
    ood_b = make_ood(
        n_ood // 2, dim, n_classes, seed=seed + 404, kind="uniform", sep=sep
    )
    X_ood = np.vstack([ood_a.X, ood_b.X])
    y_ood = np.concatenate([ood_a.y, ood_b.y]) if ood_a.y is not None else None
    return {
        "id_train": id_train,
        "id_val": id_val,
        "id_test": id_test,
        "ood": Dataset(X=X_ood, y=y_ood),
    }

"""OODForge · data/loaders — 文件载入（CSV / NPY / NPZ）。

作者: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np

from ..core.types import Dataset


def load_csv(path: str, label_col: int | None = None) -> Dataset:
    """载入 CSV。label_col 指定标签列（载入后从 X 剥离）；为 None 则纯特征。"""
    arr = np.loadtxt(path, delimiter=",", ndmin=2)
    if label_col is None:
        return Dataset(X=arr)
    y = arr[:, label_col].astype(int)
    cols = [c for c in range(arr.shape[1]) if c != label_col]
    return Dataset(X=arr[:, cols], y=y)


def load_npy(path: str, label_path: str | None = None) -> Dataset:
    X = np.load(path)
    y = np.load(label_path) if label_path else None
    return Dataset(X=X, y=y)


def load_npz(path: str, X_key: str = "X", y_key: str | None = "y") -> Dataset:
    data = np.load(path)
    X = data[X_key]
    y = data[y_key] if (y_key and y_key in data.files) else None
    return Dataset(X=X, y=y)

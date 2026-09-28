"""data — 数据生成与加载层."""

from .loaders import build_split, make_digits_split
from .synthetic import make_synthetic_split

__all__ = ["make_synthetic_split", "make_digits_split", "build_split"]

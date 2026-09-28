"""core/config.py — 全局配置 + ENV_* 覆盖.

支持通过环境变量覆盖任意字段, 形如 OODFORGE_RANDOM_STATE=42。
键名映射: 环境变量去掉前缀 OODFORGE_ 后转小写, 用 __ 表示嵌套 (本系统为扁平)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any

_ENV_PREFIX = "OODFORGE_"


@dataclass
class Config:
    """系统级配置 (扁平字段, 全可经 ENV 覆盖)."""

    random_state: int = 42
    n_classes: int = 3
    n_features: int = 12
    id_train_size: int = 600
    id_val_size: int = 200
    id_test_size: int = 400
    ood_val_size: int = 200
    ood_test_size: int = 400
    knn_k: int = 10
    mahalanobis_reg: float = 1e-3
    fpr_target: float = 0.05  # FPR@(1-0.05)TPR = FPR@95TPR
    ece_bins: int = 15
    n_bootstrap: int = 5
    use_netcal: bool = True  # 优先用 netcal 做 scaling 校准
    use_torch: bool = True  # 可选 ODIN 后端
    verbose: bool = False

    def override_from_env(self) -> "Config":
        """读取 OODFORGE_* 环境变量覆盖字段, 返回新实例 (不修改自身)."""
        kv: dict[str, Any] = {}
        for f in fields(self):
            env_key = f"{_ENV_PREFIX}{f.name.upper()}"
            if env_key in os.environ:
                raw = os.environ[env_key]
                kv[f.name] = _coerce(raw, f.type)
        if kv:
            return dataclass_replace(self, **kv)
        return self

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _coerce(raw: str, typ: Any) -> Any:  # noqa: ANN401
    raw = raw.strip()
    if typ is int or typ == "int":
        return int(raw)
    if typ is float or typ == "float":
        return float(raw)
    if typ is bool or typ == "bool":
        return raw.lower() in ("1", "true", "yes", "on")
    return raw


def dataclass_replace(cfg: Config, **changes: Any) -> Config:  # noqa: ANN401
    vals = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    vals.update(changes)
    return Config(**vals)


def load_config(**overrides: Any) -> Config:  # noqa: ANN401
    """构造配置: 默认值 → ENV 覆盖 → 显式覆盖."""
    cfg = Config().override_from_env()
    if overrides:
        cfg = dataclass_replace(cfg, **overrides)
    return cfg

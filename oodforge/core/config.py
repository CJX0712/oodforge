"""OODForge · core/config — 配置加载，支持 ENV_OODFORGE_* 覆盖。

作者: 晨星 (CJX0712)
优先级: 显式传入 PipelineConfig > 环境变量 ENV_OODFORGE_<KEY> > 默认值。
"""

from __future__ import annotations

import os

from .types import PipelineConfig

_PREFIX = "OODFORGE_"


def _coerce(key: str, raw: str) -> object:
    key = key.lower()
    if key in ("random_state", "n_classes", "ensemble_trials"):
        return int(raw)
    if key in ("target_coverage",):
        return float(raw)
    if key in ("use_optional_backends",):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return raw


def from_env(base: PipelineConfig | None = None) -> PipelineConfig:
    """用环境变量覆盖 base（或默认）配置。"""
    cfg = base or PipelineConfig()
    overrides: dict[str, object] = {}
    for raw_key, val in os.environ.items():
        if raw_key.startswith(_PREFIX):
            field_name = raw_key[len(_PREFIX) :].lower()
            if hasattr(cfg, field_name):
                overrides[field_name] = _coerce(field_name, val)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def resolve(base: PipelineConfig | None = None) -> PipelineConfig:
    """对外统一入口：合并环境变量后的最终配置。"""
    return from_env(base)

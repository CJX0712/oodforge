"""core/errors.py — 统一错误码与异常类型.

错误码分段:
  E1xx config   E2xx data   E3xx train   E4xx detector   E5xx ensemble/router
"""

from __future__ import annotations

from typing import Dict

_ERROR_REGISTRY: Dict[str, str] = {
    "E100": "配置加载失败: 字段缺失或类型错误",
    "E101": "配置覆盖失败: 环境变量 ENV_* 解析错误",
    "E200": "数据生成失败: 维度或样本数非法",
    "E201": "数据加载失败: 远程/IO 不可达且无机内兜底",
    "E300": "分类器训练失败: 输入含 NaN/Inf 或标签非法",
    "E301": "分类器推理失败: 特征维度与训练不一致",
    "E400": "检测器失败: 输入为空或后端不可用且无兜底",
    "E401": "检测器后端不可用: 可选依赖缺失",
    "E500": "路由失败: 验证集缺失或所有检测器不可用",
    "E501": "阈值计算失败: 验证 OOD 标签缺失",
}


class OODForgeError(Exception):
    """OODForge 统一异常基类."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        base = _ERROR_REGISTRY.get(code, "未知错误")
        msg = f"[{code}] {base}"
        if detail:
            msg = f"{msg} — {detail}"
        super().__init__(msg)


def err(code: str, detail: str = "") -> OODForgeError:
    """构造统一异常 (便于一处 raise)."""
    return OODForgeError(code, detail)

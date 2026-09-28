"""OODForge · core/errors — 错误码与异常（E100~E500）。

作者: 晨星 (CJX0712)
惯例: 每个模块的异常继承 OODForgeError，并带稳定错误码，便于调用方按码处理。
"""

from __future__ import annotations


class OODForgeError(Exception):
    """所有 OODForge 异常的基类。"""

    code: str = "E000"
    title: str = "OODForge 内部错误"

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(f"[{self.code}] {self.title}: {message}")


class E100ConfigError(OODForgeError):
    code = "E100"
    title = "配置错误"


class E101DataError(OODForgeError):
    code = "E101"
    title = "数据错误"


class E200FitError(OODForgeError):
    code = "E200"
    title = "拟合失败"


class E201ScoreError(OODForgeError):
    code = "E201"
    title = "打分失败"


class E300CalibError(OODForgeError):
    code = "E300"
    title = "校准失败"


class E400RouterError(OODForgeError):
    code = "E400"
    title = "路由失败"


class E500BackendUnavailable(OODForgeError):
    code = "E500"
    title = "可选后端不可用"


_ERROR_REGISTRY = {
    cls.code: cls
    for cls in (
        OODForgeError,
        E100ConfigError,
        E101DataError,
        E200FitError,
        E201ScoreError,
        E300CalibError,
        E400RouterError,
        E500BackendUnavailable,
    )
}


def lookup(code: str) -> type[OODForgeError]:
    return _ERROR_REGISTRY.get(code, OODForgeError)

"""tests/test_router.py — CCOR 路由器选型/阈值/降级。"""

import numpy as np

from oodforge.core.config import Config
from oodforge.data.synthetic import make_synthetic_split
from oodforge.ensemble.router import CCORRouter
from oodforge.train.classifiers import build_classifier


def _setup(kind="far"):
    cfg = Config(
        random_state=42,
        n_classes=3,
        n_features=10,
        id_train_size=300,
        id_val_size=100,
        id_test_size=150,
        ood_val_size=100,
        ood_test_size=150,
    )
    split = make_synthetic_split(cfg, ood_kind=kind)
    clf = build_classifier("auto", cfg)
    clf.fit(split.id_train.X, split.id_train.y)
    return cfg, split, clf


def test_router_selects_and_thresholds():
    cfg, split, clf = _setup()
    router = CCORRouter(cfg)
    router.fit(split, clf)
    rep = router.report()
    assert rep.val_auroc_of_selected > 0.7  # far OOD 易检测
    assert rep.selected_detector  # 非空
    # 阈值在验证 OOD 的 5% 分位 → ID 误报率低
    id_flags = router.flag(split.id_val.X)
    assert id_flags.mean() <= 0.08  # FPR@95TPR 设计目标 ≤5%, 留余量


def test_router_flag_orients():
    cfg, split, clf = _setup()
    router = CCORRouter(cfg)
    router.fit(split, clf)
    ood_flags = router.flag(split.ood_test.X)
    id_flags = router.flag(split.id_test.X)
    assert ood_flags.mean() > id_flags.mean()


def test_router_fallback_ok_when_only_msp(monkeypatch):
    cfg, split, clf = _setup()
    import oodforge.ensemble.router as rtmod

    orig_build = rtmod.build_detector

    def fake_build(name, cfg=None):
        if name in ("energy", "tsmsp", "mahalanobis", "knn", "ocsvm"):
            raise RuntimeError("forced unavailable")
        return orig_build(name, cfg)

    monkeypatch.setattr(rtmod, "build_detector", fake_build)
    router = CCORRouter(cfg)
    router.fit(split, clf)
    # 只剩 msp 与 calibrated_msp 可用 → 应正常产出
    assert router.report().selected_detector in ("msp", "calibrated_msp")

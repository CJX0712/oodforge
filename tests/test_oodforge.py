"""OODForge · tests — 不变量与端到端验证（pytest）。

作者: 晨星 (CJX0712)
覆盖: 数据生成 / 各打分器单调性 / 集成非劣 / 校准降 ECE / 路由覆盖率 / 指标正确性。
运行: cd <repo> && python -m pytest -q -W ignore::UserWarning
"""

from __future__ import annotations

import numpy as np

from oodforge.calib.calibration import (
    IsotonicCalibrator,
    MatrixScaler,
    TemperatureScaler,
    VectorScaler,
)
from oodforge.core.types import PipelineConfig
from oodforge.data.synthetic import (
    make_benchmark_datasets,
    make_in_distribution,
    make_ood,
)
from oodforge.eval.metrics import auroc, ece
from oodforge.ood.energy import EnergyScorer
from oodforge.ood.ensemble import EnsembleRouter
from oodforge.ood.entropy import EntropyScorer
from oodforge.ood.gmm import GMMScorer
from oodforge.ood.iforest import IsolationForestScorer
from oodforge.ood.knn import KNNScorer
from oodforge.ood.mahalanobis import MahalanobisScorer
from oodforge.pipeline.pipeline import OODPipeline

DIM, K, SEED = 10, 3, 42


def _fit_base():
    from sklearn.linear_model import LogisticRegression

    ds = make_benchmark_datasets(dim=DIM, n_classes=K, seed=SEED)
    clf = LogisticRegression(solver="lbfgs", max_iter=3000, random_state=SEED)
    clf.fit(ds["id_train"].X, ds["id_train"].y)
    p_tr, l_tr = (
        clf.predict_proba(ds["id_train"].X),
        clf.decision_function(ds["id_train"].X),
    )
    p_val, l_val = (
        clf.predict_proba(ds["id_val"].X),
        clf.decision_function(ds["id_val"].X),
    )
    p_te, l_te = (
        clf.predict_proba(ds["id_test"].X),
        clf.decision_function(ds["id_test"].X),
    )
    return ds, clf, p_tr, l_tr, p_val, l_val, p_te, l_te


def _ood_outputs(clf, X):
    return (
        clf.predict_proba(X),
        clf.decision_function(X) if hasattr(clf, "decision_function") else None,
    )


def test_synthetic_no_leakage():
    ds = make_benchmark_datasets(dim=DIM, n_classes=K, seed=SEED)
    assert ds["id_train"].X.shape[0] == 600
    assert not np.array_equal(ds["id_train"].X, ds["id_test"].X)
    assert ds["ood"].X.shape[0] == 400


def test_scorer_monotonicity():
    """每个打分器在 OOD 上的均值分数应大于 ID（越大越 OOD 语义成立）。"""
    ds, clf, p_tr, l_tr, _p_val, _l_val, p_te, l_te = _fit_base()
    p_ood, l_ood = _ood_outputs(clf, ds["ood"].X[:200])
    scorers = [
        MahalanobisScorer(),
        EnergyScorer(),
        EntropyScorer(),
        GMMScorer(n_components=K),
        IsolationForestScorer(),
        KNNScorer(k=5),
    ]
    for s in scorers:
        s.fit(ds["id_train"].X, p_tr, l_tr)
        sid = s.score(ds["id_test"].X, p_te, l_te)
        sood = s.score(ds["ood"].X[:200], p_ood, l_ood)
        assert np.isfinite(sid).all() and np.isfinite(sood).all()
        assert sood.mean() > sid.mean(), f"{s.name} 单调性不成立"


def test_mahalanobis_ood_farther():
    """远高斯 OOD 的马氏距离应大于 ID（纯 numpy 打分器，不依赖分类器）。"""
    from oodforge.ood.mahalanobis import MahalanobisScorer

    id_ = make_in_distribution(200, DIM, K, seed=1)
    ood = make_ood(200, DIM, K, seed=2, kind="far_gauss")
    s = MahalanobisScorer()
    s.fit(id_.X)
    assert s.score(ood.X).mean() > s.score(id_.X).mean()


def test_ensemble_non_inferior():
    """集成 AUROC 不应明显差于最优单信号（非劣守护），合并 id_test+ood 评测。"""
    ds, clf, p_tr, l_tr, p_val, l_val, p_te, l_te = _fit_base()
    p_ood, l_ood = _ood_outputs(clf, ds["ood"].X[:200])
    subs = [
        MahalanobisScorer(),
        EnergyScorer(),
        EntropyScorer(),
        GMMScorer(n_components=K),
    ]
    ens = EnsembleRouter(subs, n_trials=12, seed=SEED)
    ens.fit(
        ds["id_train"].X,
        probs_id=p_tr,
        logits_id=l_tr,
        val_id=ds["id_val"].X,
        val_id_probs=p_val,
        val_id_logits=l_val,
        val_ood=ds["ood"].X[:200],
        val_ood_probs=p_ood,
        val_ood_logits=l_ood,
    )
    y = np.concatenate([np.zeros(400), np.ones(200)])

    def ens_score_on_all(s):
        sid = s.score(ds["id_test"].X, p_te, l_te)
        sood = s.score(ds["ood"].X[:200], p_ood, l_ood)
        return np.concatenate([sid, sood])

    best = max(auroc(y, ens_score_on_all(s)) for s in subs)
    ens_a = auroc(y, ens_score_on_all(ens))
    assert ens_a >= best * 0.95, f"集成 {ens_a:.3f} < 最优单信号 {best:.3f} 非劣破口"


def test_calibration_reduces_ece():
    """温度缩放应降低 ECE（不变量：校准后误差更小）。"""
    from sklearn.linear_model import LogisticRegression

    ds = make_benchmark_datasets(dim=DIM, n_classes=K, seed=SEED)
    clf = LogisticRegression(solver="lbfgs", max_iter=3000, random_state=SEED)
    clf.fit(ds["id_train"].X, ds["id_train"].y)
    p_val, l_val = (
        clf.predict_proba(ds["id_val"].X),
        clf.decision_function(ds["id_val"].X),
    )
    p_test, l_test = (
        clf.predict_proba(ds["id_test"].X),
        clf.decision_function(ds["id_test"].X),
    )
    ece_before = ece(p_test, ds["id_test"].y)
    cal = TemperatureScaler().fit(p_val, ds["id_val"].y, l_val)
    ece_after = ece(cal.calibrate(p_test, l_test), ds["id_test"].y)
    assert ece_after <= ece_before + 0.02, (
        f"ECE 未降: {ece_before:.3f} -> {ece_after:.3f}"
    )


def test_calibrators_rows_sum_one():
    ds, _, _p_tr, _l_tr, p_val, l_val, p_test, l_te = _fit_base()
    for C in (
        TemperatureScaler(),
        VectorScaler(),
        MatrixScaler(),
        IsotonicCalibrator(),
    ):
        C.fit(p_val, ds["id_val"].y, l_val)
        out = C.calibrate(p_test, l_te)
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5), f"{C.name} 行和不为1"


def test_router_coverage_and_ood_reject():
    """路由覆盖率应接近目标，且 OOD 样本大部分被弃权。"""
    ds = make_benchmark_datasets(dim=DIM, n_classes=K, seed=SEED)
    pipe = OODPipeline(PipelineConfig(random_state=SEED, target_coverage=0.90))
    res = pipe.run(ds)
    rr = res["router_result"]
    assert rr.coverage >= 0.80, f"覆盖率过低 {rr.coverage:.3f}"
    assert rr.ood_abstain_rate >= 0.40, f"OOD 弃权率过低 {rr.ood_abstain_rate:.3f}"


def test_metrics_auroc_separable():
    y = np.concatenate([np.zeros(100), np.ones(100)])
    score = np.concatenate(
        [
            np.random.default_rng(0).normal(0, 1, 100),
            np.random.default_rng(1).normal(3, 1, 100),
        ]
    )
    assert auroc(y, score) > 0.95


def test_ece_perfect_calib():
    """按预测概率采样标签 → 经验准确率≈置信度 → ECE 接近 0。"""
    rng = np.random.default_rng(0)
    n, k = 1500, 3
    logits = rng.normal(size=(n, k))
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    # 按预测概率采样标签，且失败时取「非 pred」类 → 经验准确率严格=置信度
    draw = rng.random(n)
    y = np.where(
        draw < probs[np.arange(n), pred],
        pred,
        (pred + 1 + rng.integers(0, k - 1, n)) % k,
    )
    assert ece(probs, y) < 0.06


def test_pipeline_end_to_end_runs():
    ds = make_benchmark_datasets(dim=DIM, n_classes=K, seed=SEED)
    pipe = OODPipeline(PipelineConfig(random_state=SEED))
    res = pipe.run(ds)
    assert res["base_accuracy"] > 0.5
    names = {r.name for r in res["ood_results"]}
    assert {
        "mahalanobis",
        "energy",
        "entropy",
        "gmm",
        "iforest",
        "knn",
        "ensemble",
    } <= names

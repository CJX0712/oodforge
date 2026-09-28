"""OODForge · pipeline — 端到端 OOD 检测 + 校准 + 选择性路由。

作者: 晨星 (CJX0712)
链路（单向无环）: cli -> pipeline -> {data, ood, calib, router, eval} -> core
基分类器: sklearn LogisticRegression(lbfgs, 提供 logits) 或 RandomForest(由 probs 派生 logits)。
"""

from __future__ import annotations

import time

import numpy as np

from ..calib.calibration import make_calibrator
from ..core.config import resolve
from ..core.errors import E101DataError
from ..core.types import (
    BenchmarkRecord,
    CalibResult,
    OODResult,
    PipelineConfig,
)
from ..eval.metrics import auprc, auroc, ece, fpr_at_tpr
from ..ood.energy import EnergyScorer
from ..ood.ensemble import EnsembleRouter
from ..ood.entropy import EntropyScorer
from ..ood.gmm import GMMScorer
from ..ood.iforest import IsolationForestScorer
from ..ood.knn import KNNScorer
from ..ood.mahalanobis import MahalanobisScorer
from ..router.selective import ConfidenceGatedRouter


def _make_base_classifier(kind: str, seed: int):
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=seed)
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(solver="lbfgs", max_iter=3000, random_state=seed)


class OODPipeline:
    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg = resolve(cfg)
        self._fit_state: dict = {}

    # ---- 基分类器输出 ----
    @staticmethod
    def _outputs(clf, X):
        probs = clf.predict_proba(X)
        logits = clf.decision_function(X) if hasattr(clf, "decision_function") else None
        return probs, logits

    def run(self, datasets: dict) -> dict:
        cfg = self.cfg
        id_train = datasets["id_train"]
        id_val = datasets["id_val"]
        id_test = datasets["id_test"]
        ood = datasets["ood"]
        if id_train.y is None or id_test.y is None:
            raise E101DataError("OOD 评测需要 ID 标签")

        # 切分 OOD：前半调权，后半评测（防泄漏）
        n = ood.X.shape[0]
        ood_val_X, ood_test_X = ood.X[: n // 2], ood.X[n // 2 :]

        # 1) 基分类器
        clf = _make_base_classifier(cfg.base_classifier, cfg.random_state)
        clf.fit(id_train.X, id_train.y)
        acc_id = float((clf.predict(id_test.X) == id_test.y).mean())
        p_tr, l_tr = self._outputs(clf, id_train.X)
        p_val, l_val = self._outputs(clf, id_val.X)
        p_test, l_test = self._outputs(clf, id_test.X)
        # OOD 样本同样需基分类器输出（post-hoc OOD）
        p_ood_val, l_ood_val = self._outputs(clf, ood_val_X)
        p_ood_te, l_ood_te = self._outputs(clf, ood_test_X)

        # 2) 单打分器评测（ID vs OOD，统一语义分数越大越 OOD）
        scorers = [
            MahalanobisScorer(),
            EnergyScorer(),
            EntropyScorer(),
            GMMScorer(n_components=cfg.n_classes),
            IsolationForestScorer(),
            KNNScorer(k=5),
        ]
        for s in scorers:
            s.fit(id_train.X, probabilities=p_tr, logits=l_tr)

        def make_ood_result(name, s_id, s_ood):
            y = np.concatenate([np.zeros(len(s_id)), np.ones(len(s_ood))])
            sc = np.concatenate([s_id, s_ood])
            return OODResult(
                name=name,
                auroc=auroc(y, sc),
                auprc=auprc(y, sc),
                fpr95=fpr_at_tpr(y, sc),
                mean_score_id=float(s_id.mean()),
                mean_score_ood=float(s_ood.mean()),
            )

        ood_results = [
            make_ood_result(
                s.name,
                s.score(id_test.X, p_test, l_test),
                s.score(ood_test_X, p_ood_te, l_ood_te),
            )
            for s in scorers
        ]

        # 3) 集成路由（CGOR 强信号）
        ensemble = EnsembleRouter(
            scorers, n_trials=cfg.ensemble_trials, seed=cfg.random_state
        )
        ensemble.fit(
            id_train.X,
            probs_id=p_tr,
            logits_id=l_tr,
            val_id=id_val.X,
            val_id_probs=p_val,
            val_id_logits=l_val,
            val_ood=ood_val_X,
            val_ood_probs=p_ood_val,
            val_ood_logits=l_ood_val,
        )
        ens_id = ensemble.score(id_test.X, p_test, l_test)
        ens_ood = ensemble.score(ood_test_X, p_ood_te, l_ood_te)
        ood_results.append(make_ood_result("ensemble", ens_id, ens_ood))

        # 4) 校准
        calibrator = make_calibrator(cfg.calib_method)
        calibrator.fit(p_val, id_val.y, l_val)
        ece_before = ece(p_test, id_test.y, n_bins=10)
        p_test_cal = calibrator.calibrate(p_test, l_test)
        ece_after = ece(p_test_cal, id_test.y, n_bins=10)
        calib_result = CalibResult(
            method=cfg.calib_method,
            n_bins=10,
            ece_before=ece_before,
            ece_after=ece_after,
        )

        # 5) 选择性路由（CGOR 级联）
        router = ConfidenceGatedRouter(target_coverage=cfg.target_coverage)
        ood_val_scores = ensemble.score(id_val.X, p_val, l_val)
        calib_val = calibrator.calibrate(p_val, l_val)
        router.fit(calib_val, ood_val_scores, id_val.y)
        ood_test_scores = ensemble.score(ood_test_X, p_ood_te, l_ood_te)
        calib_ood_te = calibrator.calibrate(p_ood_te, l_ood_te)
        router_result = router.evaluate(
            calib_val,
            ood_val_scores,
            id_val.y,
            calib_ood_te,
            ood_test_scores,
        )

        # 6) 计时（per-request 口径，避免批量摊薄掩盖级联）
        t0 = time.perf_counter()
        for i in range(ood_test_X.shape[0]):
            ensemble.score(
                ood_test_X[i : i + 1], p_ood_te[i : i + 1], l_ood_te[i : i + 1]
            )
        per_req_ms = (time.perf_counter() - t0) / ood_test_X.shape[0] * 1000.0

        self._fit_state = {
            "clf": clf,
            "scorers": scorers,
            "ensemble": ensemble,
            "calibrator": calibrator,
            "router": router,
            "ood_test_scores": ood_test_scores,
        }

        records = [
            BenchmarkRecord(
                "base_classifier", "id_test_accuracy", acc_id, cfg.base_classifier
            ),
            BenchmarkRecord(
                "calibration", f"ece_before({cfg.calib_method})", ece_before, ""
            ),
            BenchmarkRecord(
                "calibration", f"ece_after({cfg.calib_method})", ece_after, ""
            ),
            BenchmarkRecord(
                "router",
                "coverage",
                router_result.coverage,
                f"target={cfg.target_coverage}",
            ),
            BenchmarkRecord(
                "router", "selective_risk", router_result.selective_risk, "ID accepted"
            ),
            BenchmarkRecord(
                "router",
                "ood_abstain_rate",
                router_result.ood_abstain_rate,
                "OOD rejected",
            ),
            BenchmarkRecord(
                "router", "per_request_latency_ms", per_req_ms, "ensemble score"
            ),
        ]
        for r in ood_results:
            records.append(BenchmarkRecord("ood", f"auroc({r.name})", r.auroc, ""))
            records.append(BenchmarkRecord("ood", f"auprc({r.name})", r.auprc, ""))
            records.append(BenchmarkRecord("ood", f"fpr95({r.name})", r.fpr95, ""))

        return {
            "config": cfg.to_dict(),
            "base_accuracy": acc_id,
            "ood_results": ood_results,
            "calib_result": calib_result,
            "router_result": router_result,
            "per_request_latency_ms": per_req_ms,
            "records": records,
        }

    def benchmark(self, datasets: dict) -> dict:
        """运行并打印基准表，返回结果与记录。"""
        res = self.run(datasets)
        self._print(res)
        return res

    @staticmethod
    def _print(res: dict) -> None:
        print("\n=== OODForge Benchmark ===")
        print(f"Base ID-test accuracy : {res['base_accuracy']:.4f}")
        print(f"{'scorer':<12}{'AUROC':>8}{'AUPRC':>8}{'FPR95':>8}")
        for r in res["ood_results"]:
            print(f"{r.name:<12}{r.auroc:>8.4f}{r.auprc:>8.4f}{r.fpr95:>8.4f}")
        cr = res["calib_result"]
        print(
            f"\nCalibration [{cr.method}] ECE: {cr.ece_before:.4f} -> {cr.ece_after:.4f}"
        )
        rr = res["router_result"]
        print(
            f"Router coverage={rr.coverage:.3f} selective_risk={rr.selective_risk:.4f} "
            f"id_abstain={rr.abstain_rate:.3f} ood_abstain={rr.ood_abstain_rate:.3f}"
        )
        print(
            f"Per-request latency (ensemble): {res['per_request_latency_ms']:.3f} ms\n"
        )

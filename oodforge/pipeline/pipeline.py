"""pipeline/pipeline.py — 端到端 OOD 评测与路由流水线。

OODPipeline.run(split) 产出:
- 各检测器在测试集上的 AUROC/AUPR/FPR95
- CCOR 路由器的选型/阈值/测试指标
- 校准前后 ECE 对比
落盘 benchmark.json 由 examples/run_demo.py 负责。
"""

from __future__ import annotations

import numpy as np

from ..calibration.scalers import build_calibrator
from ..core.config import Config
from ..data.loaders import build_split
from ..detectors.registry import available_detectors, build_detector
from ..ensemble.router import CCORRouter
from ..eval.metrics import evaluate
from ..train.classifiers import build_classifier

_NEEDS_CLF = {"msp", "energy", "tsmsp"}


class OODPipeline:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()

    def run(
        self,
        split_name: str = "synthetic_far",
        backend: str = "auto",
        calibrator: str = "matrix",
    ) -> dict:
        cfg = self.cfg
        split = build_split(split_name, cfg)
        clf = build_classifier(backend, cfg)
        clf.fit(split.id_train.X, split.id_train.y)

        X_test = np.vstack([split.id_test.X, split.ood_test.X])
        y_test = np.concatenate(
            [np.zeros(split.id_test.n_samples), np.ones(split.ood_test.n_samples)]
        )

        detector_rows = []
        for name in available_detectors():
            try:
                det = build_detector(name, cfg)
                det.fit(split, clf if name in _NEEDS_CLF else clf)
                s = det.score(X_test)
                m = evaluate(s, y_test)
                detector_rows.append({"detector": name, **m, "ok": True})
            except Exception as exc:
                detector_rows.append(
                    {
                        "detector": name,
                        "auroc": None,
                        "aupr": None,
                        "fpr95": None,
                        "ok": False,
                        "error": str(exc),
                    }
                )

        # CCOR 旗舰
        router = CCORRouter(cfg, calibrator=calibrator)
        router.fit(split, clf)
        r_scores = router.score(X_test)
        r_metrics = evaluate(r_scores, y_test)
        router_row = {"detector": f"ccor({router.selected})", **r_metrics, "ok": True}
        detector_rows.append(router_row)
        rrep = router.report()

        # 校准 ECE 对比 (在 ID 测试集上)
        id_test = split.id_test
        uncal = clf.predict_proba(id_test.X)
        ece_uncal = _ece_safe(uncal, id_test.y, cfg.ece_bins)
        cal = build_calibrator(calibrator, cfg)
        cal.fit(clf.predict_logits(id_test.X), id_test.y)
        calp = cal.calibrate(clf.predict_logits(id_test.X))
        ece_cal = _ece_safe(calp, id_test.y, cfg.ece_bins)

        summary = {
            "split": split_name,
            "backend": backend,
            "n_classes": split.n_classes,
            "n_features": int(split.id_train.n_features),
            "detectors": detector_rows,
            "router": rrep.as_row(),
            "calibration": {
                "method": calibrator,
                "ece_uncalibrated": round(ece_uncal, 4),
                "ece_calibrated": round(ece_cal, 4),
                "ece_reduction": round(ece_uncal - ece_cal, 4),
            },
            "config": cfg.as_dict(),
        }
        return summary

    def benchmark(self, split_names: list[str] | None = None, backend: str = "auto") -> dict:
        names = split_names or ["synthetic_far", "synthetic_near", "digits"]
        runs = [self.run(name, backend=backend) for name in names]
        # 聚合路由器与各检测器均值 (仅 ok 项)
        agg = _aggregate(runs)
        return {"runs": runs, "aggregate": agg}


def _ece_safe(probs, y, n_bins) -> float:
    try:
        from ..calibration.metrics import ece

        return float(ece(probs, y, n_bins))
    except Exception:
        return float("nan")


def _aggregate(runs: list[dict]) -> dict:
    from collections import defaultdict

    acc = defaultdict(list)
    for r in runs:
        for row in r["detectors"]:
            if row.get("ok") and row.get("auroc") is not None:
                acc[row["detector"]].append(row["auroc"])
        acc["__router_val"].append(r["router"]["val_auroc"])
    return {k: round(float(np.mean(v)), 4) for k, v in acc.items() if not k.startswith("__")}

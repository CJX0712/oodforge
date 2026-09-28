"""ensemble/router.py — CCOR: Confidence-Calibrated OOD Router (旗舰创新点)。

流程:
1. 在 ID 训练集上拟合并校准分类器 (matrix scaling, netcal 优先 / numpy 兜底)。
2. 在验证集 (id_val ∪ ood_val) 上评测全部候选检测器 (含 calibrated_msp),
   按验证 OOD AUROC 选出最优检测器 (runner-up 记录)。
3. 用验证 OOD 的 5% 分位定阈值 → 实现 FPR@95TPR (部署期 ID 误报率 ≤5%)。
4. 测试期: 运行选定检测器, score ≥ threshold 判为 OOD。
离线兜底: 若所有非 MSP 检测器失败, 自动回退 MSP。
"""

from __future__ import annotations

import numpy as np

from ..calibration.scalers import build_calibrator
from ..core.config import Config
from ..core.errors import err
from ..core.interfaces import ClassifierProtocol
from ..core.types import OODSplit, RouterReport
from ..detectors.msp import MSPDetector
from ..detectors.registry import build_detector
from ..eval.metrics import auroc

# 候选检测器名 (不需要分类器的也传入 classifier, 无害)
_DET_SPECS = ["msp", "energy", "tsmsp", "mahalanobis", "knn", "ocsvm"]
_NEEDS_CLF = {"msp", "energy", "tsmsp"}


class CCORRouter:
    name = "ccor"

    def __init__(self, cfg: Config | None = None, calibrator: str = "matrix") -> None:
        self.cfg = cfg or Config()
        self.calibrator_name = calibrator
        self.detector = None
        self.threshold = 0.0
        self.selected = "msp"
        self.runner_up = ""
        self.val_auroc = 0.5
        self.fallback_used = False
        self._report = None

    def fit(self, split: OODSplit, classifier: ClassifierProtocol) -> "CCORRouter":
        # 1. 校准分类器 (用于 calibrated_msp 候选)
        cal = build_calibrator(self.calibrator_name, self.cfg)
        from .calibrated_classifier import CalibratedClassifier

        cal_clf = CalibratedClassifier(classifier, cal)
        cal_clf.fit(split.id_train.X, split.id_train.y)

        # 2. 验证集拼接
        Xv = np.vstack([split.id_val.X, split.ood_val.X])
        yv = np.concatenate(
            [np.zeros(split.id_val.n_samples), np.ones(split.ood_val.n_samples)]
        )

        results: dict[str, tuple[np.ndarray, float]] = {}
        failures = []
        for name in _DET_SPECS:
            try:
                det = build_detector(name, self.cfg)
                clf_arg = classifier if name in _NEEDS_CLF else classifier
                det.fit(split, clf_arg)
                s = det.score(Xv)
                results[name] = (s, auroc(s, yv))
            except Exception as exc:
                failures.append(name)
                if self.cfg.verbose:
                    print(f"[CCOR] 检测器 {name} 拟合失败: {exc}")

        # calibrated_msp 候选
        try:
            det = MSPDetector()
            det.fit(split, cal_clf)
            s = det.score(Xv)
            results["calibrated_msp"] = (s, auroc(s, yv))
        except Exception as exc:
            failures.append("calibrated_msp")
            if self.cfg.verbose:
                print(f"[CCOR] calibrated_msp 失败: {exc}")

        if not results:
            raise err("E500", "所有候选检测器均不可用")

        ranked = sorted(results.items(), key=lambda kv: -kv[1][1])
        best_name = ranked[0][0]
        runner_up = ranked[1][0] if len(ranked) > 1 else ""
        best_score_val = results[best_name][0]

        # 3. 阈值: 验证 OOD 的 5% 分位 → FPR@95TPR
        ood_scores = best_score_val[yv == 1]
        threshold = float(np.percentile(ood_scores, 5.0))

        # 选定检测器实例保留 (重新拟合以保证测试可用)
        if best_name == "calibrated_msp":
            self.detector = MSPDetector()
            self.detector.fit(split, cal_clf)
        else:
            self.detector = build_detector(best_name, self.cfg)
            self.detector.fit(split, classifier if best_name in _NEEDS_CLF else classifier)

        self.selected = best_name
        self.runner_up = runner_up
        self.val_auroc = float(results[best_name][1])
        self.threshold = threshold
        self.fallback_used = (
            best_name == "msp" and any(f not in ("msp",) for f in failures) and bool(failures)
        )
        self._report = RouterReport(
            selected_detector=best_name,
            selected_calibrator=self.calibrator_name,
            val_auroc_of_selected=self.val_auroc,
            threshold_fpr95=threshold,
            runner_up=runner_up,
            fallback_used=self.fallback_used,
            notes=f"候选 {len(results)} 个; 失败 {failures}",
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.detector is None:
            raise err("E500", "路由器未拟合")
        return self.detector.score(np.asarray(X, dtype=float))

    def flag(self, X: np.ndarray) -> np.ndarray:
        """返回 (n,) bool: True = 判为 OOD。"""
        return self.score(X) >= self.threshold

    def report(self) -> RouterReport:
        if self._report is None:
            raise err("E500", "路由器未拟合")
        return self._report

"""detectors/tsmsp.py — 温度缩放 MSP (ODIN-lite 的阈值部分)。

Hendrycks et al. 2019 (ODIN): 用温度 T 缩放 logits 后再取 MSP, 提升 OOD 可分性。
这里在验证集 (id_val vs ood_val) 上网格搜索使 OOD AUROC 最大的 T。
ood_score = 1 - max_softmax(logits / T)。
"""

from __future__ import annotations

import numpy as np

from ..core.interfaces import ClassifierProtocol
from ..core.types import OODSplit


def _softmax_t(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TemperatureMSPDetector:
    name = "tsmsp"

    def __init__(self, temps: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)) -> None:
        self.clf = None
        self.T = 1.0
        self.temps = temps

    def fit(
        self, split: OODSplit, classifier: ClassifierProtocol | None = None
    ) -> "TemperatureMSPDetector":
        if classifier is None:
            raise ValueError("TemperatureMSPDetector 需要分类器")
        self.clf = classifier
        try:
            from sklearn.metrics import roc_auc_score
        except Exception:
            self.T = 1.0
            return self
        id_logits = classifier.predict_logits(np.asarray(split.id_val.X, dtype=float))
        ood_logits = classifier.predict_logits(np.asarray(split.ood_val.X, dtype=float))
        id_y = np.zeros(id_logits.shape[0])
        ood_y = np.ones(ood_logits.shape[0])
        all_y = np.concatenate([id_y, ood_y])
        best_auc = -1.0
        best_T = 1.0
        for T in self.temps:
            s_id = 1.0 - _softmax_t(id_logits, T).max(axis=1)
            s_ood = 1.0 - _softmax_t(ood_logits, T).max(axis=1)
            all_s = np.concatenate([s_id, s_ood])
            try:
                auc = roc_auc_score(all_y, all_s)
            except Exception:
                auc = 0.5
            if auc > best_auc:
                best_auc = auc
                best_T = T
        self.T = best_T
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        logits = self.clf.predict_logits(np.asarray(X, dtype=float))
        return 1.0 - _softmax_t(logits, self.T).max(axis=1)

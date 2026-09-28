"""tests/test_eval.py — 评测指标正确性。"""

import numpy as np

from oodforge.eval.metrics import aupr, auroc, evaluate, fpr95


def test_auroc_perfect():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])  # OOD=1, 越大越 OOD
    assert abs(auroc(scores, y) - 1.0) < 1e-6


def test_auroc_random():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(200)
    y = np.array([0] * 100 + [1] * 100)
    assert 0.4 < auroc(scores, y) < 0.6


def test_fpr95_one_to_one():
    # 构造: OOD 分数全 >= ID 分数 → FPR@95TPR = 0
    id_scores = np.linspace(0.0, 0.4, 100)
    ood_scores = np.linspace(0.6, 1.0, 100)
    s = np.concatenate([id_scores, ood_scores])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    assert fpr95(s, y) == 0.0


def test_evaluate_keys():
    s = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    r = evaluate(s, y)
    assert set(r.keys()) == {"auroc", "aupr", "fpr95"}


def test_aupr_perfect():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    assert abs(aupr(scores, y) - 1.0) < 1e-6

# OODForge 架构文档

> 作者: 晨星 (CJX0712) · 域: 分布外检测 (OOD) 与置信度校准 · 版本: 0.1.0
> 复用的顶级开源: `scikit-learn` · `optuna` · `numpy` · `scipy`（netcal 可选对拍）
> 创新: 置信度门控 OOD 路由 (CGOR) + Optuna 调权的多信号融合 + 选择性预测弃权带

---

## 1. 要解决什么问题

深度模型在「训练分布外」(Out-of-Distribution) 输入上会**过度自信**地给出错误预测。
OODForge 回答三件事：

1. **检测** —— 这条输入是否来自训练分布之外？（OOD 评分）
2. **校准** —— 模型给出的概率是否可信？（ECE 校准）
3. **决策** —— 不确定时是否该「弃权」而不是硬猜？（选择性预测）

三大能力串成一条端到端链路，可一键复现、零密钥、CPU 可跑。

---

## 2. 模块架构（单向无环）

```
cli.py
  └─> pipeline/pipeline.py  (OODPipeline.run)
        ├─ data/synthetic.py      (ID / OOD 合成数据，防泄漏)
        ├─ ood/                   (6 个打分器 + 集成路由)
        │     ├─ mahalanobis.py   (类条件马氏距离, numpy)
        │     ├─ energy.py        (能量基, numpy)
        │     ├─ entropy.py       (熵/边界, numpy)
        │     ├─ gmm.py           (GMM 负对数似然, sklearn + numpy EM 兜底)
        │     ├─ iforest.py       (隔离森林, sklearn + 马氏兜底)
        │     ├─ knn.py           (k 近邻距离, sklearn + 暴力兜底)
        │     └─ ensemble.py      (CGOR 融合: Optuna TPE 调权, AUROC 最大化)
        ├─ calib/calibration.py   (温度/向量/矩阵/等渗 缩放, scipy L-BFGS)
        ├─ router/selective.py    (置信度门控级联路由: 快门→强门→弃权)
        └─ eval/metrics.py        (AUROC/AUPRC/FPR95/ECE/选择性风险)
              ↑ 全部依赖 core/ (types/errors/config/interfaces)
```

调用方向严格单向：`cli → pipeline → {data, ood, calib, router, eval} → core`，无环。

---

## 3. 接口契约（core/interfaces.py）

| 角色 | 方法 | 语义 |
|------|------|------|
| `OODScorer` | `score(X, probs, logits)` | 返回 `(n,)`，约定 **分数越大越 OOD** |
| `Calibrator` | `calibrate(probs, logits)` | 返回校准概率，**每行和为 1**，越大越可信 |
| `Router` | `route(calib_probs, ood_scores)` | 返回 `(accepted: bool[], confidence: float[])` |

统一语义保证跨模块公平评测（OOD 分与校准概率量纲相反，路由层负责对齐）。

---

## 4. 创新点：置信度门控 OOD 路由 (CGOR)

单信号 OOD 检测各有盲区（能量对特征偏移敏感、马氏对协方差假设敏感…）。
CGOR 做两件事：

### 4.1 多信号融合（强信号）
在 `val_id` vs `val_ood` 上用 **Optuna TPE** 学习凸组合权重 `w≥0, Σw=1`，
目标为融合分数的 **OOD 检测 AUROC 最大化**。融合分数比任一单信号更稳。
Optuna 不可用时退化为等权（仍可用）。

### 4.2 级联路由（决策）
对每条请求：
- **快门**：校准置信 `max(prob) ≥ high` → 直接接受（廉价）。
- **强门**：否则用融合 OOD 分数 `≤ thr` → 接受（惰性触发，仅快门不自信时算）。
- **其余** → 弃权（abstain），避免出现低质预测。

强门**惰性**触发，避免级联退化为「全模型并联」而更慢；基准按 **per-request** 计时，
与单模型公平比较。

---

## 5. 可验证不变量（单测守护）

| 不变量 | 含义 | 测试 |
|--------|------|------|
| 单调性 | 每个打分器 OOD 样本均值分数 > ID | `test_scorer_monotonicity` |
| 集成非劣 | 融合 AUROC ≥ 最优单信号 × 0.95 | `test_ensemble_non_inferior` |
| 校准降 ECE | 校准后 ECE ≤ 校准前 (+0.02) | `test_calibration_reduces_ece` |
| 概率合法 | 校准输出每行和 = 1 (atol 1e-5) | `test_calibrators_rows_sum_one` |
| 路由有效 | 覆盖率≥目标附近 & OOD 多数被弃权 | `test_router_coverage_and_ood_reject` |
| 无泄漏 | 训练/测试 ID 非字面重复 | `test_synthetic_no_leakage` |

---

## 6. 评测口径（诚实优先）

- **OOD 检测**: AUROC / AUPRC / FPR95（在 95% TPR 处的 FPR，越小越好）。标签 1=OOD。
- **校准**: ECE（等宽 10 箱）。`calib_method` 默认 `temperature`。
- **选择性预测**: 覆盖率 (coverage) / 选择性风险 (selective_risk) / OOD 弃权率。
- **难度梯度**: OOD 为「远高斯 + 均匀噪声」混合，且注入跨类歧义，避免模型处处满置信导致路由永不降级。
- **固定随机种子** (`random_state=42`)：基准可复现，数值写入 `benchmark.json`。

---

## 7. 性能基线（来自 `benchmark.json`，seed=42）

> 以下为端到端 Demo 实测值，复现命令见 README。每次重跑数字应基本一致（固定 seed）。

| 指标 | 值 |
|------|----:|
| 基分类器 ID-test 准确率 | 1.000 |
| 校准 ECE（前→后, temperature） | 0.0002 → 0.0000 |
| 路由覆盖率 / 选择性风险 | 0.900 / 0.000 |
| OOD 弃权率 | 1.000 |
| 单次请求延迟（ensemble score） | 5.55 ms |

各打分器 OOD-AUROC（seed=42，越大越 OOD）：mahalanobis 0.9725 · iforest 0.9675 · knn 0.9662 ·
**ensemble 0.9700** · gmm 0.9575 · energy 0.8105 · entropy 0.8093。完整数值见 `benchmark.json`。

---

## 8. 一键复现

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
python -m oodforge.examples.run_demo --out benchmark.json   # 端到端
python -m pytest -q -W ignore::UserWarning tests/            # 单测
```

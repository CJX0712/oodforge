# OODForge · 分布外检测与置信度校准系统

> 作者：**晨星 (CJX0712)** · 域：OOD Detection & Confidence Calibration · v0.1.0
> 复用的顶级开源：`scikit-learn` · `optuna` · `numpy` · `scipy`（netcal 可选对拍）
> 创新：**置信度门控 OOD 路由 (CGOR)** + Optuna 调权的多信号融合 + 选择性预测弃权带
> 许可：MIT · 零密钥 · CPU 可跑 · 一键复现

---

## 1. 它解决什么

深度模型在**训练分布外 (OOD)** 输入上会过度自信地给错预测。OODForge 串起三件事：

| 能力 | 回答 | 关键指标 |
|------|------|----------|
| 检测 | 这条输入是否来自分布外？ | OOD AUROC / AUPRC / FPR95 |
| 校准 | 模型给的概率可信吗？ | ECE（期望校准误差） |
| 决策 | 不确定时是否该「弃权」？ | 覆盖率 / 选择性风险 / OOD 弃权率 |

---

## 2. 一键复现

```bash
# 1) 隔离环境
python -m venv .venv && .venv/Scripts/activate        # Windows
pip install -r requirements.txt

# 2) 端到端基准（生成数据 + 评测 + 落盘 benchmark.json）
python -m oodforge.examples.run_demo --out benchmark.json

# 3) 单测（10 个不变量守护，全绿）
python -m pytest -q -W ignore::UserWarning tests/

# 4) CLI
python -m oodforge.cli benchmark --out benchmark.json
```

Docker：
```bash
docker build -t oodforge . && docker run --rm oodforge
```

---

## 3. 性能基线（seed=42，固定可复现，详见 benchmark.json）

**基分类器**（LogisticRegression, lbfgs）：ID-test 准确率 **1.000**

**OOD 检测 AUROC / FPR95（越大越 OOD，1=OOD）：**

| 打分器 | AUROC | AUPRC | FPR95 |
|--------|------:|------:|------:|
| mahalanobis (numpy) | 0.9725 | 0.9009 | 0.055 |
| iforest (sklearn) | 0.9675 | 0.8850 | 0.065 |
| knn (sklearn) | 0.9662 | 0.8811 | 0.068 |
| **ensemble (CGOR)** | **0.9700** | **0.8929** | **0.060** |
| gmm (sklearn) | 0.9575 | 0.8547 | 0.085 |
| energy (numpy) | 0.8105 | 0.7295 | 1.000 |
| entropy (numpy) | 0.8093 | 0.7281 | 1.000 |

**校准**（temperature scaling）：ECE **0.0002 → 0.0000**

**选择性路由（CGOR 双门同过才接受）：**

| 指标 | 值 |
|------|----:|
| 覆盖率 coverage | 0.900 |
| 选择性风险 selective_risk（ID 接受样本） | 0.000 |
| ID 弃权率 | 0.100 |
| **OOD 弃权率** | **1.000** |
| 单次请求延迟（ensemble score） | 5.55 ms |

> 注：默认基分类器为线性 LogReg，对 OOD 仍过度自信，故 energy/entropy 信号偏弱（AUROC≈0.81）；
> 特征型信号（mahalanobis/gmm/iforest/knn）与融合集成均达 ≈0.97。换成非线性基分类器（RF）时
> energy/entropy 通常更强——可用 `--classifier rf` 复现。所有数值在同一 per-request 口径下测量。

---

## 4. 架构（单向无环）

```
cli.py → pipeline/pipeline.py (OODPipeline.run)
   ├─ data/synthetic.py       ID/OOD 合成（固定类中心，防泄漏）
   ├─ ood/                    6 打分器 + EnsembleRouter(Optuna 调权)
   ├─ calib/calibration.py    温度/向量/矩阵/等渗 缩放
   ├─ router/selective.py     置信度门控级联路由
   └─ eval/metrics.py         AUROC/AUPRC/FPR95/ECE/选择性风险
         ↑ 全部依赖 core/(types/errors/config/interfaces)
```

接口契约统一：`OODScorer.score` **分数越大越 OOD**；`Calibrator.calibrate` 返回**行和=1**的校准概率；
`Router.route` 返回 `(accepted, confidence)`。详见 `docs/architecture.md`。

---

## 5. 创新点：CGOR（Confidence-Gated OOD Router）

1. **多信号融合（强信号）**：在 `val_id` vs `val_ood` 上用 Optuna TPE 学凸组合权重，
   目标为融合分数 OOD-AUROC 最大化；比任一单信号更稳。Optuna 不可用时退化为等权。
2. **双门路由（决策）**：对每条请求，`快门(校准置信≥high) 且 强门(融合OOD分≤thr)` 才接受，
   否则弃权。过度自信的 OOD 样本虽过快门，却因强门（OOD 分高）不过而被挡下——正是门控要拦的情形。
   强门惰性触发（仅快门不自信时算），基准按 per-request 计时，与单模型公平比较。

---

## 6. 可验证不变量（单测守护）

| 不变量 | 测试 |
|--------|------|
| 各打分器 OOD 均值 > ID（越大越 OOD） | `test_scorer_monotonicity` |
| 集成 AUROC ≥ 最优单信号 ×0.95（非劣） | `test_ensemble_non_inferior` |
| 校准后 ECE ≤ 校准前 | `test_calibration_reduces_ece` |
| 校准输出每行和=1 | `test_calibrators_rows_sum_one` |
| 路由覆盖率≈目标 & OOD 多数被弃权 | `test_router_coverage_and_ood_reject` |
| 无训练/测试泄漏 | `test_synthetic_no_leakage` |

---

## 7. 模块清单

```
oodforge/
  core/      types · errors(E100~E500) · config(ENV_OODFORGE_*) · interfaces(Protocol)
  data/      synthetic(防泄漏ID/OOD) · loaders(csv/npy/npz)
  ood/       mahalanobis · energy · entropy · gmm · iforest · knn · ensemble(CGOR)
  calib/     calibration(温度/向量/矩阵/等渗, scipy L-BFGS)
  router/    selective(双门级联)
  eval/      metrics(AUROC/AUPRC/FPR95/ECE/选择性风险)
  pipeline/  pipeline(OODPipeline.run+benchmark)
  cli.py · examples/run_demo.py
tests/       10 单测
docs/        architecture.md
requirements.txt · requirements.lock.txt · Dockerfile · Makefile · .gitignore
```

---

© 晨星 (CJX0712) · OODForge v0.1.0 · MIT License

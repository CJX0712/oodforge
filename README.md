# OODForge · 分布外检测与概率校准系统

> 随机创新的**世界顶级 AI** 子系统之一：复用顶级开源 (scikit-learn / scipy / numpy / netcal)，
> 含纯 numpy·sklearn 离线兜底链路。作者 **晨星 (CJX0712)**。

**OODForge** 是一个模块化、CPU/离线可运行、一键可复现的分布外检测 (Out-of-Distribution
Detection) 与概率校准系统。核心创新点是 **CCOR（Confidence-Calibrated OOD Router）**——
一个验证集驱动的元路由器：在部署前于验证集上评测全部候选检测器，按 OOD AUROC 选出
**当前数据分布下最强**的检测器，并用 FPR@95TPR 自适应阈值控制 ID 误报率。

---

## 为什么需要它（一句话）

同一道 OOD 题，不同检测器的强弱高度依赖 OOD 形态：

| OOD 形态 | 强者 | 弱者（典型翻车） |
|---------|------|----------------|
| 远 OOD（支撑外） | Mahalanobis / kNN / OC-SVM | **Energy 0.39**（过置信） |
| 近 OOD（类间模糊） | MSP / Energy / T-MSP | **kNN 0.39 / OC-SVM 0.03** |
| 真实未见类（digits） | T-MSP / Energy | OC-SVM 0.59 |

人工写死用 MSP 或 Mahalanobis 都会在某些场景下崩。**CCOR 自动选最强的那一个**。

---

## 技术选型（顶级项目 + 离线兜底）

| 模块 | 顶级开源后端 | 离线兜底（零下载可跑） |
|------|-------------|----------------------|
| 分类器 | scikit-learn `LogisticRegression` (lbfgs) | 纯 numpy 多分类 softmax 回归 |
| 检测器 | — | MSP / Energy / Mahalanobis / kNN / OC-SVM / T-MSP（全自研，零依赖） |
| 校准 | **netcal** (Temperature/Vector/Matrix Scaling) | 纯 numpy 网格搜索 / 梯度下降 |
| 度量 | scikit-learn (`roc_auc`, `average_precision`) | 纯 numpy Mann-Whitney AUROC + 梯形 AUPR |

> netcal 在 Python 3.13 暂无 wheel，安装失败不影响系统——校准自动降级为纯 numpy 实现。

---

## 模块架构（单向无环）

```
cli → pipeline → { data, train, detectors, calibration, ensemble(CCOR), eval } → core
```

```
oodforge/
  core/         types(dataclass) · errors(E100~E500) · config(ENV_*覆盖) · interfaces(Protocol)
  data/         synthetic(远/近/混合 OOD) · loaders(digits 未见类 OOD)
  train/        classifiers(Logistic / numpy softmax 兜底)
  detectors/    msp · energy · mahalanobis · knn · ocsvm · tsmsp · registry
  calibration/  scalers(temperature/vector/matrix + netcal) · metrics(ECE/可靠性曲线)
  ensemble/     router(CCOR 旗舰) · calibrated_classifier
  eval/         metrics(AUROC/AUPR/FPR95)
  pipeline/     OODPipeline.run() + benchmark()
  cli.py        argparse 入口
  examples/run_demo.py  端到端演示（落盘 benchmark.json）
tests/          41 单测（pytest 全绿）
docs/architecture.md
```

---

## 一键复现

```bash
# 1. 安装（CPU / 离线优先）
pip install -r requirements.txt
pip install "netcal>=1.4"   # 可选 SOTA 校准后端，缺失自动降级

# 2. 跑端到端基准演示（合成远/近 OOD + digits 未见类）
python -m oodforge.examples.run_demo

# 3. 单 split
python -m oodforge.cli --split synthetic_far
python -m oodforge.cli --split digits

# 4. 强制纯 numpy 后端（验证离线兜底）
python -m oodforge.cli --split synthetic_far --backend numpy
```

Docker：

```bash
docker build -t oodforge . && docker run --rm oodforge
```

---

## 性能基线（固定 random_state=42，确定性可复现）

| split | 检测器 | AUROC | AUPR | FPR95 |
|-------|--------|-------|------|-------|
| synthetic_far | msp | 0.861 | 0.890 | 0.735 |
| synthetic_far | energy | 0.386 | 0.445 | 0.998 |
| synthetic_far | mahalanobis | **1.000** | 1.000 | 0.000 |
| synthetic_far | knn | **1.000** | 1.000 | 0.000 |
| synthetic_far | ocsvm | **1.000** | 1.000 | 0.000 |
| synthetic_far | **ccor(mahalanobis)** | **1.000** | 1.000 | 0.000 |
| synthetic_near | msp | 0.994 | 0.993 | 0.023 |
| synthetic_near | energy | 0.994 | 0.993 | 0.020 |
| synthetic_near | mahalanobis | 0.705 | 0.629 | 0.730 |
| synthetic_near | knn | 0.395 | 0.406 | 0.900 |
| synthetic_near | ocsvm | 0.028 | 0.309 | 1.000 |
| synthetic_near | **ccor(energy)** | **0.994** | 0.993 | 0.020 |
| digits(未见类) | msp | 0.947 | 0.975 | 0.243 |
| digits(未见类) | energy | 0.971 | 0.986 | 0.138 |
| digits(未见类) | mahalanobis | 0.932 | 0.942 | 0.210 |
| digits(未见类) | knn | 0.874 | 0.909 | 0.337 |
| digits(未见类) | ocsvm | 0.592 | 0.817 | 0.978 |
| digits(未见类) | **ccor(tsmsp)** | **0.973** | 0.988 | 0.127 |

**结论**：CCOR 行在每个 split 上都是最优或近最优——因为它按数据分布自动选检测器，
而不是赌某一个固定方法。验证集 AUROC：far=1.000 / near=0.998 / digits=0.949。

校准：matrix scaling 在 ID 测试集上将 ECE 从 ~0.004–0.005 降到 ~0.002（digits Δ=+0.0024）。

---

## CCOR 工作流程

1. 在 ID 训练集上拟合并校准分类器（matrix scaling，netcal 优先 / numpy 兜底）。
2. 在验证集（id_val ∪ ood_val）评测全部候选检测器（含 calibrated_msp），按验证 OOD AUROC 选最优。
3. 以验证 OOD 的 5% 分位定阈值 → 实现 FPR@95TPR（部署期 ID 误报率 ≤5%）。
4. 测试期运行选定检测器，score ≥ threshold 判为 OOD。
5. 离线兜底：若所有非 MSP 检测器不可用，自动回退 MSP。

---

## 质量等级

- ✅ 41 单测全绿（pytest），固定 seed 确定性可复现
- ✅ ruff check + ruff format 全绿
- ✅ 零下载 CPU 可跑（numpy/scipy/sklearn 兜底链路）
- ✅ 一键复现，benchmark.json 落盘

作者：晨星 (CJX0712) · License：MIT

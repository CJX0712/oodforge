# OODForge 架构说明 (architecture.md)

## 1. 设计原则

- **单一职责 + 契约先行**：每个模块只做一件事，跨模块通过 `core/interfaces.py` 的
  `Protocol` 解耦，工厂模式（registry）负责实例化。
- **单向无环调用**：`cli → pipeline → {data, train, detectors, calibration, ensemble, eval} → core`。
- **离线优先**：每个依赖 SOTA 后端（netcal）的环节都有纯 numpy / sklearn 兜底，保证
  `零下载 + CPU` 也能跑通端到端。
- **确定性可复现**：所有随机源固定 `random_state`，基准数字锁定在 benchmark.json。

## 2. 目录与职责

| 路径 | 职责 | 关键不变量 |
|------|------|-----------|
| `core/types.py` | Dataset / OODSplit / DetectionResult / DetectorReport / CalibrationReport / RouterReport | `Dataset.X` 必为 2D；`ood_score` 越大越 OOD |
| `core/errors.py` | 错误码 E100~E500 注册表 | 所有业务异常经 `err(code, detail)` 抛出 |
| `core/config.py` | 全局配置 + `ENV_OODFORGE_*` 覆盖 | 字段可经环境变量覆盖，不修改代码 |
| `core/interfaces.py` | Classifier / OODDetector / Calibrator Protocol | 可插拔后端统一契约 |
| `data/synthetic.py` | 合成远/近/混合 OOD | ID 与 OOD 同维度、不同分布；无字面重复（防泄漏） |
| `data/loaders.py` | digits 未见类 OOD 基准 | 训练 0..k-1，OOD 用 k..9（真实未见类） |
| `train/classifiers.py` | Logistic / numpy softmax | 输出 `predict_logits` 与 `predict_proba` |
| `detectors/*` | MSP / Energy / Mahalanobis / kNN / OC-SVM / T-MSP | 统一 `fit(split, clf)` → `score(X)` 返回 ood_score |
| `calibration/scalers.py` | 温度/向量/矩阵缩放 | netcal 优先，否则 numpy 网格/梯度下降；校准不劣化 ECE |
| `calibration/metrics.py` | ECE / 可靠性曲线 | ECE = Σ 桶权重·|acc−conf| |
| `ensemble/router.py` | **CCOR 旗舰** | 验证集选最优检测器 + FPR@95TPR 阈值 + MSP 降级 |
| `ensemble/calibrated_classifier.py` | 校准分类器包装 | 仅替换 `predict_proba`，保留 `predict_logits` |
| `eval/metrics.py` | AUROC / AUPR / FPR95 | sklearn 优先，否则纯 numpy |
| `pipeline/pipeline.py` | `run()` + `benchmark()` | 串联全链路，产出 summary |
| `cli.py` / `examples/run_demo.py` | 入口 | argparse；落盘 benchmark.json |

## 3. 检测器数学（ood_score 越大越 OOD）

- **MSP**：`1 - max_softmax(logits)`。基线，线性模型对远 OOD 易过置信 → 信号弱。
- **Energy**：`E(x) = -logsumexp(logits)`。远 OOD 能量更高；但温度缩放对其仅平移，AUROC 不变。
- **Mahalanobis**：类条件高斯，取 `min_c ‖x−μ_c‖_{Σ⁻¹}`。远 OOD 距离大。
- **kNN**：标准化特征空间到 ID 第 k 近邻的欧氏距离。
- **OC-SVM**：`−decision_function`（正=inlier）；缺失时退化为单高斯对数密度。
- **T-MSP**：在验证集网格搜索使 OOD AUROC 最大的温度 T，`1 - max_softmax(logits/T)`。

## 4. CCOR 路由算法

```
输入: split(id_train/id_val/ood_val/id_test/ood_test), classifier
1. cal ← matrix scaling; cal_clf ← CalibratedClassifier(classifier, cal)  # 用于 calibrated_msp 候选
2. Xv = [id_val; ood_val], yv = [0..0, 1..1]
3. for det in {msp, energy, tsmsp, mahalanobis, knn, ocsvm, calibrated_msp}:
       s = det.fit(split, clf).score(Xv)
       results[det] = auroc(s, yv)
4. best = argmax results;  threshold = percentile(scores_ood, 5%)   # FPR@95TPR
5. 保留 best 检测器实例; report(selected, val_auroc, threshold, fallback)
部署: flag(x) = (best.score(x) >= threshold)
```

**为什么不退化成“全模型并联”**：CCOR 只在验证集上做**一次性选型**，测试期只跑**单个**
选定检测器（惰性、单请求计时），不并联所有模型——因此延迟与单检测器一致，却拿到
自适应选型的鲁棒性。

## 5. 验证证据（见 benchmark.json / README 基线表）

- 41 单测覆盖 core/data/train/detectors/calibration/eval/pipeline/router。
- 三套 split（far / near / digits）上 CCOR 行均达最优或近最优 AUROC。
- 校准 ECE 在 ID 测试集上不劣化（matrix scaling 后下降）。

## 6. 复现命令

```bash
pip install -r requirements.txt
python -m oodforge.examples.run_demo     # 落盘 benchmark.json
pytest -q -W ignore::UserWarning          # 41 passed
```

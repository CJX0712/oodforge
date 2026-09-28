"""examples/run_demo.py — 端到端演示: 跨多 split 基准 + 落盘 benchmark.json。

用法:
    python -m oodforge.examples.run_demo
    python -m oodforge.examples.run_demo --splits synthetic_far synthetic_near digits
输出: 控制台表格 + benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os

from ..core.config import load_config
from ..pipeline.pipeline import OODPipeline

H = f"{'detector':<22}{'AUROC':>9}{'AUPR':>9}{'FPR95':>9}"


def _fmt(v) -> str:
    if v is None:
        return "  n/a "
    return f"{v:>9.4f}"


def print_run(run: dict) -> None:
    print(
        f"\n=== split: {run['split']}  (backend={run['backend']}, n_classes={run['n_classes']}, n_features={run['n_features']}) ==="
    )
    print(H)
    print("-" * len(H))
    for row in run["detectors"]:
        auroc = row.get("auroc")
        aupr = row.get("aupr")
        fpr = row.get("fpr95")
        print(f"{row['detector']:<22}{_fmt(auroc)}{_fmt(aupr)}{_fmt(fpr)}")
    rt = run["router"]
    print("-" * len(H))
    print(f"CCOR 选型: {rt['selected_detector']}  (runner-up={rt['runner_up'] or '-'})")
    print(
        f"  验证 AUROC={rt['val_auroc']:.4f}  阈值(FPR@95TPR)={rt['threshold_fpr95']:.4f}  校准器={rt['selected_calibrator']}  降级={rt['fallback_used']}"
    )
    cal = run["calibration"]
    print(
        f"  校准 ECE: 未校准={cal['ece_uncalibrated']:.4f} → 校准后={cal['ece_calibrated']:.4f}  (Δ={cal['ece_reduction']:+.4f})"
    )


def main() -> dict:
    ap = argparse.ArgumentParser(description="OODForge 端到端演示")
    ap.add_argument(
        "--splits", nargs="+", default=["synthetic_far", "synthetic_near", "digits"]
    )
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--calibrator", default="matrix")
    ap.add_argument(
        "--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "benchmark.json")
    )
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(random_state=args.random_state)
    pipe = OODPipeline(cfg)
    result = pipe.benchmark(args.splits, backend=args.backend)
    for run in result["runs"]:
        print_run(run)
    print("\n=== 聚合 (各检测器跨 split 平均 AUROC) ===")
    for k, v in sorted(result["aggregate"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22}{v:.4f}")

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n已落盘: {out_path}")
    return result


if __name__ == "__main__":
    main()

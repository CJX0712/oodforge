"""OODForge · examples/run_demo — 端到端演示（落盘 benchmark.json）。

作者: 晨星 (CJX0712)
用法: python -m oodforge.examples.run_demo [--out benchmark.json] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os

from ..core.types import PipelineConfig
from ..data.synthetic import make_benchmark_datasets
from ..pipeline.pipeline import OODPipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="OODForge end-to-end demo")
    ap.add_argument("--out", default="benchmark.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--sep", type=float, default=4.0)
    ap.add_argument("--classifier", default="logreg", choices=["logreg", "rf"])
    args = ap.parse_args()

    cfg = PipelineConfig(
        random_state=args.seed,
        base_classifier=args.classifier,
        n_classes=args.n_classes,
    )
    datasets = make_benchmark_datasets(
        dim=args.dim,
        n_classes=args.n_classes,
        sep=args.sep,
        seed=args.seed,
    )
    pipe = OODPipeline(cfg)
    res = pipe.benchmark(datasets)

    payload = {
        "system": "OODForge",
        "version": "0.1.0",
        "author": "晨星 (CJX0712)",
        "config": res["config"],
        "base_accuracy": res["base_accuracy"],
        "per_request_latency_ms": res["per_request_latency_ms"],
        "ood_results": [r.__dict__ for r in res["ood_results"]],
        "calib_result": res["calib_result"].__dict__,
        "router_result": res["router_result"].__dict__,
        "records": [r.__dict__ for r in res["records"]],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[demo] benchmark.json 已写入: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

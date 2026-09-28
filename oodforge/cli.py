"""OODForge · cli — 命令行入口（argparse）。

作者: 晨星 (CJX0712)
子命令:
  benchmark  运行端到端基准（生成数据 + 评测 + 落盘 JSON）
  score      对外部数据（CSV/NPY）打分（需已训练的流水线，默认走基准）
"""

from __future__ import annotations

import argparse

from .core.types import PipelineConfig
from .data.synthetic import make_benchmark_datasets
from .pipeline.pipeline import OODPipeline


def _cmd_benchmark(args: argparse.Namespace) -> None:
    cfg = PipelineConfig(
        random_state=args.seed,
        base_classifier=args.classifier,
        n_classes=args.n_classes,
        calib_method=args.calib,
        target_coverage=args.coverage,
        ensemble_trials=args.trials,
    )
    ds = make_benchmark_datasets(
        dim=args.dim,
        n_classes=args.n_classes,
        sep=args.sep,
        seed=args.seed,
    )
    pipe = OODPipeline(cfg)
    res = pipe.benchmark(ds)
    if args.out:
        import json

        payload = {
            "system": "OODForge",
            "author": "晨星 (CJX0712)",
            "config": res["config"],
            "base_accuracy": res["base_accuracy"],
            "ood_results": [r.__dict__ for r in res["ood_results"]],
            "calib_result": res["calib_result"].__dict__,
            "router_result": res["router_result"].__dict__,
            "records": [r.__dict__ for r in res["records"]],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[cli] 已写入 {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="oodforge", description="OODForge CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="运行端到端基准")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--dim", type=int, default=10)
    b.add_argument("--n-classes", type=int, default=3)
    b.add_argument("--sep", type=float, default=4.0)
    b.add_argument("--classifier", default="logreg", choices=["logreg", "rf"])
    b.add_argument(
        "--calib",
        default="temperature",
        choices=["temperature", "vector", "matrix", "isotonic"],
    )
    b.add_argument("--coverage", type=float, default=0.90)
    b.add_argument("--trials", type=int, default=24)
    b.add_argument("--out", default="benchmark.json")
    b.set_defaults(func=_cmd_benchmark)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

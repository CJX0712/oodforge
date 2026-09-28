"""cli.py — 命令行入口 (argparse)。

用法:
    python -m oodforge.cli --split synthetic_far
    python -m oodforge.cli --demo --splits synthetic_far synthetic_near digits
"""

from __future__ import annotations

import argparse

from .core.config import load_config
from .examples.run_demo import main as demo_main
from .pipeline.pipeline import OODPipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="OODForge — 分布外检测与校准 CLI")
    ap.add_argument("--split", default="synthetic_far")
    ap.add_argument("--backend", default="auto", choices=["auto", "sklearn", "numpy"])
    ap.add_argument(
        "--calibrator", default="matrix", choices=["temperature", "vector", "matrix"]
    )
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--demo", action="store_true", help="跨多 split 基准")
    ap.add_argument(
        "--splits", nargs="+", default=["synthetic_far", "synthetic_near", "digits"]
    )
    args = ap.parse_args()

    if args.demo:
        demo_main()
        return

    cfg = load_config(random_state=args.random_state)
    pipe = OODPipeline(cfg)
    run = pipe.run(args.split, backend=args.backend, calibrator=args.calibrator)
    from .examples.run_demo import print_run

    print_run(run)


if __name__ == "__main__":
    main()

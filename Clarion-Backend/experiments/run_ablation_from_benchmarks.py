#!/usr/bin/env python
"""
Build ablation summary from benchmark JSON files.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.benchmarking.ablation import run_ablation_from_benchmarks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ablation synthesis from benchmark JSON files")
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Benchmark JSON files. If omitted, scans data/benchmarks/*.json",
    )
    parser.add_argument("--out-dir", type=str, default="data/reports/ablation")
    args = parser.parse_args()

    if args.benchmarks:
        paths = [Path(p) for p in args.benchmarks]
    else:
        paths = sorted(Path("data/benchmarks").glob("*.json"))

    if not paths:
        print("No benchmark JSON files found.")
        return 1

    outputs = run_ablation_from_benchmarks(paths, Path(args.out_dir))
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

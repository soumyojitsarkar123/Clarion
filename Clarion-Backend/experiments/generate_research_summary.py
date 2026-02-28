#!/usr/bin/env python
"""
Generate concise human-readable summary from benchmark JSON outputs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.benchmarking.summary import generate_summary, write_summary_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate research summary from benchmark JSON")
    parser.add_argument("--benchmarks", nargs="+", required=True, help="Benchmark JSON files")
    parser.add_argument("--ablation-json", type=str, default=None, help="Optional ablation summary JSON")
    parser.add_argument("--out-dir", type=str, default="data/reports")
    parser.add_argument("--stem", type=str, default="research")
    args = parser.parse_args()

    benchmarks = [
        json.loads(Path(p).read_text(encoding="utf-8"))
        for p in args.benchmarks
    ]
    ablation = None
    if args.ablation_json:
        ablation = json.loads(Path(args.ablation_json).read_text(encoding="utf-8"))

    summary = generate_summary(benchmarks, ablation=ablation)
    outputs = write_summary_files(summary, Path(args.out_dir), args.stem)
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

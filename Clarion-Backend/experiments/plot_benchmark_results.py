#!/usr/bin/env python
"""
Generate reporting plots from benchmark JSON files.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.benchmarking.reporting import (
    deterministic_stem,
    load_benchmark,
    manifest_linkage,
    multi_benchmark_stem,
    plot_confidence_shift,
    plot_embedding_comparison,
    plot_hallucination_reduction,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot benchmark JSON reporting figures")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        required=True,
        help="One or more benchmark JSON files",
    )
    parser.add_argument("--out-dir", type=str, default="data/reports/plots")
    parser.add_argument("--write-linkage", action="store_true", help="Write manifest linkage JSON")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = [load_benchmark(Path(p)) for p in args.benchmarks]
    primary = benchmarks[0]
    primary_stem = deterministic_stem(primary)
    comparison_stem = multi_benchmark_stem(benchmarks)

    outputs = {
        "hallucination_plot": plot_hallucination_reduction(
            primary, out_dir / f"{primary_stem}_hallucination_reduction.png"
        ),
        "confidence_shift_plot": plot_confidence_shift(
            primary, out_dir / f"{primary_stem}_confidence_shift.png"
        ),
        "embedding_comparison_plot": plot_embedding_comparison(
            benchmarks, out_dir / f"{comparison_stem}_embedding_comparison.png"
        ),
    }
    if args.write_linkage:
        linkage = manifest_linkage(primary, outputs)
        linkage["comparison_inputs"] = [
            b.get("manifest", {}).get("experiment_id") for b in benchmarks
        ]
        linkage_path = out_dir / f"{primary_stem}_plots_linkage.json"
        linkage_path.write_text(json.dumps(linkage, indent=2, default=str), encoding="utf-8")
        outputs["linkage"] = str(linkage_path)
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

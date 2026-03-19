#!/usr/bin/env python
"""
Export benchmark table_ready section to LaTeX and CSV.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.benchmarking.reporting import (
    deterministic_stem,
    export_table_ready_to_csv_and_latex,
    load_benchmark,
    manifest_linkage,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export benchmark tables to LaTeX/CSV")
    parser.add_argument("benchmark_json", type=str, help="Path to benchmark JSON")
    parser.add_argument("--out-dir", type=str, default="data/reports", help="Output directory")
    parser.add_argument("--stem", type=str, default=None, help="File stem for outputs")
    parser.add_argument("--write-linkage", action="store_true", help="Write manifest linkage JSON")
    args = parser.parse_args()

    bench_path = Path(args.benchmark_json)
    benchmark = load_benchmark(bench_path)
    stem = args.stem or deterministic_stem(benchmark)
    outputs = export_table_ready_to_csv_and_latex(benchmark, Path(args.out_dir), stem)
    if args.write_linkage:
        linkage = manifest_linkage(benchmark, outputs)
        linkage_path = Path(args.out_dir) / f"{stem}_tables_linkage.json"
        linkage_path.write_text(json.dumps(linkage, indent=2, default=str), encoding="utf-8")
        outputs["linkage"] = str(linkage_path)
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

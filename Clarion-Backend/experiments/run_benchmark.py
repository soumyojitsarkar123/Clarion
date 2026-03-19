#!/usr/bin/env python
"""
Run research-grade benchmark harness for relation validation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.benchmarking import BenchmarkHarness, SplitConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible benchmark harness")
    parser.add_argument("--dataset", type=str, default=None, help="Path to relation_dataset.db")
    parser.add_argument("--runs", type=int, default=10, help="Number of repeated runs")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--stratify-key", type=str, default="relation_type")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--no-graph-features", action="store_true")
    parser.add_argument("--description", type=str, default="relation validation benchmark")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum labeled records required")
    args = parser.parse_args()

    harness = BenchmarkHarness(Path(args.dataset) if args.dataset else None)
    split_cfg = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify_key=args.stratify_key,
        seed=args.seed,
    )

    try:
        result = harness.run_benchmark(
            num_runs=args.runs,
            base_seed=args.seed,
            split_config=split_cfg,
            use_graph_features=not args.no_graph_features,
            llm_provider=args.llm_provider,
            experiment_description=args.description,
            min_samples=args.min_samples,
        )
    except ValueError as e:
        print(f"Benchmark failed: {e}")
        return 1

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"Benchmark saved to {out}")
    else:
        print(f"Benchmark saved to {result['output_file']}")

    agg = result["aggregate"]["metrics"]
    print("Validated F1 mean:", round(agg["validated_f1"]["mean"], 4))
    print("Validated F1 CI95:", [round(agg["validated_f1"]["ci95_low"], 4), round(agg["validated_f1"]["ci95_high"], 4)])
    print("Hallucination reduction (abs):", round(agg["abs_reduction"]["mean"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

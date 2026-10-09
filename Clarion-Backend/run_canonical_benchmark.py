#!/usr/bin/env python3
"""
Canonical Benchmark Execution Script for Clarion.

Executes reproducible, leak-free benchmarking across multi-signal models,
scientific baselines, statistical calibrators, and ablation configurations.

Provenance tracking:
- Git commit SHA
- Host environment & dependency versions
- Deterministic dataset versioning (SHA-256)
- Document-isolated train/val/test splits
- Document-cluster paired bootstrap resampling (1,000 iterations)
"""

import argparse
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import numpy as np
import sklearn
from core.benchmarking.harness import BenchmarkHarness
from core.benchmarking.manifest import compute_dataset_version, determine_benchmark_type
from core.benchmarking.splitter import SplitConfig
from utils.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("canonical_benchmark")


def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "unversioned"


def get_environment_info() -> Dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "networkx_version": nx.__version__,
        "git_commit": get_git_commit(),
    }


def format_table(headers: list, rows: list) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(cell)))

    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-+-".join("-" * w for w in col_widths)

    lines = [fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*[str(c) for c in row]))
    return "\n".join(lines)


def run_canonical_benchmark(
    dataset_path: Path,
    num_runs: int = 5,
    base_seed: int = 42,
    use_graph_features: bool = False,
    output_dir: Path = None,
) -> Dict[str, Any]:
    output_dir = output_dir or (settings.data_dir / "benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    harness = BenchmarkHarness(dataset_path=dataset_path)
    harness.output_dir = output_dir

    split_config = SplitConfig(
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        split_strategy="document",
        doc_key="document_id",
        min_docs_for_grouping=3,
        seed=base_seed,
    )

    bench_type = determine_benchmark_type(dataset_path)

    print("=" * 80)
    print(f"CLARION CANONICAL BENCHMARK — [{bench_type}]")
    print("=" * 80)
    print(f"Dataset Path:    {dataset_path}")
    print(f"Dataset SHA-256: {compute_dataset_version(dataset_path)}")
    print(f"Number of Runs:  {num_runs} (seeds: {list(range(base_seed, base_seed + num_runs))})")
    print(f"Graph Features:  {use_graph_features}")
    print(f"Split Strategy:  Document-Isolated (train=0.6, val=0.2, test=0.2)")

    if bench_type == "SYNTHETIC_REGRESSION":
        print("\n> [IMPORTANT NOTE - SCIENTIFIC INTEGRITY]")
        print("> This execution uses the synthetic regression fixture.")
        print("> These results demonstrate software pipeline integrity and absence of leakage,")
        print("> NOT empirical scientific evidence on real-world knowledge graphs.")
        print("> Empirical claims require human-annotated gold datasets.\n")

    results = harness.run_benchmark(
        num_runs=num_runs,
        base_seed=base_seed,
        split_config=split_config,
        use_graph_features=use_graph_features,
        experiment_description=f"canonical-{bench_type.lower()}-run",
    )

    # Attach provenance
    env_info = get_environment_info()
    results["provenance"] = env_info

    # Save final artifact
    output_file = Path(results["output_file"])
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Render results to stdout
    table_ready = results.get("table_ready", {})

    print("\n### 1. Model Classification Performance (Mean ± Std, 95% CI)")
    perf_headers = ["Method", "F1 Score", "95% CI", "Accuracy", "Precision", "Recall", "AUROC"]
    perf_rows = []
    for row in table_ready.get("model_performance_table", []):
        f1_str = f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f}"
        ci_str = f"[{row['f1_ci95'][0]:.3f}, {row['f1_ci95'][1]:.3f}]"
        perf_rows.append([
            row["method"],
            f1_str,
            ci_str,
            f"{row['accuracy_mean']:.4f}",
            f"{row['precision_mean']:.4f}",
            f"{row['recall_mean']:.4f}",
            f"{row['auroc_mean']:.4f}",
        ])
    print(format_table(perf_headers, perf_rows))

    print("\n### 2. Statistical Calibration & Hallucination Metrics")
    cal_headers = ["Method", "ECE", "MCE", "Brier Score", "FAR"]
    cal_rows = []
    for row in table_ready.get("calibration_table", []):
        cal_rows.append([
            row["method"],
            f"{row['ece_mean']:.4f} ± {row['ece_std']:.4f}",
            f"{row['mce_mean']:.4f}",
            f"{row['brier_mean']:.4f}",
            f"{row['far_mean']:.4f}",
        ])
    print(format_table(cal_headers, cal_rows))

    print("\n### 3. Signal Ablation Study")
    abl_headers = ["Configuration", "F1 Score", "ECE", "FAR"]
    abl_rows = []
    for row in table_ready.get("ablation_table", []):
        abl_rows.append([
            row["configuration"],
            f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f}",
            f"{row['ece_mean']:.4f} ± {row['ece_std']:.4f}",
            f"{row['far_mean']:.4f}",
        ])
    print(format_table(abl_headers, abl_rows))

    print("\n### 4. Paired Document-Cluster Bootstrap (1,000 Iterations)")
    boot_headers = ["Comparison", "Δ F1 (Mean [95% CI])", "Δ ECE (Mean [95% CI])", "Runs"]
    boot_rows = []
    for row in table_ready.get("bootstrap_table", []):
        f1_delta = f"{row['delta_f1_mean']:+.4f} [{row['delta_f1_ci95'][0]:.3f}, {row['delta_f1_ci95'][1]:.3f}]"
        ece_delta = f"{row['delta_ece_mean']:+.4f} [{row['delta_ece_ci95'][0]:.3f}, {row['delta_ece_ci95'][1]:.3f}]"
        boot_rows.append([row["comparison"], f1_delta, ece_delta, row["runs_evaluated"]])
    if boot_rows:
        print(format_table(boot_headers, boot_rows))
    else:
        print("Document cluster bootstrap: Single document per split or low sample size.")

    print("\n" + "=" * 80)
    print(f"Benchmark artifact written to: {output_file}")
    print("=" * 80 + "\n")

    return results


def main():
    default_dataset = Path("Clarion-Backend/data/ground_truth/synthetic_benchmark_fixture.json")
    if not default_dataset.exists():
        default_dataset = Path("data/ground_truth/synthetic_benchmark_fixture.json")

    parser = argparse.ArgumentParser(description="Clarion Canonical Benchmark CLI")
    parser.add_argument("--dataset", type=Path, default=default_dataset, help="Dataset path (.json or .db)")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--graph-features", action="store_true", help="Include graph structural features")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to store benchmark JSON")

    args = parser.parse_args()

    run_canonical_benchmark(
        dataset_path=args.dataset,
        num_runs=args.runs,
        base_seed=args.seed,
        use_graph_features=args.graph_features,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

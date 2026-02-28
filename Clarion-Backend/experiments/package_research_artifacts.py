#!/usr/bin/env python
"""
Bundle benchmark-derived research artifacts into a review-ready directory.
"""

import argparse
import json
import shutil
import sys
from hashlib import sha1
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.benchmarking.ablation import run_ablation_from_benchmarks
from core.benchmarking.reporting import (
    deterministic_stem,
    export_table_ready_to_csv_and_latex,
    load_benchmark,
    manifest_linkage,
    multi_benchmark_stem,
    plot_confidence_shift,
    plot_embedding_comparison,
    plot_hallucination_reduction,
)
from core.benchmarking.summary import generate_summary, write_summary_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Package benchmark research artifacts")
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Benchmark JSON files. If omitted, uses data/benchmarks/*.json",
    )
    parser.add_argument("--out-root", type=str, default="data/reports/packages")
    args = parser.parse_args()

    benchmark_paths = (
        [Path(p) for p in args.benchmarks]
        if args.benchmarks
        else sorted(Path("data/benchmarks").glob("*.json"))
    )
    if not benchmark_paths:
        print("No benchmark JSON files found.")
        return 1

    benchmarks = [load_benchmark(p) for p in benchmark_paths]
    package_dir = _package_dir(Path(args.out_root), benchmarks)
    dirs = {
        "benchmarks": package_dir / "benchmarks",
        "tables": package_dir / "tables",
        "plots": package_dir / "plots",
        "ablation": package_dir / "ablation",
        "summary": package_dir / "summary",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    artifact_index: Dict[str, Dict] = {"benchmarks": {}, "tables": {}, "plots": {}, "ablation": {}, "summary": {}}

    # Copy benchmark JSON inputs with deterministic names.
    for src, bench in zip(benchmark_paths, benchmarks):
        stem = deterministic_stem(bench)
        dst = dirs["benchmarks"] / f"{stem}.json"
        shutil.copy2(src, dst)
        artifact_index["benchmarks"][stem] = {
            "source": str(src),
            "copied_to": str(dst),
            "manifest": bench.get("manifest", {}),
        }

    # Export tables for each benchmark.
    for bench in benchmarks:
        stem = deterministic_stem(bench)
        files = export_table_ready_to_csv_and_latex(bench, dirs["tables"], stem)
        artifact_index["tables"][stem] = manifest_linkage(bench, files)

    # Plot outputs.
    plot_files: Dict[str, str] = {}
    primary = benchmarks[0]
    primary_stem = deterministic_stem(primary)
    comparison_stem = multi_benchmark_stem(benchmarks)
    try:
        plot_files["hallucination_plot"] = plot_hallucination_reduction(
            primary, dirs["plots"] / f"{primary_stem}_hallucination_reduction.png"
        )
        plot_files["confidence_shift_plot"] = plot_confidence_shift(
            primary, dirs["plots"] / f"{primary_stem}_confidence_shift.png"
        )
        plot_files["embedding_comparison_plot"] = plot_embedding_comparison(
            benchmarks, dirs["plots"] / f"{comparison_stem}_embedding_comparison.png"
        )
        artifact_index["plots"]["status"] = "ok"
    except RuntimeError as e:
        artifact_index["plots"]["status"] = "skipped"
        artifact_index["plots"]["reason"] = str(e)
    artifact_index["plots"]["files"] = plot_files
    artifact_index["plots"]["linkage"] = manifest_linkage(primary, plot_files)

    # Ablation.
    ablation_result = run_ablation_from_benchmarks(benchmark_paths, dirs["ablation"])
    artifact_index["ablation"] = ablation_result

    # Summary.
    ablation_payload = json.loads(Path(ablation_result["json"]).read_text(encoding="utf-8"))
    summary = generate_summary(benchmarks, ablation=ablation_payload)
    summary_files = write_summary_files(summary, dirs["summary"], "package")
    artifact_index["summary"] = summary_files

    linkage_path = package_dir / "manifest_linkage.json"
    linkage_path.write_text(json.dumps(artifact_index, indent=2, default=str), encoding="utf-8")
    _write_package_readme(package_dir, benchmark_paths)

    print(json.dumps({"package_dir": str(package_dir), "linkage": str(linkage_path)}, indent=2))
    return 0


def _package_dir(out_root: Path, benchmarks: List[Dict]) -> Path:
    ids = sorted(str(b.get("manifest", {}).get("experiment_id", "")) for b in benchmarks)
    digest = sha1("|".join(ids).encode("utf-8")).hexdigest()[:10]
    return out_root / f"package_{digest}"


def _write_package_readme(package_dir: Path, benchmark_paths: List[Path]) -> None:
    benchmark_args = " ".join(str(p).replace("\\", "/") for p in benchmark_paths)
    text = f"""# Research Artifact Package

This directory contains benchmark-derived outputs prepared for supervisor review and paper submission.

## Structure
- `benchmarks/`: input benchmark JSON files copied with deterministic names.
- `tables/`: LaTeX and CSV tables exported from `table_ready`.
- `plots/`: figure outputs (hallucination reduction, confidence shift, embedding comparison).  
  Note: plotting requires `matplotlib`.
- `ablation/`: synthesized ablation summary in JSON, CSV, and LaTeX.
- `summary/`: concise human-readable and JSON summaries.
- `manifest_linkage.json`: mapping from artifacts back to experiment manifests.

## Interpretation Guide
- `model_performance_table`: core performance metrics (accuracy/F1 with uncertainty).
- `hallucination_table`: before-vs-after false-accept comparison and reduction.
- `confidence_shift_table`: direction/magnitude of confidence calibration effect.
- `ablation_summary`: staged contribution analysis:
  baseline -> +statistical -> +graph -> +validation model.

## Example End-to-End Workflow
1. Run benchmark:
```bash
python experiments/run_benchmark.py --runs 10 --seed 42 --description "paper benchmark"
```
2. Export tables:
```bash
python experiments/export_benchmark_tables.py data/benchmarks/<benchmark>.json --out-dir data/reports --write-linkage
```
3. Generate summary:
```bash
python experiments/generate_research_summary.py --benchmarks data/benchmarks/<benchmark>.json --ablation-json data/reports/ablation/ablation_summary.json --out-dir data/reports/summary --stem paper
```
4. Package artifacts:
```bash
python experiments/package_research_artifacts.py --benchmarks {benchmark_args} --out-root data/reports/packages
```
"""
    (package_dir / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

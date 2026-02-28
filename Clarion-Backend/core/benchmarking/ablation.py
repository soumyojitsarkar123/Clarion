"""
Ablation synthesis from benchmark JSON outputs.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.benchmarking.reporting import _write_latex


def run_ablation_from_benchmarks(benchmark_paths: List[Path], out_dir: Path) -> Dict[str, Any]:
    benchmarks = [json.loads(Path(p).read_text(encoding="utf-8")) for p in benchmark_paths]
    if not benchmarks:
        raise ValueError("No benchmark files provided")

    grouped: Dict[Tuple[str, str, str], Dict[bool, Dict[str, Any]]] = {}
    for b in benchmarks:
        m = b.get("manifest", {})
        key = (
            str(m.get("dataset_version", "")),
            str(m.get("embedding_model", "")),
            str(m.get("llm_provider", "")),
        )
        grouped.setdefault(key, {})[bool(m.get("use_graph_features", False))] = b

    rows = []
    for key, variants in grouped.items():
        no_graph = variants.get(False)
        with_graph = variants.get(True)
        if not no_graph and not with_graph:
            continue

        source = with_graph or no_graph
        manifest = source.get("manifest", {})

        baseline = _metric_row(no_graph or with_graph, "baseline")
        statistical = _metric_row(no_graph or with_graph, "model")
        graph = _metric_row(with_graph, "model")
        validated = _metric_row(with_graph, "validated")

        rows.extend(
            [
                _build_row(manifest, "Baseline (LLM confidence only)", baseline),
                _build_row(manifest, "+ Statistical features", statistical),
                _build_row(manifest, "+ Graph features", graph),
                _build_row(manifest, "+ Validation model (calibrated)", validated),
            ]
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ablation_summary.json"
    csv_path = out_dir / "ablation_summary.csv"
    tex_path = out_dir / "ablation_summary.tex"

    payload = {"rows": rows, "num_rows": len(rows)}
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_latex(tex_path, "Ablation Summary", rows)

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "latex": str(tex_path),
        "rows": len(rows),
    }


def _metric_row(benchmark: Optional[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    if not benchmark:
        return {
            "accuracy_mean": None,
            "accuracy_std": None,
            "f1_mean": None,
            "f1_std": None,
        }
    m = benchmark.get("aggregate", {}).get("metrics", {})
    acc = m.get(f"{prefix}_accuracy", {})
    f1 = m.get(f"{prefix}_f1", {})
    return {
        "accuracy_mean": acc.get("mean"),
        "accuracy_std": acc.get("std"),
        "f1_mean": f1.get("mean"),
        "f1_std": f1.get("std"),
    }


def _build_row(manifest: Dict[str, Any], stage: str, metric: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_version": manifest.get("dataset_version"),
        "embedding_model": manifest.get("embedding_model"),
        "llm_provider": manifest.get("llm_provider"),
        "ablation_stage": stage,
        "accuracy_mean": metric.get("accuracy_mean"),
        "accuracy_std": metric.get("accuracy_std"),
        "f1_mean": metric.get("f1_mean"),
        "f1_std": metric.get("f1_std"),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

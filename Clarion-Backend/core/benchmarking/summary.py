"""
Concise human-readable benchmark summary generation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_summary(
    benchmarks: List[Dict[str, Any]], ablation: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if not benchmarks:
        raise ValueError("No benchmark inputs provided")

    rows = []
    for b in benchmarks:
        manifest = b.get("manifest", {})
        metrics = b.get("aggregate", {}).get("metrics", {})
        rows.append(
            {
                "experiment_id": manifest.get("experiment_id"),
                "embedding_model": manifest.get("embedding_model"),
                "llm_provider": manifest.get("llm_provider"),
                "use_graph_features": bool(manifest.get("use_graph_features", False)),
                "validated_f1_mean": _metric(metrics, "validated_f1", "mean"),
                "validated_f1_ci95": [
                    _metric(metrics, "validated_f1", "ci95_low"),
                    _metric(metrics, "validated_f1", "ci95_high"),
                ],
                "validated_accuracy_mean": _metric(metrics, "validated_accuracy", "mean"),
                "hallucination_abs_reduction_mean": _metric(metrics, "abs_reduction", "mean"),
                "hallucination_rel_reduction_pct_mean": _metric(metrics, "rel_reduction_pct", "mean"),
                "confidence_shift_mean": _metric(metrics, "mean_shift_all", "mean"),
            }
        )

    best = max(
        rows,
        key=lambda r: (
            r["validated_f1_mean"],
            r["hallucination_abs_reduction_mean"],
            r["validated_accuracy_mean"],
        ),
    )

    ablation_highlights = _ablation_highlights(ablation) if ablation else []

    text = _render_text(rows, best, ablation_highlights)
    return {
        "summary_rows": rows,
        "best_configuration": best,
        "ablation_highlights": ablation_highlights,
        "report_text": text,
    }


def write_summary_files(summary: Dict[str, Any], out_dir: Path, stem: str) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}_summary.json"
    txt_path = out_dir / f"{stem}_summary.txt"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(summary.get("report_text", ""), encoding="utf-8")
    return {"json": str(json_path), "txt": str(txt_path)}


def _metric(metrics: Dict[str, Any], key: str, field: str) -> float:
    return float(metrics.get(key, {}).get(field, 0.0))


def _ablation_highlights(ablation: Dict[str, Any]) -> List[str]:
    rows = ablation.get("rows", [])
    if not rows:
        return ["No ablation rows available."]

    stage_map = {}
    for row in rows:
        stage_map[row.get("ablation_stage")] = row

    highlights = []
    base = stage_map.get("Baseline (LLM confidence only)")
    graph = stage_map.get("+ Graph features")
    validated = stage_map.get("+ Validation model (calibrated)")

    if base and graph:
        delta = _safe(graph.get("f1_mean")) - _safe(base.get("f1_mean"))
        highlights.append(f"Graph features vs baseline F1 delta: {delta:.4f}")
    if graph and validated:
        delta = _safe(validated.get("f1_mean")) - _safe(graph.get("f1_mean"))
        highlights.append(f"Calibration vs graph F1 delta: {delta:.4f}")
    if base and validated:
        delta_acc = _safe(validated.get("accuracy_mean")) - _safe(base.get("accuracy_mean"))
        highlights.append(f"End-to-end validated vs baseline accuracy delta: {delta_acc:.4f}")
    if not highlights:
        highlights.append("Ablation data present but insufficient for pairwise deltas.")
    return highlights


def _safe(v: Any) -> float:
    return float(v) if v is not None else 0.0


def _render_text(rows: List[Dict[str, Any]], best: Dict[str, Any], ablation: List[str]) -> str:
    lines = []
    lines.append("Research Benchmark Summary")
    lines.append("")
    lines.append("Key Aggregate Metrics:")
    for r in rows:
        lines.append(
            f"- {r['experiment_id']}: validated_f1={r['validated_f1_mean']:.4f}, "
            f"validated_accuracy={r['validated_accuracy_mean']:.4f}, "
            f"hallucination_abs_reduction={r['hallucination_abs_reduction_mean']:.4f}, "
            f"confidence_shift_mean={r['confidence_shift_mean']:.4f}"
        )
    lines.append("")
    lines.append("Best Configuration:")
    lines.append(
        f"- experiment_id={best['experiment_id']}, embedding={best['embedding_model']}, "
        f"llm_provider={best['llm_provider']}, use_graph_features={best['use_graph_features']}, "
        f"validated_f1={best['validated_f1_mean']:.4f}, "
        f"f1_ci95=[{best['validated_f1_ci95'][0]:.4f}, {best['validated_f1_ci95'][1]:.4f}]"
    )
    lines.append("")
    lines.append("Hallucination Reduction Summary:")
    lines.append(
        f"- Best run-group absolute reduction mean={best['hallucination_abs_reduction_mean']:.4f}, "
        f"relative reduction mean={best['hallucination_rel_reduction_pct_mean']:.4f}%"
    )
    lines.append("")
    lines.append("Ablation Outcome Highlights:")
    for h in ablation:
        lines.append(f"- {h}")
    lines.append("")
    return "\n".join(lines)

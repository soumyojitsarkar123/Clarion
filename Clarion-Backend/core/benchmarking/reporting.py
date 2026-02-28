"""
Reporting utilities for benchmark JSON outputs.
"""

import csv
import json
import re
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List


def load_benchmark(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def deterministic_stem(benchmark: Dict[str, Any]) -> str:
    manifest = benchmark.get("manifest", {})
    experiment_id = str(manifest.get("experiment_id", "exp"))
    dataset_version = str(manifest.get("dataset_version", "unknown")).replace(":", "_")
    embedding = str(manifest.get("embedding_model", "embedding"))
    embedding_short = embedding.split("/")[-1]
    graph_flag = "graph" if manifest.get("use_graph_features") else "nograph"
    return _sanitize_filename(f"{experiment_id}_{dataset_version}_{embedding_short}_{graph_flag}")


def manifest_linkage(benchmark: Dict[str, Any], artifact_paths: Dict[str, str]) -> Dict[str, Any]:
    manifest = benchmark.get("manifest", {})
    return {
        "experiment_id": manifest.get("experiment_id"),
        "dataset_version": manifest.get("dataset_version"),
        "embedding_model": manifest.get("embedding_model"),
        "llm_provider": manifest.get("llm_provider"),
        "use_graph_features": manifest.get("use_graph_features"),
        "artifacts": artifact_paths,
    }


def export_table_ready_to_csv_and_latex(
    benchmark: Dict[str, Any], out_dir: Path, stem: str
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_ready = benchmark.get("table_ready", {})

    files = {}
    for table_name, rows in table_ready.items():
        csv_path = out_dir / f"{stem}_{table_name}.csv"
        tex_path = out_dir / f"{stem}_{table_name}.tex"
        _write_csv(csv_path, rows)
        _write_latex(tex_path, table_name, rows)
        files[f"{table_name}_csv"] = str(csv_path)
        files[f"{table_name}_latex"] = str(tex_path)
    return files


def plot_hallucination_reduction(benchmark: Dict[str, Any], out_path: Path) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("matplotlib is required for plotting utilities") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = benchmark.get("table_ready", {}).get("hallucination_table", [{}])[0]

    baseline = float(row.get("baseline_mean", 0.0))
    validated = float(row.get("validated_mean", 0.0))
    reduction = float(row.get("absolute_reduction_mean", 0.0))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Baseline FAR", "Validated FAR"], [baseline, validated], color=["#cc6677", "#228833"])
    ax.set_ylabel("False Accept Rate")
    ax.set_title(f"Hallucination Reduction (abs={reduction:.4f})")
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)


def plot_embedding_comparison(benchmarks: List[Dict[str, Any]], out_path: Path) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("matplotlib is required for plotting utilities") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = []
    f1_means = []
    f1_low = []
    f1_high = []

    for b in benchmarks:
        manifest = b.get("manifest", {})
        agg = b.get("aggregate", {}).get("metrics", {})
        f1 = agg.get("validated_f1", {})
        labels.append(str(manifest.get("embedding_model", "unknown")))
        f1_means.append(float(f1.get("mean", 0.0)))
        f1_low.append(float(f1.get("ci95_low", 0.0)))
        f1_high.append(float(f1.get("ci95_high", 0.0)))

    if not labels:
        raise ValueError("No benchmark inputs provided")

    x = list(range(len(labels)))
    yerr_lower = [max(0.0, m - lo) for m, lo in zip(f1_means, f1_low)]
    yerr_upper = [max(0.0, hi - m) for m, hi in zip(f1_means, f1_high)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, f1_means, color="#4477aa")
    ax.errorbar(x, f1_means, yerr=[yerr_lower, yerr_upper], fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Validated F1")
    ax.set_title("Embedding Comparison (from benchmark JSON)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)


def plot_confidence_shift(benchmark: Dict[str, Any], out_path: Path) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("matplotlib is required for plotting utilities") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    runs = benchmark.get("runs", [])
    shifts = [
        float(
            r.get("hallucination_evaluation", {})
            .get("confidence_shift", {})
            .get("mean_shift_all", 0.0)
        )
        for r in runs
    ]
    if not shifts:
        shifts = [0.0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(shifts, bins=min(10, max(3, len(shifts))), color="#66c2a5", edgecolor="black")
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean Confidence Shift (validated - baseline)")
    ax.set_ylabel("Run Count")
    ax.set_title("Confidence Distribution Shift Across Runs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_latex(path: Path, caption: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("% empty table\n", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    col_spec = "l" + "c" * (len(columns) - 1)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{_escape_latex(caption)}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    lines.append(" & ".join(_escape_latex(c) for c in columns) + " \\\\")
    lines.append("\\hline")
    for row in rows:
        vals = [_format_cell(row.get(c)) for c in columns]
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return _escape_latex("[" + ", ".join(str(v) for v in value) + "]")
    return _escape_latex(str(value))


def _escape_latex(text: str) -> str:
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "_": "\\_",
        "#": "\\#",
        "{": "\\{",
        "}": "\\}",
    }
    out = text
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


def multi_benchmark_stem(benchmarks: List[Dict[str, Any]]) -> str:
    ids = sorted(str(b.get("manifest", {}).get("experiment_id", "")) for b in benchmarks)
    digest = sha1("|".join(ids).encode("utf-8")).hexdigest()[:10]
    return _sanitize_filename(f"comparison_{digest}")


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")

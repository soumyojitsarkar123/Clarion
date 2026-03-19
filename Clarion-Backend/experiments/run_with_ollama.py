#!/usr/bin/env python
"""
Helper script for running experiments with Ollama local models.

Usage:
    python experiments/run_with_ollama.py --model qwen3:latest --task benchmark
    python experiments/run_with_ollama.py --model deepseek-coder:6.7b --task analyze --file document.pdf
    python experiments/run_with_ollama.py --task stats

Available models (installed):
    - qwen3:latest (Qwen 3 8B)
    - deepseek-coder:6.7b (DeepSeek Coder 6.7B)
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OLLAMA_API_BASE = "http://localhost:11434/v1"


def set_ollama_env(model: str) -> None:
    """Configure environment for Ollama model."""
    os.environ["OLLAMA_MODEL"] = model
    os.environ["OLLAMA_API_BASE"] = OLLAMA_API_BASE


def check_ollama_connection() -> bool:
    """Check if Ollama server is running."""
    import httpx

    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def check_model_available(model: str) -> bool:
    """Check if specific model is available in Ollama."""
    import httpx

    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(model in m.get("name", "") for m in models)
    except Exception:
        pass
    return False


def get_ollama_provider(model: str):
    """Create an Ollama-compatible LLM provider."""
    from core.llm.glm_provider import GLMProvider

    return GLMProvider(api_key="ollama", model=model, api_base=OLLAMA_API_BASE)


def task_stats(args) -> int:
    """Show dataset statistics."""
    from services.relation_dataset_service import RelationDatasetService

    service = RelationDatasetService()
    stats = service.get_dataset_stats()

    print("\n=== Relation Dataset Statistics ===")
    print(f"Total records:       {stats['total_records']}")
    print(f"Labeled records:     {stats['labeled_records']}")
    print(f"Unlabeled records:   {stats['unlabeled_records']}")
    print(f"Avg LLM confidence:  {stats['average_llm_confidence']:.3f}")
    print(f"Avg co-occurrence:   {stats['average_cooccurrence']:.3f}")
    print("\nRelation types:")
    for rel_type, count in stats["relation_types"].items():
        print(f"  - {rel_type}: {count}")

    if stats["labeled_records"] < 30:
        print(
            "\n[!] WARNING: Less than 30 labeled records. Need more data for reliable experiments."
        )
    elif stats["labeled_records"] < 100:
        print("\n[!] NOTE: 30-100 labeled records. Results may have higher variance.")
    else:
        print("\n[OK] Sufficient labeled records for stable benchmarking.")

    return 0


def task_benchmark(args) -> int:
    """Run benchmark with Ollama model."""
    from core.benchmarking import BenchmarkHarness, SplitConfig

    set_ollama_env(args.model)

    print(f"\n=== Running Benchmark with {args.model} ===")

    if not check_ollama_connection():
        print("ERROR: Ollama server not running. Start with: ollama serve")
        return 1

    if not check_model_available(args.model):
        print(
            f"ERROR: Model '{args.model}' not found. Pull with: ollama pull {args.model}"
        )
        return 1

    harness = BenchmarkHarness()

    split_cfg = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    try:
        result = harness.run_benchmark(
            num_runs=args.runs,
            base_seed=args.seed,
            split_config=split_cfg,
            use_graph_features=not args.no_graph_features,
            llm_provider="ollama",
            experiment_description=args.description or f"Ollama {args.model} benchmark",
            min_samples=args.min_samples,
        )
    except ValueError as e:
        print(f"Benchmark failed: {e}")
        return 1

    agg = result["aggregate"]["metrics"]
    print("\n=== Results ===")
    print(f"Validated F1 mean:  {agg['validated_f1']['mean']:.4f}")
    print(
        f"Validated F1 CI95:  [{agg['validated_f1']['ci95_low']:.4f}, {agg['validated_f1']['ci95_high']:.4f}]"
    )
    print(f"Hallucination reduction (abs): {agg['abs_reduction']['mean']:.4f}")
    print(f"\nOutput: {result['output_file']}")

    return 0


def task_analyze(args) -> int:
    """Analyze document with Ollama model."""
    set_ollama_env(args.model)

    if not check_ollama_connection():
        print("ERROR: Ollama server not running. Start with: ollama serve")
        return 1

    if not check_model_available(args.model):
        print(
            f"ERROR: Model '{args.model}' not found. Pull with: ollama pull {args.model}"
        )
        return 1

    if not args.file:
        print("ERROR: --file required for analyze task")
        return 1

    import httpx

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return 1

    base_url = args.api_base or "http://localhost:8000"

    print(f"\n=== Analyzing {file_path.name} with {args.model} ===")

    try:
        with open(file_path, "rb") as f:
            upload_response = httpx.post(
                f"{base_url}/upload", files={"file": (file_path.name, f)}, timeout=60
            )
        upload_response.raise_for_status()
        doc_id = upload_response.json()["document_id"]
        print(f"Uploaded: {doc_id}")

        analyze_response = httpx.post(f"{base_url}/analyze/{doc_id}", timeout=300)
        analyze_response.raise_for_status()
        result = analyze_response.json()

        print(f"Chunks: {result.get('chunk_count', 0)}")
        print(f"Concepts: {result.get('concept_count', 0)}")
        print(f"Relations logged: {result.get('relations_logged', 0)}")

    except httpx.HTTPError as e:
        print(f"ERROR: API request failed: {e}")
        return 1

    return 0


def task_label(args) -> int:
    """Interactive labeling helper."""
    from services.relation_dataset_service import RelationDatasetService

    service = RelationDatasetService()
    records = service.get_dataset(is_valid=None, limit=args.limit)

    unlabeled = [r for r in records if r.is_valid is None]

    if not unlabeled:
        print("No unlabeled records found.")
        return 0

    print(f"\n=== Labeling {len(unlabeled)} Records ===")
    print("Commands: y=yes (valid), n=no (invalid), s=skip, q=quit\n")

    labeled = 0
    for i, record in enumerate(unlabeled):
        print(f"[{i + 1}/{len(unlabeled)}]")
        print(f"  Concept A: {record.concept_a}")
        print(f"  Concept B: {record.concept_b}")
        print(f"  Type: {record.relation_type}")
        print(f"  LLM Confidence: {record.llm_confidence:.2f}")
        print(f"  Context: {record.chunk_context[:200]}...")

        while True:
            cmd = input("Label [y/n/s/q]: ").strip().lower()
            if cmd in ("y", "n", "s", "q"):
                break
            print("Invalid command. Use y/n/s/q")

        if cmd == "q":
            break
        elif cmd == "s":
            continue
        elif cmd in ("y", "n"):
            service.update_validation(record.record_id, cmd == "y")
            labeled += 1
            print(f"  -> Labeled as {'valid' if cmd == 'y' else 'invalid'}\n")

    print(f"\nLabeled {labeled} records.")
    return 0


def task_compare(args) -> int:
    """Compare multiple Ollama models."""
    from core.benchmarking import BenchmarkHarness

    if not check_ollama_connection():
        print("ERROR: Ollama server not running. Start with: ollama serve")
        return 1

    results = {}

    for model in args.models:
        if not check_model_available(model):
            print(f"Skipping {model}: not found. Pull with: ollama pull {model}")
            continue

        print(f"\n=== Benchmarking {model} ===")
        set_ollama_env(model)

        harness = BenchmarkHarness()
        try:
            result = harness.run_benchmark(
                num_runs=args.runs,
                base_seed=args.seed,
                llm_provider="ollama",
                experiment_description=f"Ollama {model} comparison",
                min_samples=args.min_samples,
            )
            agg = result["aggregate"]["metrics"]
            results[model] = {
                "f1": agg["validated_f1"]["mean"],
                "f1_ci": [
                    agg["validated_f1"]["ci95_low"],
                    agg["validated_f1"]["ci95_high"],
                ],
                "hallucination_reduction": agg["abs_reduction"]["mean"],
            }
        except ValueError as e:
            print(f"  Failed: {e}")
            continue

    if results:
        print("\n=== Comparison Results ===")
        print(f"{'Model':<20} {'F1':>8} {'F1 CI95':>20} {'Halluc. Red.':>12}")
        print("-" * 62)
        for model, r in sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True):
            ci = f"[{r['f1_ci'][0]:.3f}, {r['f1_ci'][1]:.3f}]"
            print(
                f"{model:<20} {r['f1']:>8.3f} {ci:>20} {r['hallucination_reduction']:>12.4f}"
            )

    return 0


def task_list(args) -> int:
    """List available Ollama models."""
    import httpx

    if not check_ollama_connection():
        print("ERROR: Ollama server not running. Start with: ollama serve")
        return 1

    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json().get("models", [])

        print("\n=== Available Ollama Models ===")
        if not models:
            print("No models installed. Pull with: ollama pull <model>")
            return 0

        for m in models:
            name = m.get("name", "unknown")
            size = m.get("size", 0) / (1024**3)
            print(f"  {name} ({size:.1f} GB)")

        print("\nUse --model flag to select a model for experiments.")

    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run experiments with Ollama local models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="qwen3:latest",
        help="Ollama model name (e.g., qwen3:latest, deepseek-coder:6.7b)",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        required=True,
        choices=["benchmark", "analyze", "stats", "label", "compare", "list"],
        help="Task to run",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000",
        help="Backend API base URL",
    )

    # Benchmark args
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--no-graph-features", action="store_true")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--description", type=str, default=None)

    # Analyze args
    parser.add_argument("--file", "-f", type=str, help="Document file to analyze")

    # Label args
    parser.add_argument(
        "--limit", type=int, default=50, help="Max records for labeling"
    )

    # Compare args
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3:latest", "deepseek-coder:6.7b"],
        help="Models to compare",
    )

    args = parser.parse_args()

    tasks = {
        "stats": task_stats,
        "benchmark": task_benchmark,
        "analyze": task_analyze,
        "label": task_label,
        "compare": task_compare,
        "list": task_list,
    }

    return tasks[args.task](args)


if __name__ == "__main__":
    raise SystemExit(main())

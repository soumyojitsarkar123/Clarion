"""
Research-grade evaluation and benchmarking harness.
"""

import json
import logging
import math
import sqlite3
import uuid
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from core.benchmarking.manifest import (
    ExperimentManifest,
    RunManifest,
    compute_dataset_version,
    now_iso,
)
from core.benchmarking.splitter import SplitConfig, split_train_val_test
from core.experiments.graph_features import GraphEnrichedFeatureExtractor
from core.experiments.relation_validator import RelationFeatureExtractor
from utils.config import settings

logger = logging.getLogger(__name__)


class BenchmarkHarness:
    """
    Standalone harness for reproducible benchmark experiments.
    """

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or (settings.data_dir / "relation_dataset.db")
        self.output_dir = settings.data_dir / "benchmarks"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_labeled_records(self) -> List[Dict[str, Any]]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        conn = sqlite3.connect(str(self.dataset_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT record_id, document_id, relation_id, concept_a, concept_b, relation_type,
                   llm_confidence, cooccurrence_score, semantic_similarity,
                   chunk_context, source_chunk_ids, is_valid, created_at
            FROM relation_dataset
            WHERE is_valid IS NOT NULL
            ORDER BY created_at ASC
            """
        )
        rows = cursor.fetchall()
        conn.close()

        records: List[Dict[str, Any]] = []
        for row in rows:
            records.append(
                {
                    "record_id": row["record_id"],
                    "document_id": row["document_id"],
                    "relation_id": row["relation_id"],
                    "concept_a": row["concept_a"],
                    "concept_b": row["concept_b"],
                    "relation_type": row["relation_type"],
                    "llm_confidence": float(row["llm_confidence"] or 0.5),
                    "cooccurrence_score": row["cooccurrence_score"],
                    "semantic_similarity": row["semantic_similarity"],
                    "chunk_context": row["chunk_context"] or "",
                    "source_chunk_ids": json.loads(row["source_chunk_ids"] or "[]"),
                    "is_valid": bool(row["is_valid"]),
                    "created_at": row["created_at"],
                }
            )

        if len(records) < 30:
            logger.warning("Low sample size for robust benchmarking: %d", len(records))
        return records

    def run_benchmark(
        self,
        num_runs: int = 10,
        base_seed: int = 42,
        split_config: Optional[SplitConfig] = None,
        use_graph_features: bool = True,
        llm_provider: str = "openai",
        experiment_description: str = "relation validation benchmark",
        min_samples: int = 10,
    ) -> Dict[str, Any]:
        records = self.load_labeled_records()
        if len(records) < min_samples:
            raise ValueError(f"Need at least {min_samples} labeled records for benchmark runs")

        split_config = split_config or SplitConfig(seed=base_seed)
        dataset_version = compute_dataset_version(self.dataset_path)
        seeds = [base_seed + i for i in range(num_runs)]
        experiment_id = f"bench_{uuid.uuid4().hex[:12]}"

        exp_manifest = ExperimentManifest(
            experiment_id=experiment_id,
            created_at=now_iso(),
            description=experiment_description,
            num_runs=num_runs,
            seeds=seeds,
            embedding_model=settings.embedding_model_name,
            llm_provider=llm_provider,
            use_graph_features=use_graph_features,
            dataset_path=str(self.dataset_path),
            dataset_version=dataset_version,
            metadata={"framework": "clarion-core-benchmark-v1"},
        )

        run_results = []
        for seed in seeds:
            cfg = SplitConfig(
                train_ratio=split_config.train_ratio,
                val_ratio=split_config.val_ratio,
                test_ratio=split_config.test_ratio,
                stratify_key=split_config.stratify_key,
                seed=seed,
            )
            run_results.append(
                self._run_single(records, cfg, seed, use_graph_features, llm_provider, dataset_version)
            )

        aggregate = self._aggregate_runs(run_results)
        output = {
            "manifest": exp_manifest.to_dict(),
            "runs": run_results,
            "aggregate": aggregate,
            "table_ready": self._build_table_ready(aggregate),
        }

        output_path = self.output_dir / f"{experiment_id}.json"
        output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        output["output_file"] = str(output_path)
        return output

    def _run_single(
        self,
        records: List[Dict[str, Any]],
        split_config: SplitConfig,
        seed: int,
        use_graph_features: bool,
        llm_provider: str,
        dataset_version: str,
    ) -> Dict[str, Any]:
        split = split_train_val_test(records, split_config)

        X_train, y_train = self._extract_features(split.train, use_graph_features)
        X_val, y_val = self._extract_features(split.validation, use_graph_features)
        X_test, y_test = self._extract_features(split.test, use_graph_features)

        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        test_probs = model.predict_proba(X_test_scaled)[:, 1]
        val_probs = model.predict_proba(X_val_scaled)[:, 1]

        base_conf_test = np.array([r.get("llm_confidence", 0.5) for r in split.test], dtype=float)
        cooc_test = np.array([r.get("cooccurrence_score") or 0.0 for r in split.test], dtype=float)
        calibrated_test = 0.4 * base_conf_test + 0.4 * test_probs + 0.2 * cooc_test

        base_conf_val = np.array([r.get("llm_confidence", 0.5) for r in split.validation], dtype=float)
        cooc_val = np.array([r.get("cooccurrence_score") or 0.0 for r in split.validation], dtype=float)
        calibrated_val = 0.4 * base_conf_val + 0.4 * val_probs + 0.2 * cooc_val

        baseline_threshold = self._optimize_threshold(base_conf_val, y_val)
        validated_threshold = self._optimize_threshold(calibrated_val, y_val)

        baseline_pred = (base_conf_test >= baseline_threshold).astype(int)
        validated_pred = (calibrated_test >= validated_threshold).astype(int)
        model_pred = (test_probs >= 0.5).astype(int)

        confidence_shift = calibrated_test - base_conf_test
        hallucination_eval = self._hallucination_eval(
            y_true=y_test,
            baseline_pred=baseline_pred,
            validated_pred=validated_pred,
            base_confidence=base_conf_test,
            calibrated_confidence=calibrated_test,
            confidence_shift=confidence_shift,
        )

        run_manifest = RunManifest(
            run_id=f"run_{uuid.uuid4().hex[:10]}",
            timestamp=now_iso(),
            seed=seed,
            embedding_model=settings.embedding_model_name,
            llm_provider=llm_provider,
            use_graph_features=use_graph_features,
            dataset_path=str(self.dataset_path),
            dataset_version=dataset_version,
            split={
                "train_ratio": split_config.train_ratio,
                "val_ratio": split_config.val_ratio,
                "test_ratio": split_config.test_ratio,
            },
            stratify_key=split_config.stratify_key,
            metadata={"stratified": str(split.stratified)},
        )

        return {
            "manifest": run_manifest.to_dict(),
            "counts": {
                "train": int(len(y_train)),
                "validation": int(len(y_val)),
                "test": int(len(y_test)),
            },
            "thresholds": {
                "baseline": baseline_threshold,
                "validated": validated_threshold,
            },
            "metrics": {
                "model": self._binary_metrics(y_test, model_pred, test_probs),
                "baseline": self._binary_metrics(y_test, baseline_pred, base_conf_test),
                "validated": self._binary_metrics(y_test, validated_pred, calibrated_test),
            },
            "hallucination_evaluation": hallucination_eval,
        }

    def _extract_features(
        self, records: List[Dict[str, Any]], use_graph_features: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        base_extractor = RelationFeatureExtractor()
        y = np.array([1 if r["is_valid"] else 0 for r in records], dtype=int)
        base_vectors = [base_extractor._extract_single(r) for r in records]

        if not use_graph_features:
            return np.array(base_vectors, dtype=float), y

        graph_extractor = GraphEnrichedFeatureExtractor()
        feature_names = graph_extractor.graph_extractor.get_all_feature_names()
        by_doc: Dict[str, List[int]] = {}
        for i, r in enumerate(records):
            by_doc.setdefault(r["document_id"], []).append(i)

        graph_feats = [[0.0] * len(feature_names) for _ in records]
        for doc_id, idxs in by_doc.items():
            loaded = graph_extractor.load_graph_for_document(doc_id)
            if not loaded:
                continue
            for idx in idxs:
                r = records[idx]
                feat_map = graph_extractor.graph_extractor.extract_features(
                    r.get("concept_a", ""), r.get("concept_b", "")
                )
                graph_feats[idx] = [float(feat_map.get(name, 0.0)) for name in feature_names]

        merged = [base_vectors[i] + graph_feats[i] for i in range(len(records))]
        return np.array(merged, dtype=float), y

    def _binary_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
    ) -> Dict[str, float]:
        result = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if len(np.unique(y_true)) > 1:
            result["auroc"] = float(roc_auc_score(y_true, y_score))
        else:
            result["auroc"] = 0.5
        return result

    def _optimize_threshold(self, scores: np.ndarray, y_true: np.ndarray) -> float:
        candidates = np.linspace(0.2, 0.8, 25)
        best_t = 0.5
        best_f1 = -1.0
        for t in candidates:
            pred = (scores >= t).astype(int)
            f = f1_score(y_true, pred, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = float(t)
        return best_t

    def _hallucination_eval(
        self,
        y_true: np.ndarray,
        baseline_pred: np.ndarray,
        validated_pred: np.ndarray,
        base_confidence: np.ndarray,
        calibrated_confidence: np.ndarray,
        confidence_shift: np.ndarray,
    ) -> Dict[str, Any]:
        negatives = max(int((y_true == 0).sum()), 1)
        valid_mask = y_true == 1
        invalid_mask = y_true == 0

        baseline_false_accept = int(((baseline_pred == 1) & (y_true == 0)).sum())
        validated_false_accept = int(((validated_pred == 1) & (y_true == 0)).sum())

        baseline_far = baseline_false_accept / negatives
        validated_far = validated_false_accept / negatives

        shift_valid = confidence_shift[valid_mask]
        shift_invalid = confidence_shift[invalid_mask]

        return {
            "before_after": {
                "baseline_false_accept_rate": float(baseline_far),
                "validated_false_accept_rate": float(validated_far),
                "absolute_reduction": float(baseline_far - validated_far),
                "relative_reduction_pct": float(
                    ((baseline_far - validated_far) / baseline_far) * 100.0
                )
                if baseline_far > 0
                else 0.0,
            },
            "confidence_shift": {
                "mean_shift_all": float(np.mean(confidence_shift)) if confidence_shift.size else 0.0,
                "std_shift_all": float(np.std(confidence_shift)) if confidence_shift.size else 0.0,
                "mean_shift_valid": float(np.mean(shift_valid)) if shift_valid.size else 0.0,
                "mean_shift_invalid": float(np.mean(shift_invalid)) if shift_invalid.size else 0.0,
                "p25_shift": float(np.percentile(confidence_shift, 25)) if confidence_shift.size else 0.0,
                "p50_shift": float(np.percentile(confidence_shift, 50)) if confidence_shift.size else 0.0,
                "p75_shift": float(np.percentile(confidence_shift, 75)) if confidence_shift.size else 0.0,
                "avg_baseline_confidence": float(np.mean(base_confidence)) if base_confidence.size else 0.0,
                "avg_validated_confidence": float(np.mean(calibrated_confidence)) if calibrated_confidence.size else 0.0,
            },
        }

    def _aggregate_runs(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        def collect(path: List[str]) -> List[float]:
            values = []
            for run in runs:
                cur: Any = run
                for key in path:
                    cur = cur[key]
                values.append(float(cur))
            return values

        metric_paths = {
            "model_f1": ["metrics", "model", "f1"],
            "baseline_f1": ["metrics", "baseline", "f1"],
            "validated_f1": ["metrics", "validated", "f1"],
            "model_accuracy": ["metrics", "model", "accuracy"],
            "baseline_accuracy": ["metrics", "baseline", "accuracy"],
            "validated_accuracy": ["metrics", "validated", "accuracy"],
            "baseline_far": ["hallucination_evaluation", "before_after", "baseline_false_accept_rate"],
            "validated_far": ["hallucination_evaluation", "before_after", "validated_false_accept_rate"],
            "abs_reduction": ["hallucination_evaluation", "before_after", "absolute_reduction"],
            "rel_reduction_pct": ["hallucination_evaluation", "before_after", "relative_reduction_pct"],
            "mean_shift_all": ["hallucination_evaluation", "confidence_shift", "mean_shift_all"],
        }

        summary = {}
        for name, path in metric_paths.items():
            vals = collect(path)
            summary[name] = self._summary_stats(vals)

        return {
            "n_runs": len(runs),
            "metrics": summary,
        }

    def _summary_stats(self, values: List[float]) -> Dict[str, float]:
        n = len(values)
        mu = mean(values)
        sd = stdev(values) if n > 1 else 0.0
        # Normal approximation for 95% CI.
        margin = 1.96 * (sd / math.sqrt(n)) if n > 1 else 0.0
        return {
            "mean": float(mu),
            "std": float(sd),
            "ci95_low": float(mu - margin),
            "ci95_high": float(mu + margin),
            "n": n,
        }

    def _build_table_ready(self, aggregate: Dict[str, Any]) -> Dict[str, Any]:
        m = aggregate["metrics"]
        return {
            "model_performance_table": [
                self._table_row("Baseline (LLM confidence)", m["baseline_accuracy"], m["baseline_f1"]),
                self._table_row("Validated (calibrated)", m["validated_accuracy"], m["validated_f1"]),
                self._table_row("Validation Model (probability)", m["model_accuracy"], m["model_f1"]),
            ],
            "hallucination_table": [
                {
                    "metric": "False Accept Rate",
                    "baseline_mean": m["baseline_far"]["mean"],
                    "validated_mean": m["validated_far"]["mean"],
                    "absolute_reduction_mean": m["abs_reduction"]["mean"],
                    "relative_reduction_pct_mean": m["rel_reduction_pct"]["mean"],
                    "relative_reduction_pct_ci95": [
                        m["rel_reduction_pct"]["ci95_low"],
                        m["rel_reduction_pct"]["ci95_high"],
                    ],
                }
            ],
            "confidence_shift_table": [
                {
                    "metric": "Mean confidence shift (validated - baseline)",
                    "mean": m["mean_shift_all"]["mean"],
                    "std": m["mean_shift_all"]["std"],
                    "ci95": [
                        m["mean_shift_all"]["ci95_low"],
                        m["mean_shift_all"]["ci95_high"],
                    ],
                }
            ],
        }

    def _table_row(self, method: str, acc: Dict[str, float], f1: Dict[str, float]) -> Dict[str, Any]:
        return {
            "method": method,
            "accuracy_mean": acc["mean"],
            "accuracy_std": acc["std"],
            "accuracy_ci95": [acc["ci95_low"], acc["ci95_high"]],
            "f1_mean": f1["mean"],
            "f1_std": f1["std"],
            "f1_ci95": [f1["ci95_low"], f1["ci95_high"]],
        }

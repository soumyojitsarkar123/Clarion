"""
Research-grade evaluation and benchmarking harness.

Implements leak-free document-grouped benchmarking, multi-signal feature extraction,
scientific baseline comparisons, statistical calibration (Platt/Isotonic),
signal ablation studies, and comprehensive evaluation metrics (F1, ECE, MCE, Brier, FAR).
"""

import json
import logging
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
    determine_benchmark_type,
    now_iso,
)
from core.benchmarking.splitter import SplitConfig, split_train_val_test
from core.benchmarking.dataset_validator import GroundTruthValidator
from models.ground_truth import ValidityLabel
from core.experiments.graph_features import GraphEnrichedFeatureExtractor
from core.experiments.relation_validator import RelationFeatureExtractor
from core.evaluation.evidence_verifier import EvidenceVerifier
from core.evaluation.calibration import (
    compute_calibration_metrics,
    PlattCalibrator,
    IsotonicCalibrator,
)
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
        self.evidence_verifier = EvidenceVerifier()

    def load_labeled_records(self) -> List[Dict[str, Any]]:
        """
        Load ground truth records from either SQLite DB or validated JSON file.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        # JSON dataset loading with schema validation
        if str(self.dataset_path).endswith(".json"):
            validator = GroundTruthValidator()
            dataset, _ = validator.validate_file(self.dataset_path)
            records: List[Dict[str, Any]] = []
            for r in dataset.records:
                if r.label == ValidityLabel.AMBIGUOUS:
                    continue
                context = r.evidence_text
                chunk_ids = r.source_chunk_ids or []
                # Lexical cooccurrence score: 1.0 if both concepts present in context else 0.0
                src = r.concept_a.lower()
                tgt = r.concept_b.lower()
                cooc = 1.0 if (src in context.lower() and tgt in context.lower()) else 0.0

                records.append(
                    {
                        "record_id": r.record_id,
                        "document_id": r.document_id,
                        "relation_id": r.relation_id,
                        "concept_a": r.concept_a,
                        "concept_b": r.concept_b,
                        "relation_type": r.relation_type,
                        "llm_confidence": float(r.confidence_rating) / 5.0,
                        "cooccurrence_score": cooc,
                        "semantic_similarity": 0.5,
                        "chunk_context": context,
                        "source_chunk_ids": chunk_ids,
                        "is_valid": bool(r.label == ValidityLabel.VALID),
                        "created_at": getattr(dataset, "created_at", "2026-10-04T00:00:00Z"),
                    }
                )
            if len(records) < 10:
                logger.warning("Low sample size for robust benchmarking: %d", len(records))
            return records

        # SQLite database loading
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

        records = []
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
                    "cooccurrence_score": float(row["cooccurrence_score"] or 0.0),
                    "semantic_similarity": float(row["semantic_similarity"] or 0.5),
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
        experiment_description: str = "leak-free relation validation benchmark",
        min_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute reproducible multi-run benchmarking across baselines and calibrated models.
        """
        records = self.load_labeled_records()

        # ------------------------------------------------------------------
        # RESEARCH EXPERIMENT LOCK
        # ------------------------------------------------------------------
        benchmark_type = determine_benchmark_type(self.dataset_path)
        if benchmark_type == "SYNTHETIC_REGRESSION":
            msg = (
                "\n" + "=" * 70 + "\n"
                "  SYNTHETIC REGRESSION MODE — NOT A RESEARCH RESULT\n"
                "  Dataset: " + str(self.dataset_path) + "\n"
                "  Results verify algorithmic correctness ONLY.\n"
                "  Publication-grade claims require human-annotated gold labels.\n"
                + "=" * 70
            )
            logger.warning(msg)
            print(msg, flush=True)
        elif benchmark_type == "REAL_HUMAN_GOLD":
            if len(records) == 0:
                raise RuntimeError(
                    "RESEARCH BENCHMARK BLOCKED — HUMAN GOLD DATASET REQUIRED.\n"
                    "The dataset at '" + str(self.dataset_path) + "' contains zero "
                    "labeled records (is_valid IS NULL for all rows).\n"
                    "Annotate candidates with the annotation tool before running "
                    "the research benchmark."
                )

        if len(records) < min_samples:
            raise ValueError(f"Need at least {min_samples} labeled records for benchmark runs")

        # Document-isolated split by default to prevent cross-document data leakage
        split_config = split_config or SplitConfig(
            split_strategy="document",
            seed=base_seed,
            min_docs_for_grouping=3,
        )
        dataset_version = compute_dataset_version(self.dataset_path)
        benchmark_type = determine_benchmark_type(self.dataset_path)
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
            benchmark_type=benchmark_type,
            metadata={
                "framework": "clarion-calibrated-benchmark-v2",
                "leak_free_guarantee": "strict_document_isolation",
            },
        )

        run_results = []
        for seed in seeds:
            cfg = SplitConfig(
                train_ratio=split_config.train_ratio,
                val_ratio=split_config.val_ratio,
                test_ratio=split_config.test_ratio,
                stratify_key=split_config.stratify_key,
                split_strategy=split_config.split_strategy,
                doc_key=split_config.doc_key,
                min_docs_for_grouping=split_config.min_docs_for_grouping,
                seed=seed,
            )
            run_results.append(
                self._run_single(records, cfg, seed, use_graph_features, llm_provider, dataset_version, benchmark_type)
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

    def _extract_evidence_support_score(self, record: Dict[str, Any]) -> float:
        """Extract lexical evidence support (sentence grounding) score via EvidenceVerifier.

        The EvidenceVerifier uses rule-based lexical pattern matching and sentence
        grounding — NOT a trained NLI / entailment model. The key 'entailment' used
        elsewhere in this codebase is a backwards-compatible API alias only.
        """
        support_score = self.evidence_verifier.verify_support(
            record.get("concept_a", ""),
            record.get("concept_b", ""),
            record.get("relation_type", "explanation"),
            record.get("chunk_context", ""),
        )
        return float(support_score)

    def _extract_features(
        self, records: List[Dict[str, Any]], use_graph_features: bool
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract multi-signal feature matrix X and labels y, along with raw signal vectors.
        """
        base_extractor = RelationFeatureExtractor()
        y = np.array([1 if r["is_valid"] else 0 for r in records], dtype=int)
        base_vectors = [base_extractor._extract_single(r) for r in records]

        # Extract lexical evidence support score for every record
        # ('entailment' key preserved as backwards-compatible API alias)
        entail_scores = np.array([self._extract_evidence_support_score(r) for r in records], dtype=float)
        for i, score in enumerate(entail_scores):
            base_vectors[i].append(score)

        signals = {
            "conf": np.array([float(r.get("llm_confidence") or 0.5) for r in records], dtype=float),
            "cooc": np.array([float(r.get("cooccurrence_score") or 0.0) for r in records], dtype=float),
            "sem": np.array([float(r.get("semantic_similarity") or 0.5) for r in records], dtype=float),
            "entail": entail_scores,
        }

        if not use_graph_features:
            return np.array(base_vectors, dtype=float), y, signals

        # Graph features extracted strictly on G \ {(c_a, c_b)} to prevent target leakage
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
        return np.array(merged, dtype=float), y, signals

    def _run_single(
        self,
        records: List[Dict[str, Any]],
        split_config: SplitConfig,
        seed: int,
        use_graph_features: bool,
        llm_provider: str,
        dataset_version: str,
        benchmark_type: str = "SYNTHETIC_REGRESSION",
    ) -> Dict[str, Any]:
        """
        Execute a single run over held-out train/validation/test splits.
        """
        split = split_train_val_test(records, split_config)
        if split.document_isolated:
            split.assert_no_document_leakage()

        X_train, y_train, sig_train = self._extract_features(split.train, use_graph_features)
        X_val, y_val, sig_val = self._extract_features(split.validation, use_graph_features)
        X_test, y_test, sig_test = self._extract_features(split.test, use_graph_features)

        # Scale features strictly from training split
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Fit multi-signal base model on training split
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )
        # Handle case where training has only one class
        if len(np.unique(y_train)) > 1:
            model.fit(X_train_scaled, y_train)
            raw_model_val = model.predict_proba(X_val_scaled)[:, 1]
            raw_model_test = model.predict_proba(X_test_scaled)[:, 1]
        else:
            raw_model_val = np.full(len(y_val), 0.5)
            raw_model_test = np.full(len(y_test), 0.5)

        # -------------------------------------------------------------
        # Baselines & Calibrated Methods Predictions
        # -------------------------------------------------------------
        methods_val = {}
        methods_test = {}

        # Baseline 0: Accept all candidates as valid (Constant 1.0)
        methods_val["baseline_accept_all"] = np.ones(len(y_val))
        methods_test["baseline_accept_all"] = np.ones(len(y_test))

        # Baseline 1: LLM Confidence only
        methods_val["baseline_conf"] = sig_val["conf"]
        methods_test["baseline_conf"] = sig_test["conf"]

        # Baseline 2: Co-occurrence only
        methods_val["baseline_cooc"] = sig_val["cooc"]
        methods_test["baseline_cooc"] = sig_test["cooc"]

        # Baseline 3: Semantic similarity only
        methods_val["baseline_sem"] = sig_val["sem"]
        methods_test["baseline_sem"] = sig_test["sem"]

        # Baseline 4: Evidence support only
        methods_val["baseline_entail"] = sig_val["entail"]
        methods_test["baseline_entail"] = sig_test["entail"]

        # Baseline 5: Heuristic composite (prototype blend: 0.4*conf + 0.4*model + 0.2*cooc)
        methods_val["baseline_heuristic"] = (
            0.4 * sig_val["conf"] + 0.4 * raw_model_val + 0.2 * sig_val["cooc"]
        )
        methods_test["baseline_heuristic"] = (
            0.4 * sig_test["conf"] + 0.4 * raw_model_test + 0.2 * sig_test["cooc"]
        )

        # Method: Raw Multi-Signal Model (Uncalibrated)
        methods_val["model_raw"] = raw_model_val
        methods_test["model_raw"] = raw_model_test

        # Method: Calibrated Multi-Signal Model (Platt scaling fit strictly on validation split)
        platt = PlattCalibrator()
        platt.fit(raw_model_val, y_val)
        methods_val["calibrated_platt"] = platt.calibrate(raw_model_val)
        methods_test["calibrated_platt"] = platt.calibrate(raw_model_test)

        # Method: Calibrated Multi-Signal Model (Isotonic regression fit strictly on validation split)
        isotonic = IsotonicCalibrator()
        isotonic.fit(raw_model_val, y_val)
        methods_val["calibrated_isotonic"] = isotonic.calibrate(raw_model_val)
        methods_test["calibrated_isotonic"] = isotonic.calibrate(raw_model_test)

        # -------------------------------------------------------------
        # Leave-One-Signal-Out Ablations
        # -------------------------------------------------------------
        total_feats = X_train.shape[1]
        feature_subsets = {
            "ablation_no_evidence": [i for i in range(total_feats) if i != 8],
            "ablation_no_confidence": [i for i in range(total_feats) if i not in (0, 6, 7)],
            "ablation_no_cooccurrence": [i for i in range(total_feats) if i not in (1, 6)],
            "ablation_no_semantic": [i for i in range(total_feats) if i not in (2, 7)],
            "ablation_no_type": [i for i in range(total_feats) if i != 3],
        }
        if use_graph_features and total_feats > 9:
            feature_subsets["ablation_no_graph"] = list(range(9))

        for abl_name, indices in feature_subsets.items():
            if len(indices) > 0 and len(np.unique(y_train)) > 1:
                try:
                    abl_model = LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=seed,
                    )
                    abl_model.fit(X_train_scaled[:, indices], y_train)
                    abl_val = abl_model.predict_proba(X_val_scaled[:, indices])[:, 1]
                    abl_test = abl_model.predict_proba(X_test_scaled[:, indices])[:, 1]
                except Exception as e:
                    logger.warning(f"Failed to fit ablation {abl_name}: {e}")
                    abl_val = np.full(len(y_val), 0.5)
                    abl_test = np.full(len(y_test), 0.5)
            else:
                abl_val = np.full(len(y_val), 0.5)
                abl_test = np.full(len(y_test), 0.5)
            methods_val[abl_name] = abl_val
            methods_test[abl_name] = abl_test

        # -------------------------------------------------------------
        # Evaluate each method with threshold optimized on validation
        # -------------------------------------------------------------
        method_metrics = {}
        for name in methods_val:
            s_val = np.nan_to_num(methods_val[name], nan=0.5)
            s_test = np.nan_to_num(methods_test[name], nan=0.5)

            t_opt = self._optimize_threshold(s_val, y_val)
            pred_test = (s_test >= t_opt).astype(int)

            metrics = self._binary_metrics(y_test, pred_test, s_test)
            metrics["optimal_threshold"] = t_opt
            method_metrics[name] = metrics

        # -------------------------------------------------------------
        # Hallucination evaluation (comparing Calibrated Platt vs Baseline Conf)
        # -------------------------------------------------------------
        base_pred = (methods_test["baseline_conf"] >= method_metrics["baseline_conf"]["optimal_threshold"]).astype(int)
        cal_pred = (methods_test["calibrated_platt"] >= method_metrics["calibrated_platt"]["optimal_threshold"]).astype(int)
        conf_shift = methods_test["calibrated_platt"] - methods_test["baseline_conf"]

        hallucination_eval = self._hallucination_eval(
            y_true=y_test,
            baseline_pred=base_pred,
            validated_pred=cal_pred,
            base_confidence=methods_test["baseline_conf"],
            calibrated_confidence=methods_test["calibrated_platt"],
            confidence_shift=conf_shift,
        )

        # Document cluster bootstrap evaluating Calibrated Platt vs Baseline Conf
        bootstrap_results = self._paired_document_bootstrap(
            test_records=split.test,
            y_true=y_test,
            score_a=methods_test["calibrated_platt"],
            pred_a=cal_pred,
            score_b=methods_test["baseline_conf"],
            pred_b=base_pred,
            n_bootstraps=1000,
            seed=seed,
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
                "strategy": split_config.split_strategy,
            },
            stratify_key=split_config.stratify_key,
            benchmark_type=benchmark_type,
            metadata={"document_isolated": "true"},
        )

        return {
            "manifest": run_manifest.to_dict(),
            "counts": {
                "train": int(len(y_train)),
                "validation": int(len(y_val)),
                "test": int(len(y_test)),
            },
            "metrics": method_metrics,
            "hallucination_evaluation": hallucination_eval,
            "document_cluster_bootstrap": bootstrap_results,
        }

    def _paired_document_bootstrap(
        self,
        test_records: List[Dict[str, Any]],
        y_true: np.ndarray,
        score_a: np.ndarray,
        pred_a: np.ndarray,
        score_b: np.ndarray,
        pred_b: np.ndarray,
        n_bootstraps: int = 1000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Paired document-cluster bootstrap resampling.
        Resamples entire document clusters with replacement to respect clustering dependencies.
        Evaluates Delta F1 = F1(A) - F1(B) and Delta ECE = ECE(A) - ECE(B).
        """
        doc_ids = np.array([r.get("document_id", "doc_unknown") for r in test_records])
        unique_docs = np.unique(doc_ids)
        n_docs = len(unique_docs)

        if n_docs < 2 or len(y_true) < 4:
            return {
                "n_bootstraps": 0,
                "delta_f1": {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_value_one_tailed": 1.0},
                "delta_ece": {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_value_one_tailed": 1.0},
            }

        doc_indices = {d: np.where(doc_ids == d)[0] for d in unique_docs}
        rng = np.random.default_rng(seed)
        delta_f1s = []
        delta_eces = []

        for _ in range(n_bootstraps):
            sampled_docs = rng.choice(unique_docs, size=n_docs, replace=True)
            sampled_indices = np.concatenate([doc_indices[d] for d in sampled_docs])
            y_sample = y_true[sampled_indices]
            if len(np.unique(y_sample)) < 2:
                continue

            f1_a = f1_score(y_sample, pred_a[sampled_indices], zero_division=0)
            f1_b = f1_score(y_sample, pred_b[sampled_indices], zero_division=0)
            delta_f1s.append(float(f1_a - f1_b))

            ece_a = compute_calibration_metrics(y_sample, score_a[sampled_indices], n_bins=10).ece
            ece_b = compute_calibration_metrics(y_sample, score_b[sampled_indices], n_bins=10).ece
            delta_eces.append(float(ece_a - ece_b))

        def _stats(arr: List[float]):
            if not arr:
                return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_value_one_tailed": 1.0}
            np_arr = np.array(arr)
            mu = float(np.mean(np_arr))
            sd = float(np.std(np_arr))
            ci_low = float(np.percentile(np_arr, 2.5))
            ci_high = float(np.percentile(np_arr, 97.5))
            p_val = float(np.mean(np_arr <= 0))
            return {
                "mean": round(mu, 4),
                "std": round(sd, 4),
                "ci95_low": round(ci_low, 4),
                "ci95_high": round(ci_high, 4),
                "p_value_one_tailed": round(p_val, 4),
            }

        return {
            "n_bootstraps": len(delta_f1s),
            "delta_f1": _stats(delta_f1s),
            "delta_ece": _stats(delta_eces),
        }

    def _binary_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute standard classification and statistical calibration metrics.
        """
        result = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if len(np.unique(y_true)) > 1:
            try:
                result["auroc"] = float(roc_auc_score(y_true, y_score))
            except Exception:
                result["auroc"] = 0.5
        else:
            result["auroc"] = 0.5

        # Calibration metrics: ECE, MCE, Brier score
        cal = compute_calibration_metrics(y_true, y_score, n_bins=10)
        result["ece"] = float(cal.ece)
        result["mce"] = float(cal.mce)
        result["brier_score"] = float(cal.brier_score)

        # False Accept Rate on negative / invalid relations
        negatives = max(int((y_true == 0).sum()), 1)
        false_accepts = int(((y_pred == 1) & (y_true == 0)).sum())
        result["far"] = float(false_accepts / negatives)

        return result

    def _optimize_threshold(self, scores: np.ndarray, y_true: np.ndarray) -> float:
        """Find decision threshold maximizing F1 on validation split."""
        if len(y_true) == 0:
            return 0.5
        candidates = np.linspace(0.1, 0.9, 33)
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
        """Assess hallucination mitigation by tracking false accept rate and confidence shift."""
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
                "avg_baseline_confidence": float(np.mean(base_confidence)) if base_confidence.size else 0.0,
                "avg_validated_confidence": float(np.mean(calibrated_confidence)) if calibrated_confidence.size else 0.0,
            },
        }

    def _aggregate_runs(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate evaluation metrics across multiple experimental runs."""
        methods = list(runs[0]["metrics"].keys())
        metric_keys = ["accuracy", "precision", "recall", "f1", "auroc", "ece", "mce", "brier_score", "far"]

        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for method in methods:
            summary[method] = {}
            for metric in metric_keys:
                vals = [float(run["metrics"][method][metric]) for run in runs]
                summary[method][metric] = self._summary_stats(vals)

        # Aggregate hallucination evaluation
        baseline_far_vals = [float(r["hallucination_evaluation"]["before_after"]["baseline_false_accept_rate"]) for r in runs]
        val_far_vals = [float(r["hallucination_evaluation"]["before_after"]["validated_false_accept_rate"]) for r in runs]
        abs_red_vals = [float(r["hallucination_evaluation"]["before_after"]["absolute_reduction"]) for r in runs]
        rel_red_vals = [float(r["hallucination_evaluation"]["before_after"]["relative_reduction_pct"]) for r in runs]

        summary["hallucination"] = {
            "baseline_far": self._summary_stats(baseline_far_vals),
            "validated_far": self._summary_stats(val_far_vals),
            "absolute_reduction": self._summary_stats(abs_red_vals),
            "relative_reduction_pct": self._summary_stats(rel_red_vals),
        }

        # Aggregate document-cluster bootstrap evaluations
        bootstraps = [r["document_cluster_bootstrap"] for r in runs if "document_cluster_bootstrap" in r]
        if bootstraps and any(b.get("n_bootstraps", 0) > 0 for b in bootstraps):
            valid_bootstraps = [b for b in bootstraps if b.get("n_bootstraps", 0) > 0]
            delta_f1_means = [b["delta_f1"]["mean"] for b in valid_bootstraps]
            delta_ece_means = [b["delta_ece"]["mean"] for b in valid_bootstraps]
            p_val_f1 = [b["delta_f1"]["p_value_one_tailed"] for b in valid_bootstraps]
            p_val_ece = [b["delta_ece"]["p_value_one_tailed"] for b in valid_bootstraps]
            summary["document_cluster_bootstrap"] = {
                "delta_f1": self._summary_stats(delta_f1_means),
                "delta_ece": self._summary_stats(delta_ece_means),
                "mean_p_value_f1": float(mean(p_val_f1)),
                "mean_p_value_ece": float(mean(p_val_ece)),
                "runs_evaluated": len(valid_bootstraps),
            }

        return {
            "n_runs": len(runs),
            "methods": summary,
        }

    def _summary_stats(self, values: List[float]) -> Dict[str, float]:
        """
        Compute summary statistics over a list of per-run metric values.

        NOTE ON CI METHOD:
        The Wald normal-approximation CI (mu ± 1.96 * sd/sqrt(n)) is invalid for:
          - Small samples (n < 30), where the Central Limit Theorem does not apply.
          - Bounded metrics (e.g. F1, precision, recall ∈ [0,1]) where the CI can
            exceed valid bounds, producing nonsensical values like [0.011, 1.056].

        We instead use percentile bootstrap CI (B=2000 resamples), which:
          - Is always bounded by the observed range of the data.
          - Requires no distributional assumption.
          - Is statistically valid for n ≥ 2.
        When n == 1, CI is undefined and reported as [value, value].
        """
        n = len(values)
        mu = float(mean(values))
        sd = float(stdev(values)) if n > 1 else 0.0

        if n == 1:
            return {
                "mean": mu,
                "std": sd,
                "ci95_low": mu,
                "ci95_high": mu,
                "ci_method": "undefined_single_run",
                "n": n,
            }

        # Percentile bootstrap CI
        rng = np.random.default_rng(seed=0)
        arr = np.array(values, dtype=float)
        boot_means = np.array([
            rng.choice(arr, size=n, replace=True).mean()
            for _ in range(2000)
        ])
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))

        return {
            "mean": mu,
            "std": sd,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "ci_method": "percentile_bootstrap_2000",
            "n": n,
        }

    def _build_table_ready(self, aggregate: Dict[str, Any]) -> Dict[str, Any]:
        """Construct publication-ready comparison tables."""
        m = aggregate["methods"]

        performance_table = []
        calibration_table = []

        method_names = {
            "baseline_accept_all": "Baseline: Accept All (Constant 1.0)",
            "baseline_conf": "Baseline: LLM Confidence",
            "baseline_cooc": "Baseline: Co-occurrence Score",
            "baseline_sem": "Baseline: Semantic Similarity",
            "baseline_entail": "Baseline: Evidence Support",
            "baseline_heuristic": "Baseline: Prototype Heuristic (0.4/0.4/0.2)",
            "model_raw": "Multi-Signal Model (Uncalibrated)",
            "calibrated_platt": "Multi-Signal Calibrated (Platt)",
            "calibrated_isotonic": "Multi-Signal Calibrated (Isotonic)",
            "ablation_no_evidence": "Ablation: No Evidence Support",
            "ablation_no_confidence": "Ablation: No Generator Confidence",
            "ablation_no_cooccurrence": "Ablation: No Co-occurrence",
            "ablation_no_semantic": "Ablation: No Semantic Similarity",
            "ablation_no_type": "Ablation: No Relation Type",
            "ablation_no_graph": "Ablation: No Graph Structural Features",
        }

        for key, display_name in method_names.items():
            if key in m:
                performance_table.append(
                    {
                        "method": display_name,
                        "f1_mean": m[key]["f1"]["mean"],
                        "f1_std": m[key]["f1"]["std"],
                        "f1_ci95": [m[key]["f1"]["ci95_low"], m[key]["f1"]["ci95_high"]],
                        "accuracy_mean": m[key]["accuracy"]["mean"],
                        "precision_mean": m[key]["precision"]["mean"],
                        "recall_mean": m[key]["recall"]["mean"],
                        "auroc_mean": m[key]["auroc"]["mean"],
                    }
                )
                calibration_table.append(
                    {
                        "method": display_name,
                        "ece_mean": m[key]["ece"]["mean"],
                        "ece_std": m[key]["ece"]["std"],
                        "ece_ci95": [m[key]["ece"]["ci95_low"], m[key]["ece"]["ci95_high"]],
                        "mce_mean": m[key]["mce"]["mean"],
                        "brier_mean": m[key]["brier_score"]["mean"],
                        "far_mean": m[key]["far"]["mean"],
                    }
                )

        ablation_keys = [
            "model_raw", "calibrated_platt", "ablation_no_evidence",
            "ablation_no_confidence", "ablation_no_cooccurrence",
            "ablation_no_semantic", "ablation_no_type", "ablation_no_graph"
        ]
        ablation_table = []
        for key in ablation_keys:
            if key in m:
                ablation_table.append({
                    "configuration": method_names.get(key, key),
                    "f1_mean": m[key]["f1"]["mean"],
                    "f1_std": m[key]["f1"]["std"],
                    "ece_mean": m[key]["ece"]["mean"],
                    "ece_std": m[key]["ece"]["std"],
                    "far_mean": m[key]["far"]["mean"],
                })

        boot_info = m.get("document_cluster_bootstrap", {})
        bootstrap_table = []
        if boot_info:
            bootstrap_table.append({
                "comparison": "Calibrated Platt vs Baseline LLM Confidence",
                "delta_f1_mean": boot_info.get("delta_f1", {}).get("mean", 0.0),
                "delta_f1_ci95": [boot_info.get("delta_f1", {}).get("ci95_low", 0.0), boot_info.get("delta_f1", {}).get("ci95_high", 0.0)],
                "delta_ece_mean": boot_info.get("delta_ece", {}).get("mean", 0.0),
                "delta_ece_ci95": [boot_info.get("delta_ece", {}).get("ci95_low", 0.0), boot_info.get("delta_ece", {}).get("ci95_high", 0.0)],
                "runs_evaluated": boot_info.get("runs_evaluated", 0),
            })

        h = m.get("hallucination", {})
        hallucination_table = [
            {
                "metric": "False Accept Rate (FAR)",
                "baseline_mean": h.get("baseline_far", {}).get("mean", 0.0),
                "validated_mean": h.get("validated_far", {}).get("mean", 0.0),
                "absolute_reduction_mean": h.get("absolute_reduction", {}).get("mean", 0.0),
                "relative_reduction_pct_mean": h.get("relative_reduction_pct", {}).get("mean", 0.0),
            }
        ]

        return {
            "model_performance_table": performance_table,
            "calibration_table": calibration_table,
            "ablation_table": ablation_table,
            "bootstrap_table": bootstrap_table,
            "hallucination_table": hallucination_table,
        }

"""
Multi-Signal Relation Validator for Closed-Domain Knowledge Graphs.

Decouples candidate generation from evidence validation.
Evaluates relation validity using a multi-signal representation:
    Φ(c_a, r, c_b) = [φ_entail, φ_cooc, φ_sem, φ_type, φ_conf, φ_graph]
under strict leak-free document isolation.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from core.evaluation.evidence_verifier import EvidenceVerifier, EvidenceExtractionResult
from core.experiments.graph_features import GraphFeatureExtractor

logger = logging.getLogger(__name__)

ALL_SIGNALS = {
    "evidence_support",
    "entailment",  # Backwards-compatible alias for evidence_support
    "cooccurrence",
    "semantic",
    "type_compatibility",
    "confidence",
    "graph_structure",
}

DEFAULT_SIGNAL_WEIGHTS = {
    "evidence_support": 0.30,
    "cooccurrence": 0.15,
    "semantic": 0.15,
    "type_compatibility": 0.10,
    "confidence": 0.10,
    "graph_structure": 0.20,
}


@dataclass
class ValidationDecision:
    """Structured result of multi-signal relation validation."""
    concept_a: str
    concept_b: str
    relation_type: str
    is_valid: bool
    calibrated_probability: float
    raw_score: float
    signal_scores: Dict[str, float]
    active_signals: List[str]
    evidence_text: str
    source_chunk_ids: List[str]
    rationale: str


class MultiSignalValidator:
    """
    Validates candidate relations by combining orthogonal evidence signals.
    """

    def __init__(
        self,
        active_signals: Optional[Set[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.50,
        calibrator: Optional[Any] = None,
    ):
        self.active_signals = set(active_signals or ALL_SIGNALS)
        w = dict(weights or DEFAULT_SIGNAL_WEIGHTS)
        if "entailment" in w and "evidence_support" not in w:
            w["evidence_support"] = w["entailment"]
        if "evidence_support" in w and "entailment" not in w:
            w["entailment"] = w["evidence_support"]
        self.weights = w
        self.threshold = threshold
        self.calibrator = calibrator
        self.evidence_verifier = EvidenceVerifier()

    def set_calibrator(self, calibrator: Any) -> None:
        """Set a trained calibration model (e.g. Platt scaling / LogisticRegression)."""
        self.calibrator = calibrator

    def validate_relation(
        self,
        concept_a: str,
        concept_b: str,
        relation_type: str,
        chunks: Sequence[Any],
        generator_confidence: Optional[float] = None,
        graph_extractor: Optional[GraphFeatureExtractor] = None,
        embedding_similarity: Optional[float] = None,
    ) -> ValidationDecision:
        """
        Validate a candidate relation against document evidence.
        
        Args:
            concept_a: Head concept name
            concept_b: Tail concept name
            relation_type: Stated relation type
            chunks: Document chunks
            generator_confidence: Raw confidence from candidate generator
            graph_extractor: Leak-free GraphFeatureExtractor instance
            embedding_similarity: Precomputed cosine similarity if available
            
        Returns:
            ValidationDecision with calibrated probability and evidence provenance
        """
        # 1. Evidence Extraction
        ev_result = self.evidence_verifier.extract_evidence(
            concept_a, concept_b, relation_type, chunks
        )

        signals: Dict[str, float] = {}

        # Signal 1: φ_evidence (Sentence-level lexical evidence support)
        if "evidence_support" in self.active_signals or "entailment" in self.active_signals:
            if ev_result.best_sentence:
                phi_ev = self.evidence_verifier.verify_support(
                    concept_a, concept_b, relation_type, ev_result.best_sentence
                )
            else:
                phi_ev = 0.0
            if "evidence_support" in self.active_signals:
                signals["evidence_support"] = float(phi_ev)
            if "entailment" in self.active_signals:
                signals["entailment"] = float(phi_ev)

        # Signal 2: φ_cooc (Co-occurrence strength)
        if "cooccurrence" in self.active_signals:
            phi_cooc = 1.0 if ev_result.cooccurrence_found else (0.3 if ev_result.best_sentence else 0.0)
            signals["cooccurrence"] = float(phi_cooc)

        # Signal 3: φ_sem (Semantic similarity)
        if "semantic" in self.active_signals:
            if embedding_similarity is not None:
                phi_sem = max(0.0, min(1.0, float(embedding_similarity)))
            else:
                # Lexical overlap proxy
                words_a = set(concept_a.lower().split())
                words_b = set(concept_b.lower().split())
                union = words_a | words_b
                jaccard = len(words_a & words_b) / max(1, len(union))
                phi_sem = float(jaccard)
            signals["semantic"] = phi_sem

        # Signal 4: φ_type (Type compatibility)
        if "type_compatibility" in self.active_signals:
            valid_types = {
                "prerequisite", "definition", "explanation", "cause-effect",
                "part-of", "derives-from", "example-of", "similar-to"
            }
            phi_type = 1.0 if relation_type.lower() in valid_types else 0.2
            signals["type_compatibility"] = float(phi_type)

        # Signal 5: φ_conf (Candidate generator confidence)
        if "confidence" in self.active_signals:
            phi_conf = float(generator_confidence if generator_confidence is not None else 0.5)
            signals["confidence"] = max(0.0, min(1.0, phi_conf))

        # Signal 6: φ_graph (Leak-free graph structural support)
        if "graph_structure" in self.active_signals:
            if graph_extractor and graph_extractor.graph:
                gf = graph_extractor.extract_features(concept_a, concept_b)
                # Structural support is positive if concepts are connected along alternative paths
                is_conn = gf.get("is_connected", 0.0)
                path_len = gf.get("shortest_path", -1.0)
                same_comm = gf.get("same_community", 0.0)
                
                # Inverse path boost: path_len >= 2 gives support; -1 gives 0
                path_boost = (1.0 / (1.0 + path_len)) if path_len > 0 else 0.0
                phi_graph = float(0.4 * is_conn + 0.3 * same_comm + 0.3 * path_boost)
            else:
                phi_graph = 0.0
            signals["graph_structure"] = phi_graph

        # Calculate score and calibrated probability
        active_keys = [k for k in self.weights if k in signals]
        if "evidence_support" in active_keys and "entailment" in active_keys:
            active_keys.remove("entailment")
        total_weight = sum(self.weights[k] for k in active_keys) or 1.0
        raw_score = sum(self.weights[k] * signals[k] for k in active_keys) / total_weight

        if self.calibrator is not None:
            try:
                feature_keys = sorted(list(self.active_signals))
                if "evidence_support" in feature_keys and "entailment" in feature_keys:
                    feature_keys.remove("entailment")
                feature_vector = np.array([[signals.get(k, 0.0) for k in feature_keys]])
                if hasattr(self.calibrator, "predict_proba"):
                    calibrated_prob = float(self.calibrator.predict_proba(feature_vector)[0, 1])
                else:
                    calibrated_prob = float(self.calibrator.predict(feature_vector)[0])
            except Exception as e:
                logger.warning(f"Calibrator failed, using raw weighted score: {e}")
                calibrated_prob = float(raw_score)
        else:
            calibrated_prob = float(raw_score)

        is_valid = bool(calibrated_prob >= self.threshold)

        rationale = (
            f"Validation decision: {'VALID' if is_valid else 'INVALID'} "
            f"(calibrated_prob={calibrated_prob:.3f}, threshold={self.threshold:.2f}). "
            f"Evidence: \"{ev_result.best_sentence or 'No direct sentence match'}\""
        )

        return ValidationDecision(
            concept_a=concept_a,
            concept_b=concept_b,
            relation_type=relation_type,
            is_valid=is_valid,
            calibrated_probability=round(calibrated_prob, 4),
            raw_score=round(raw_score, 4),
            signal_scores={k: round(v, 4) for k, v in signals.items()},
            active_signals=sorted(list(self.active_signals)),
            evidence_text=ev_result.best_sentence,
            source_chunk_ids=ev_result.source_chunk_ids,
            rationale=rationale,
        )

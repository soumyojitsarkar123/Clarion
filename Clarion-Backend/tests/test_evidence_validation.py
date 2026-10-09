"""
Tests for EvidenceVerifier and MultiSignalValidator in closed-domain document graphs.
"""

import networkx as nx
import pytest

from core.evaluation.evidence_verifier import EvidenceVerifier
from core.evaluation.multi_signal_validator import MultiSignalValidator
from core.experiments.graph_features import GraphFeatureExtractor


@pytest.fixture
def sample_chunks():
    return [
        {
            "id": "chk-01",
            "text": (
                "Hypothesis testing is a statistical method for making inferences about population parameters. "
                "Formulating the null hypothesis is a prerequisite before calculating the test statistic. "
                "The significance level controls the probability of a Type I error."
            ),
        },
        {
            "id": "chk-02",
            "text": (
                "Confidence intervals are derived from the sampling distribution of the sample mean. "
                "The standard normal distribution is used when the population variance is known."
            ),
        },
    ]


@pytest.fixture
def sample_graph():
    G = nx.DiGraph()
    G.add_node("c-1", label="Null Hypothesis", node_type="concept")
    G.add_node("c-2", label="Hypothesis Testing", node_type="concept")
    G.add_node("c-3", label="Test Statistic", node_type="concept")
    # Hierarchy connection
    G.add_edge("c-1", "c-3", relation_type="prerequisite")
    G.add_edge("c-3", "c-2", relation_type="explanation")
    return G


def test_evidence_verifier_extracts_best_sentence(sample_chunks):
    """Verify that EvidenceVerifier identifies the precise co-occurring sentence."""
    verifier = EvidenceVerifier()
    res = verifier.extract_evidence(
        concept_a="Null Hypothesis",
        concept_b="Hypothesis Testing",
        relation_type="prerequisite",
        chunks=sample_chunks,
    )

    assert res.cooccurrence_found is True
    assert "Formulating the null hypothesis is a prerequisite" in res.best_sentence
    assert "chk-01" in res.source_chunk_ids
    assert res.lexical_support_score > 0.60


def test_evidence_verifier_support_scoring():
    """Verify entailment / support scoring logic."""
    verifier = EvidenceVerifier()
    sentence = "Backpropagation calculates gradients which gradient descent requires to update weights."

    score_valid = verifier.verify_support(
        concept_a="Backpropagation",
        concept_b="Gradient Descent",
        relation_type="prerequisite",
        evidence_text=sentence,
    )
    assert score_valid >= 0.70

    # Unrelated sentence
    score_unrelated = verifier.verify_support(
        concept_a="Random Forest",
        concept_b="Gradient Descent",
        relation_type="prerequisite",
        evidence_text=sentence,
    )
    assert score_unrelated <= 0.20


def test_multi_signal_validator_full_pipeline(sample_chunks, sample_graph):
    """Verify MultiSignalValidator produces complete decision with all signals."""
    graph_extractor = GraphFeatureExtractor(sample_graph)
    validator = MultiSignalValidator()

    decision = validator.validate_relation(
        concept_a="Null Hypothesis",
        concept_b="Hypothesis Testing",
        relation_type="prerequisite",
        chunks=sample_chunks,
        generator_confidence=0.85,
        graph_extractor=graph_extractor,
        embedding_similarity=0.72,
    )

    assert decision.is_valid is True
    assert decision.calibrated_probability >= 0.50
    assert "entailment" in decision.signal_scores
    assert "graph_structure" in decision.signal_scores
    assert "cooccurrence" in decision.signal_scores
    assert decision.evidence_text != ""
    assert "chk-01" in decision.source_chunk_ids
    assert "VALID" in decision.rationale


def test_multi_signal_validator_ablation_support(sample_chunks):
    """Verify that individual signals can be cleanly ablated."""
    # Ablation 1: Only generator confidence (Baseline 1)
    conf_only_validator = MultiSignalValidator(active_signals={"confidence"})
    decision_conf = conf_only_validator.validate_relation(
        concept_a="Alpha",
        concept_b="Beta",
        relation_type="prerequisite",
        chunks=sample_chunks,
        generator_confidence=0.90,
    )
    assert decision_conf.active_signals == ["confidence"]
    assert set(decision_conf.signal_scores.keys()) == {"confidence"}
    assert decision_conf.calibrated_probability == 0.90

    # Ablation 2: Only entailment (Baseline 4)
    entail_validator = MultiSignalValidator(active_signals={"entailment"})
    decision_entail = entail_validator.validate_relation(
        concept_a="Null Hypothesis",
        concept_b="Hypothesis Testing",
        relation_type="prerequisite",
        chunks=sample_chunks,
        generator_confidence=0.10,  # low generator confidence ignored
    )
    assert decision_entail.active_signals == ["entailment"]
    assert "entailment" in decision_entail.signal_scores
    assert decision_entail.is_valid is True  # evidence text strongly supports it


def test_evidence_verifier_hard_case_1_negation():
    """Hard Case 1: Both concepts occur but relation is explicitly negated."""
    verifier = EvidenceVerifier()
    text = "Linear regression is not a prerequisite for understanding decision trees."
    score = verifier.verify_support(
        concept_a="Linear Regression",
        concept_b="Decision Trees",
        relation_type="prerequisite",
        evidence_text=text,
    )
    assert score == 0.0, f"Expected 0.0 for negated assertion, got {score}"


def test_evidence_verifier_hard_case_2_direction_inversion():
    """Hard Case 2: Both concepts occur but relation direction is reversed."""
    verifier = EvidenceVerifier()
    # Candidate asserts A -> B (A is prerequisite for B)
    # But text asserts "A requires B" (B is prerequisite for A)
    text = "Gradient descent requires calculating derivatives first."
    score = verifier.verify_support(
        concept_a="Gradient Descent",
        concept_b="Derivatives",
        relation_type="prerequisite",
        evidence_text=text,
    )
    # Should detect that A requires B, meaning direction A -> B as prerequisite is inverted
    assert score <= 0.15, f"Expected inversion penalty (<= 0.15), got {score}"


def test_evidence_verifier_hard_case_3_unrelated_clauses():
    """Hard Case 3: Both concepts occur in unrelated contrasting clauses."""
    verifier = EvidenceVerifier()
    text = "While gradient descent was analyzed in Chapter 2, random forests are discussed in Chapter 9."
    score = verifier.verify_support(
        concept_a="Gradient Descent",
        concept_b="Random Forests",
        relation_type="prerequisite",
        evidence_text=text,
    )
    assert score <= 0.10, f"Expected low score for contrasting clauses without predicate, got {score}"


def test_evidence_verifier_hard_case_4_cross_sentence():
    """Hard Case 4: One concept occurs only in another sentence."""
    verifier = EvidenceVerifier()
    text = "Gradient descent optimizes neural network parameters. Backpropagation is an efficient algorithm."
    # Backpropagation only in sentence 2, Neural Network only in sentence 1
    score = verifier.verify_support(
        concept_a="Backpropagation",
        concept_b="Neural Network Parameters",
        relation_type="prerequisite",
        evidence_text="Gradient descent optimizes neural network parameters.",
    )
    assert score == 0.0, f"Expected 0.0 when concept is missing from sentence, got {score}"


def test_evidence_verifier_hard_case_5_explicitly_stated():
    """Hard Case 5: Relation is explicitly and directly stated."""
    verifier = EvidenceVerifier()
    text = "Matrix multiplication is a prerequisite before computing eigenvalues."
    score = verifier.verify_support(
        concept_a="Matrix Multiplication",
        concept_b="Eigenvalues",
        relation_type="prerequisite",
        evidence_text=text,
    )
    assert score >= 0.70, f"Expected high score (>= 0.70) for explicit relation, got {score}"


def test_evidence_verifier_hard_case_6_contradiction():
    """Hard Case 6: Evidence explicitly contradicts the relation."""
    verifier = EvidenceVerifier()
    text = "Empirical studies contradict the assertion that activation functions cause exploding gradients."
    score = verifier.verify_support(
        concept_a="Activation Functions",
        concept_b="Exploding Gradients",
        relation_type="cause-effect",
        evidence_text=text,
    )
    assert score == 0.0, f"Expected 0.0 for contradicted relation, got {score}"


def test_evidence_verifier_hard_case_7_wrong_relation_type():
    """Hard Case 7: Text asserts an incompatible rival relation type."""
    verifier = EvidenceVerifier()
    # Text describes similarity, candidate claims cause-effect
    text = "Batch normalization is analogous to layer normalization in modern transformers."
    score = verifier.verify_support(
        concept_a="Batch Normalization",
        concept_b="Layer Normalization",
        relation_type="cause-effect",
        evidence_text=text,
    )
    assert score <= 0.30, f"Expected penalty for incompatible relation type match, got {score}"


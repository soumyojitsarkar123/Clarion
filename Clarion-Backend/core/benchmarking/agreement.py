"""
Inter-Annotator Agreement Engine for Knowledge Graph Relation Adjudication.

Implements standard psychometric and linguistic agreement metrics:
- Observed Agreement (Po)
- Cohen's Kappa (kappa) for pairs of annotators
- Fleiss' Kappa (kappa_fleiss) for multiple annotators (m >= 2)
- Category-level breakdown (VALID, INVALID, AMBIGUOUS)
- Disagreement extraction for adjudication committee review
- Landis & Koch (1977) interpretation scale
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


@dataclass
class AgreementMetrics:
    """Statistical summary of inter-annotator agreement."""
    metric_name: str
    observed_agreement: float
    expected_agreement: float
    kappa: float
    standard_error: Optional[float] = None
    confidence_interval_95: Optional[Tuple[float, float]] = None
    interpretation: str = "Unknown"
    num_items: int = 0
    num_raters: int = 0
    categories: List[str] = field(default_factory=list)
    confusion_matrix: Optional[Dict[str, Dict[str, int]]] = None
    disagreement_indices: List[str] = field(default_factory=list)


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) scale for interpreting kappa coefficients."""
    if math.isnan(kappa):
        return "Undefined (No Variation)"
    if kappa < 0.0:
        return "Poor (Less than chance agreement)"
    elif kappa <= 0.20:
        return "Slight agreement"
    elif kappa <= 0.40:
        return "Fair agreement"
    elif kappa <= 0.60:
        return "Moderate agreement"
    elif kappa <= 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def compute_cohens_kappa(
    rater_a: Sequence[str],
    rater_b: Sequence[str],
    item_ids: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
) -> AgreementMetrics:
    """
    Compute Cohen's Kappa for two raters on categorical labels.
    
    Args:
        rater_a: Labels assigned by rater A
        rater_b: Labels assigned by rater B
        item_ids: Optional identifiers for the items
        categories: Optional explicit universe of categories
        
    Returns:
        AgreementMetrics dataclass
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"Rater lists must be of equal length: {len(rater_a)} vs {len(rater_b)}")
    
    n = len(rater_a)
    if n == 0:
        return AgreementMetrics(
            metric_name="Cohen's Kappa",
            observed_agreement=0.0,
            expected_agreement=0.0,
            kappa=0.0,
            num_items=0,
            num_raters=2,
            interpretation="No data",
        )

    all_cats = sorted(list(categories or set(rater_a) | set(rater_b)))
    cat_to_idx = {c: i for i, c in enumerate(all_cats)}
    k = len(all_cats)

    # Contingency table
    matrix = [[0 for _ in range(k)] for _ in range(k)]
    disagreements: List[str] = []

    for idx, (a, b) in enumerate(zip(rater_a, rater_b)):
        ia = cat_to_idx.get(a)
        ib = cat_to_idx.get(b)
        if ia is not None and ib is not None:
            matrix[ia][ib] += 1
            if ia != ib:
                disagreements.append(str(item_ids[idx]) if item_ids is not None else str(idx))

    # Observed agreement
    p_o = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement
    row_sums = [sum(matrix[i][j] for j in range(k)) for i in range(k)]
    col_sums = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    p_e = sum((row_sums[i] / n) * (col_sums[i] / n) for i in range(k))

    if p_e >= 1.0 or abs(1.0 - p_e) < 1e-12:
        kappa = 1.0 if p_o == 1.0 else 0.0
        se = 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)
        # Large-sample standard error of Cohen's Kappa
        se_num = p_o * (1.0 - p_o)
        se_den = n * ((1.0 - p_e) ** 2)
        se = math.sqrt(se_num / se_den) if se_den > 0 and se_num >= 0 else 0.0

    ci_low = max(-1.0, kappa - 1.96 * se)
    ci_high = min(1.0, kappa + 1.96 * se)

    confusion_dict = {
        all_cats[i]: {all_cats[j]: matrix[i][j] for j in range(k)}
        for i in range(k)
    }

    return AgreementMetrics(
        metric_name="Cohen's Kappa",
        observed_agreement=round(p_o, 4),
        expected_agreement=round(p_e, 4),
        kappa=round(kappa, 4),
        standard_error=round(se, 4),
        confidence_interval_95=(round(ci_low, 4), round(ci_high, 4)),
        interpretation=interpret_kappa(kappa),
        num_items=n,
        num_raters=2,
        categories=all_cats,
        confusion_matrix=confusion_dict,
        disagreement_indices=disagreements,
    )


def compute_fleiss_kappa(
    ratings_matrix: Sequence[Sequence[str]],
    item_ids: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
) -> AgreementMetrics:
    """
    Compute Fleiss' Kappa for m raters per item across N items.
    
    Args:
        ratings_matrix: N x m matrix of ratings (each row is an item, each entry is a rater's category)
        item_ids: Optional list of item IDs of length N
        categories: Optional predefined categories
        
    Returns:
        AgreementMetrics dataclass
    """
    n = len(ratings_matrix)
    if n == 0:
        return AgreementMetrics(
            metric_name="Fleiss' Kappa",
            observed_agreement=0.0,
            expected_agreement=0.0,
            kappa=0.0,
            num_items=0,
            num_raters=0,
            interpretation="No data",
        )

    m = len(ratings_matrix[0])
    if m < 2:
        raise ValueError(f"Fleiss' Kappa requires at least 2 raters per item, got {m}")

    all_cats = sorted(list(categories or {cat for row in ratings_matrix for cat in row}))
    cat_to_idx = {c: i for i, c in enumerate(all_cats)}
    k = len(all_cats)

    # Build count matrix: n_ij = number of raters who assigned category j to item i
    table = [[0 for _ in range(k)] for _ in range(n)]
    disagreements: List[str] = []

    for i, row in enumerate(ratings_matrix):
        if len(row) != m:
            raise ValueError(f"Item {i} has {len(row)} ratings, expected {m}")
        for rating in row:
            c_idx = cat_to_idx.get(rating)
            if c_idx is not None:
                table[i][c_idx] += 1
        # If not all raters chose the same category, record disagreement
        if max(table[i]) < m:
            disagreements.append(str(item_ids[i]) if item_ids is not None else str(i))

    # P_i = degree of agreement for i-th subject
    # P_i = (sum(n_ij^2) - m) / (m * (m - 1))
    p_i = [
        (sum(table[i][j] ** 2 for j in range(k)) - m) / (m * (m - 1))
        for i in range(n)
    ]
    p_o = sum(p_i) / n

    # p_j = proportion of all assignments to j-th category
    total_assignments = n * m
    p_j = [
        sum(table[i][j] for i in range(n)) / total_assignments
        for j in range(k)
    ]

    p_e = sum(pj ** 2 for pj in p_j)

    if p_e >= 1.0 or abs(1.0 - p_e) < 1e-12:
        kappa = 1.0 if p_o == 1.0 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return AgreementMetrics(
        metric_name="Fleiss' Kappa",
        observed_agreement=round(p_o, 4),
        expected_agreement=round(p_e, 4),
        kappa=round(kappa, 4),
        interpretation=interpret_kappa(kappa),
        num_items=n,
        num_raters=m,
        categories=all_cats,
        disagreement_indices=disagreements,
    )


def extract_disagreements_for_adjudication(
    batches: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Given a map of annotator_id -> batch_data (each containing 'records'),
    identify candidate relations where annotators diverged in their labels.
    
    Returns:
        List of items needing adjudication committee resolution.
    """
    annotators = list(batches.keys())
    if len(annotators) < 2:
        return []

    # Map record_id -> annotator -> record
    items: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for ann_id, batch in batches.items():
        for r in batch.get("records", []):
            rec_id = r["record_id"]
            items[rec_id][ann_id] = r

    disagreements = []
    for rec_id, ann_records in items.items():
        labels = {ann: rec.get("label") for ann, rec in ann_records.items() if rec.get("label")}
        unique_labels = set(labels.values())
        if len(unique_labels) > 1:
            first_rec = next(iter(ann_records.values()))
            disagreements.append({
                "record_id": rec_id,
                "document_id": first_rec.get("document_id"),
                "concept_a": first_rec.get("concept_a"),
                "concept_b": first_rec.get("concept_b"),
                "relation_type": first_rec.get("relation_type"),
                "assigned_labels": labels,
                "chunk_context": first_rec.get("chunk_context"),
                "status": "NEEDS_ADJUDICATION",
            })

    return disagreements

"""
Dataset splitting utilities for controlled experiments.

Provides leak-free document-level isolation (Group splitting) to prevent
intra-document contextual and vocabulary memorization across train, validation,
and test splits.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import logging
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from sklearn.model_selection import KFold, train_test_split

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify_key: Optional[str] = "relation_type"
    seed: int = 42
    split_strategy: str = "document"  # "document" (leak-free) or "random"
    doc_key: str = "document_id"
    min_docs_for_grouping: int = 3

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        if min(self.train_ratio, self.val_ratio, self.test_ratio) <= 0:
            raise ValueError("All split ratios must be > 0")


@dataclass
class SplitResult:
    train: List[dict]
    validation: List[dict]
    test: List[dict]
    config: SplitConfig
    stratified: bool
    document_isolated: bool = False

    @property
    def train_doc_ids(self) -> Set[str]:
        key = self.config.doc_key
        return {r[key] for r in self.train if key in r and r[key] is not None}

    @property
    def val_doc_ids(self) -> Set[str]:
        key = self.config.doc_key
        return {r[key] for r in self.validation if key in r and r[key] is not None}

    @property
    def test_doc_ids(self) -> Set[str]:
        key = self.config.doc_key
        return {r[key] for r in self.test if key in r and r[key] is not None}

    def assert_no_document_leakage(self) -> None:
        """
        Verify mathematical condition:
            train_docs ∩ val_docs == ∅
            train_docs ∩ test_docs == ∅
            val_docs ∩ test_docs == ∅
        """
        train_docs = self.train_doc_ids
        val_docs = self.val_doc_ids
        test_docs = self.test_doc_ids

        leak_train_val = train_docs & val_docs
        leak_train_test = train_docs & test_docs
        leak_val_test = val_docs & test_docs

        if leak_train_val or leak_train_test or leak_val_test:
            raise AssertionError(
                f"Document-level data leakage detected! "
                f"Train/Val overlap: {leak_train_val}; "
                f"Train/Test overlap: {leak_train_test}; "
                f"Val/Test overlap: {leak_val_test}"
            )


def split_by_document(
    records: List[dict], config: SplitConfig, doc_key: Optional[str] = None
) -> SplitResult:
    """
    Split records strictly by document ID.
    All records belonging to a given document are assigned to exactly ONE split.
    
    Guarantees:
        train_docs ∩ val_docs == ∅
        train_docs ∩ test_docs == ∅
        val_docs ∩ test_docs == ∅
    """
    config.validate()
    key = doc_key or config.doc_key

    # Group records by document
    doc_to_records: Dict[str, List[dict]] = defaultdict(list)
    unassigned: List[dict] = []

    for r in records:
        d_id = r.get(key)
        if d_id is not None:
            doc_to_records[str(d_id)].append(r)
        else:
            unassigned.append(r)

    unique_docs = sorted(list(doc_to_records.keys()))
    m = len(unique_docs)

    if m < config.min_docs_for_grouping:
        raise ValueError(
            f"Document-level split requires at least {config.min_docs_for_grouping} distinct documents, "
            f"but found {m}."
        )

    # Deterministic shuffle by seed
    rng = random.Random(config.seed)
    docs_shuffled = list(unique_docs)
    rng.shuffle(docs_shuffled)

    # Compute allocation ensuring at least 1 document per split for m >= 3
    num_test = max(1, int(round(m * config.test_ratio)))
    num_val = max(1, int(round(m * config.val_ratio)))

    # Bound check so train always gets at least 1 document
    if num_test + num_val >= m:
        num_test = 1
        num_val = 1
    num_train = m - num_test - num_val

    test_docs = set(docs_shuffled[:num_test])
    val_docs = set(docs_shuffled[num_test : num_test + num_val])
    train_docs = set(docs_shuffled[num_test + num_val :])

    train_records = [r for d in train_docs for r in doc_to_records[d]]
    val_records = [r for d in val_docs for r in doc_to_records[d]]
    test_records = [r for d in test_docs for r in doc_to_records[d]]

    # If any records lacked a document_id, place them deterministically in train
    if unassigned:
        train_records.extend(unassigned)

    result = SplitResult(
        train=train_records,
        validation=val_records,
        test=test_records,
        config=config,
        stratified=False,
        document_isolated=True,
    )
    result.assert_no_document_leakage()
    return result


def split_train_val_test(records: List[dict], config: SplitConfig) -> SplitResult:
    """
    Split records into train/validation/test sets.
    If config.split_strategy == "document" and at least min_docs_for_grouping distinct
    documents exist, performs leak-free document-level isolation.
    Otherwise, performs record-level splitting with optional stratification.
    """
    config.validate()
    if len(records) < 3:
        raise ValueError("Need at least 3 records for train/validation/test split")

    if config.split_strategy == "document":
        key = config.doc_key
        unique_docs = {r.get(key) for r in records if r.get(key) is not None}
        if len(unique_docs) >= config.min_docs_for_grouping:
            return split_by_document(records, config)
        logger.warning(
            "Requested document-level split, but only %d distinct documents were found "
            "(min required: %d). Falling back to record-level splitting.",
            len(unique_docs),
            config.min_docs_for_grouping,
        )

    # Record-level split fallback / random strategy
    indices = list(range(len(records)))
    strat_labels = _extract_stratification_labels(records, config.stratify_key)
    stratified = False

    test_size = config.test_ratio
    try:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=config.seed,
            stratify=strat_labels if strat_labels is not None else None,
        )
        stratified = strat_labels is not None
    except ValueError:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=config.seed,
            stratify=None,
        )

    val_relative = config.val_ratio / (config.train_ratio + config.val_ratio)
    train_val_labels = None
    if strat_labels is not None:
        train_val_labels = [strat_labels[i] for i in train_val_idx]

    try:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_relative,
            random_state=config.seed,
            stratify=train_val_labels if train_val_labels is not None else None,
        )
        stratified = stratified and train_val_labels is not None
    except ValueError:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_relative,
            random_state=config.seed,
            stratify=None,
        )
        stratified = False

    return SplitResult(
        train=[records[i] for i in train_idx],
        validation=[records[i] for i in val_idx],
        test=[records[i] for i in test_idx],
        config=config,
        stratified=stratified,
        document_isolated=False,
    )


def split_document_kfold(
    records: List[dict],
    n_splits: int = 5,
    seed: int = 42,
    doc_key: str = "document_id",
) -> List[Tuple[List[dict], List[dict]]]:
    """
    K-Fold cross-validation split grouped strictly by document.
    Ensures that for every fold, all records from any given document
    are strictly in either the train fold or the test fold, never both.
    """
    doc_to_records: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        d_id = r.get(doc_key)
        if d_id is not None:
            doc_to_records[str(d_id)].append(r)

    unique_docs = sorted(list(doc_to_records.keys()))
    if len(unique_docs) < n_splits:
        raise ValueError(
            f"Cannot perform {n_splits}-fold document split with only {len(unique_docs)} documents."
        )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    for train_idx, test_idx in kf.split(unique_docs):
        train_docs = {unique_docs[i] for i in train_idx}
        test_docs = {unique_docs[i] for i in test_idx}

        train_records = [r for d in train_docs for r in doc_to_records[d]]
        test_records = [r for d in test_docs for r in doc_to_records[d]]

        folds.append((train_records, test_records))

    return folds


def _extract_stratification_labels(
    records: List[dict], stratify_key: Optional[str]
) -> Optional[List[Any]]:
    if not stratify_key:
        return None
    labels = [r.get(stratify_key) for r in records]
    if any(label is None for label in labels):
        return None
    unique = {}
    for label in labels:
        unique[label] = unique.get(label, 0) + 1
    if len(unique) < 2 or min(unique.values()) < 2:
        return None
    return labels

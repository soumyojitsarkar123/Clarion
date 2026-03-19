"""
Dataset splitting utilities for controlled experiments.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from sklearn.model_selection import train_test_split


@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify_key: Optional[str] = "relation_type"
    seed: int = 42

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


def split_train_val_test(records: List[dict], config: SplitConfig) -> SplitResult:
    """
    Split records into train/validation/test sets with optional stratification.
    """
    config.validate()
    if len(records) < 3:
        raise ValueError("Need at least 3 records for train/validation/test split")

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

    # Normalize validation share relative to remaining train+val pool.
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
    )


def _extract_stratification_labels(
    records: List[dict], stratify_key: Optional[str]
) -> Optional[List[Any]]:
    if not stratify_key:
        return None
    labels = [r.get(stratify_key) for r in records]
    if any(label is None for label in labels):
        return None
    # Need at least 2 classes and at least 2 samples per class for stable stratified split.
    unique = {}
    for label in labels:
        unique[label] = unique.get(label, 0) + 1
    if len(unique) < 2 or min(unique.values()) < 2:
        return None
    return labels

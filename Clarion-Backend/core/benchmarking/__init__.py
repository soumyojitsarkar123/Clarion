"""
Research benchmarking harness for reproducible relation-validation experiments.
"""

from core.benchmarking.splitter import (
    SplitConfig,
    SplitResult,
    split_train_val_test,
    split_by_document,
    split_document_kfold,
)
from core.benchmarking.manifest import (
    RunManifest,
    ExperimentManifest,
    compute_dataset_version,
)
from core.benchmarking.harness import BenchmarkHarness
from core.benchmarking.agreement import (
    AgreementMetrics,
    compute_cohens_kappa,
    compute_fleiss_kappa,
    extract_disagreements_for_adjudication,
)
from core.benchmarking.dataset_validator import (
    GroundTruthValidator,
    DatasetValidationError,
)
from core.benchmarking.gold_dataset_contract import (
    GoldDatasetValidator,
    GoldContractResult,
    RecordViolation,
)

__all__ = [
    "SplitConfig",
    "SplitResult",
    "split_train_val_test",
    "split_by_document",
    "split_document_kfold",
    "RunManifest",
    "ExperimentManifest",
    "compute_dataset_version",
    "BenchmarkHarness",
    "AgreementMetrics",
    "compute_cohens_kappa",
    "compute_fleiss_kappa",
    "extract_disagreements_for_adjudication",
    "GroundTruthValidator",
    "DatasetValidationError",
    "GoldDatasetValidator",
    "GoldContractResult",
    "RecordViolation",
]

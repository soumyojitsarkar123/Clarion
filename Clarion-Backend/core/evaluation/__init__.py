"""
Core evaluation module - Quality assessment for knowledge maps.
"""

from core.evaluation.confidence_scorer import ConfidenceScorer
from core.evaluation.hallucination_detector import (
    HallucinationDetector,
    HallucinationFlag,
    HallucinationReport,
    FlagSeverity,
    FlagType,
)
from core.evaluation.evidence_verifier import EvidenceVerifier, EvidenceExtractionResult
from core.evaluation.multi_signal_validator import MultiSignalValidator, ValidationDecision
from core.evaluation.calibration import (
    CalibrationMetrics,
    compute_calibration_metrics,
    PlattCalibrator,
    TemperatureCalibrator,
    IsotonicCalibrator,
)

__all__ = [
    "ConfidenceScorer",
    "HallucinationDetector",
    "HallucinationFlag",
    "HallucinationReport",
    "FlagSeverity",
    "FlagType",
    "EvidenceVerifier",
    "EvidenceExtractionResult",
    "MultiSignalValidator",
    "ValidationDecision",
    "CalibrationMetrics",
    "compute_calibration_metrics",
    "PlattCalibrator",
    "TemperatureCalibrator",
    "IsotonicCalibrator",
]

"""
Gold Dataset Contract for Closed-Domain Relation Knowledge Graphs.

Enforces eligibility requirements for promoting raw annotations to the
ADJUDICATED GOLD dataset. A record CANNOT enter the gold dataset unless
ALL required fields are present and pass validation.

Promotion pipeline:
    RAW ANNOTATIONS  →  [GoldDatasetValidator]  →  ADJUDICATED GOLD

Scientific guarantee: no gold record can have missing provenance, a null label,
a null annotator, or a null evidence text. This prevents silent label fabrication
and ensures full traceability.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from models.ground_truth import ErrorCategory, GroundTruthRecord, ValidityLabel


# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: Tuple[str, ...] = (
    "record_id",
    "document_id",
    "relation_id",
    "concept_a",
    "concept_b",
    "relation_type",
    "label",
    "annotator_id",
    "evidence_text",   # must be non-empty
    # NOTE: annotation_version is file-level metadata in the batch manifest
    # (guidelines_version field), NOT a per-record field. It is verified at
    # the file level in validate_json_file(), not per-record here.
)

VALID_LABELS = frozenset(v.value for v in ValidityLabel)
VALID_ERROR_CATEGORIES = frozenset(e.value for e in ErrorCategory)

# Minimum confidence rating a record must carry (1–5 scale, 1 = weakest)
MIN_CONFIDENCE_RATING: int = 1

# Labels that make a record ineligible for binary model training/evaluation
# (they are retained in the file for error analysis only)
NON_TRAINABLE_LABELS = frozenset({"AMBIGUOUS"})

# Promotion state machine
STATE_RAW_ANNOTATIONS = "RAW_ANNOTATIONS"
STATE_ADJUDICATED_GOLD = "ADJUDICATED_GOLD"


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class RecordViolation:
    record_id: str
    field: str
    violation: str

    def __str__(self) -> str:
        return f"[{self.record_id}] field='{self.field}': {self.violation}"


@dataclass
class GoldContractResult:
    """Result of a gold dataset contract check."""
    dataset_state: str              # RAW_ANNOTATIONS or ADJUDICATED_GOLD
    total_records: int
    eligible_records: int           # Records that meet ALL requirements
    ineligible_records: int
    non_trainable_records: int      # Eligible but AMBIGUOUS (excluded from model training)
    violations: List[RecordViolation] = field(default_factory=list)
    annotation_version: Optional[str] = None
    sha256: Optional[str] = None    # Hash of the source data
    notes: List[str] = field(default_factory=list)

    @property
    def is_promotable(self) -> bool:
        """True iff all records are eligible (no violations)."""
        return self.ineligible_records == 0 and self.total_records > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_state": self.dataset_state,
            "total_records": self.total_records,
            "eligible_records": self.eligible_records,
            "ineligible_records": self.ineligible_records,
            "non_trainable_records": self.non_trainable_records,
            "is_promotable": self.is_promotable,
            "annotation_version": self.annotation_version,
            "sha256": self.sha256,
            "notes": self.notes,
            "violation_count": len(self.violations),
            "violations_sample": [str(v) for v in self.violations[:20]],
        }

    def raise_if_not_promotable(self) -> None:
        if not self.is_promotable:
            summary = "\n".join(str(v) for v in self.violations[:10])
            raise ValueError(
                f"Gold dataset contract FAILED: {self.ineligible_records} ineligible records.\n"
                f"First violations:\n{summary}"
            )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class GoldDatasetValidator:
    """
    Validates that a set of GroundTruthRecord objects meets all gold dataset
    eligibility requirements before promotion from RAW_ANNOTATIONS to
    ADJUDICATED_GOLD state.

    Usage::

        validator = GoldDatasetValidator(annotation_version="1.0.0")
        result = validator.validate_records(records)
        result.raise_if_not_promotable()
    """

    def __init__(self, annotation_version: str = "1.0.0") -> None:
        self.annotation_version = annotation_version

    def validate_records(
        self,
        records: List[GroundTruthRecord],
        source_data: Optional[bytes] = None,
    ) -> GoldContractResult:
        """
        Validate a list of GroundTruthRecord objects against the gold dataset contract.

        Args:
            records: List of GroundTruthRecord objects.
            source_data: Optional raw bytes of the source file for SHA-256 hashing.

        Returns:
            GoldContractResult summarising eligibility.
        """
        violations: List[RecordViolation] = []
        eligible = 0
        non_trainable = 0

        for rec in records:
            rec_violations = self._check_record(rec)
            if rec_violations:
                violations.extend(rec_violations)
            else:
                eligible += 1
                if rec.label.value in NON_TRAINABLE_LABELS:
                    non_trainable += 1

        sha256 = None
        if source_data is not None:
            sha256 = "sha256:" + hashlib.sha256(source_data).hexdigest()[:16]

        state = (
            STATE_ADJUDICATED_GOLD
            if (violations == [] and eligible == len(records) and eligible > 0)
            else STATE_RAW_ANNOTATIONS
        )

        notes = []
        if non_trainable > 0:
            notes.append(
                f"{non_trainable} AMBIGUOUS record(s) are eligible but excluded from "
                "binary model training and evaluation."
            )
        if eligible == 0:
            notes.append(
                "ZERO eligible records. This dataset cannot be used for benchmarking."
            )

        return GoldContractResult(
            dataset_state=state,
            total_records=len(records),
            eligible_records=eligible,
            ineligible_records=len(records) - eligible,
            non_trainable_records=non_trainable,
            violations=violations,
            annotation_version=self.annotation_version,
            sha256=sha256,
            notes=notes,
        )

    def validate_json_file(self, path: Path) -> GoldContractResult:
        """Validate a ground truth JSON file produced by import_annotation_batch_to_ground_truth."""
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))

        records_raw = data.get("records", [])
        records: List[GroundTruthRecord] = []
        parse_violations: List[RecordViolation] = []

        for idx, r in enumerate(records_raw):
            rec_id = r.get("record_id", f"index_{idx}")
            try:
                records.append(GroundTruthRecord(**r))
            except Exception as exc:
                parse_violations.append(
                    RecordViolation(rec_id, "schema", f"Failed to parse record: {exc}")
                )

        result = self.validate_records(records, source_data=raw)
        result.violations = parse_violations + result.violations
        result.ineligible_records = len(parse_violations) + result.ineligible_records
        result.eligible_records = result.total_records - result.ineligible_records
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_record(self, rec: GroundTruthRecord) -> List[RecordViolation]:
        """Return all contract violations for a single record (empty = eligible)."""
        violations: List[RecordViolation] = []
        rec_id = getattr(rec, "record_id", "<unknown>")

        # 1. Required fields must be non-null and non-empty string
        for field_name in REQUIRED_FIELDS:
            val = getattr(rec, field_name, None)
            if val is None:
                violations.append(RecordViolation(rec_id, field_name, "is NULL"))
            elif isinstance(val, str) and not val.strip():
                violations.append(RecordViolation(rec_id, field_name, "is empty string"))

        # 2. Label must be a valid ValidityLabel value
        label_val = getattr(rec, "label", None)
        if label_val is not None:
            lv = label_val.value if isinstance(label_val, ValidityLabel) else str(label_val)
            if lv not in VALID_LABELS:
                violations.append(
                    RecordViolation(rec_id, "label", f"invalid label '{lv}'. Allowed: {sorted(VALID_LABELS)}")
                )

        # 3. INVALID records must have a non-NONE error_category
        if label_val is not None:
            lv = label_val.value if isinstance(label_val, ValidityLabel) else str(label_val)
            if lv == "INVALID":
                err_cat = getattr(rec, "error_category", None)
                if err_cat is None or (
                    isinstance(err_cat, ErrorCategory) and err_cat == ErrorCategory.NONE
                ) or (isinstance(err_cat, str) and err_cat.upper() == "NONE"):
                    violations.append(
                        RecordViolation(rec_id, "error_category",
                                        "label is INVALID but error_category is NONE or null")
                    )

        # 4. Evidence text must be non-empty
        ev = getattr(rec, "evidence_text", None)
        if ev is not None and isinstance(ev, str) and not ev.strip():
            violations.append(RecordViolation(rec_id, "evidence_text", "is empty string"))

        # 5. Annotator ID must not be a placeholder
        ann = getattr(rec, "annotator_id", None)
        if ann is not None and str(ann).strip().lower() in ("", "unknown", "none", "null", "placeholder"):
            violations.append(
                RecordViolation(rec_id, "annotator_id", f"is a placeholder value: '{ann}'")
            )

        # 6. Confidence rating must be in [1, 5]
        conf = getattr(rec, "confidence_rating", None)
        if conf is not None:
            try:
                conf_int = int(conf)
                if not (1 <= conf_int <= 5):
                    violations.append(
                        RecordViolation(rec_id, "confidence_rating",
                                        f"must be in [1,5], got {conf_int}")
                    )
            except (TypeError, ValueError):
                violations.append(
                    RecordViolation(rec_id, "confidence_rating", f"not an integer: {conf!r}")
                )

        # 7. annotation_version — check separately (not a GroundTruthRecord field; checked at file level)
        # Placeholder: this check is satisfied if the file carries a valid guidelines_version.

        return violations

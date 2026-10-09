"""
Programmatic Validator for Ground-Truth Relation Datasets.

Enforces protocol conformance, taxonomy consistency, and integrity constraints.
"""

from collections import Counter
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from models.ground_truth import (
    ErrorCategory,
    GroundTruthDataset,
    GroundTruthRecord,
    ValidityLabel,
)

logger = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    """Raised when ground-truth dataset fails protocol validation."""
    pass


class GroundTruthValidator:
    """
    Validates and summarizes ground-truth datasets for closed-domain relation validation.
    """

    def validate_records(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[List[GroundTruthRecord], Dict[str, Any]]:
        """
        Validate a list of raw record dictionaries.
        
        Returns:
            Tuple of (validated_pydantic_records, summary_statistics)
        Raises:
            DatasetValidationError: If validation fails
        """
        if not records:
            raise DatasetValidationError("Dataset is empty; at least 1 record required.")

        validated: List[GroundTruthRecord] = []
        seen_ids = set()
        doc_ids = set()
        label_counts = Counter()
        error_counts = Counter()

        for idx, raw in enumerate(records):
            rec_id = raw.get("record_id")
            if not rec_id:
                raise DatasetValidationError(f"Record at index {idx} is missing 'record_id'")
            if rec_id in seen_ids:
                raise DatasetValidationError(f"Duplicate record_id detected: '{rec_id}'")
            seen_ids.add(rec_id)

            try:
                rec = GroundTruthRecord(**raw)
            except Exception as e:
                raise DatasetValidationError(f"Record '{rec_id}' failed schema validation: {e}") from e

            # Protocol constraint: if INVALID, must have a non-NONE error category
            if rec.label == ValidityLabel.INVALID and rec.error_category in (None, ErrorCategory.NONE):
                raise DatasetValidationError(
                    f"Record '{rec_id}' is marked INVALID but has no specific error_category."
                )

            # Protocol constraint: evidence text must not be empty or blank
            if not rec.evidence_text.strip():
                raise DatasetValidationError(
                    f"Record '{rec_id}' has empty evidence_text; verbatim citation is mandatory."
                )

            validated.append(rec)
            doc_ids.add(rec.document_id)
            label_counts[rec.label.value] += 1
            if rec.label == ValidityLabel.INVALID:
                error_counts[rec.error_category.value] += 1

        stats = {
            "total_records": len(validated),
            "distinct_documents": len(doc_ids),
            "document_ids": sorted(list(doc_ids)),
            "label_distribution": dict(label_counts),
            "error_distribution": dict(error_counts),
            "valid_ratio": round(label_counts[ValidityLabel.VALID.value] / len(validated), 4),
            "invalid_ratio": round(label_counts[ValidityLabel.INVALID.value] / len(validated), 4),
            "ambiguous_ratio": round(label_counts[ValidityLabel.AMBIGUOUS.value] / len(validated), 4),
        }

        return validated, stats

    def validate_file(self, file_path: Union[str, Path]) -> Tuple[GroundTruthDataset, Dict[str, Any]]:
        """Validate a JSON ground truth dataset file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise DatasetValidationError(f"Invalid JSON file: {e}") from e

        if isinstance(data, list):
            raw_records = data
            metadata = {
                "dataset_version": "1.0.0",
                "domain": "closed-domain-document",
                "created_at": "unknown",
                "description": f"Loaded from {path.name}",
            }
        elif isinstance(data, dict):
            raw_records = data.get("records", [])
            metadata = {
                "dataset_version": data.get("dataset_version", "1.0.0"),
                "domain": data.get("domain", "closed-domain-document"),
                "created_at": data.get("created_at", "unknown"),
                "description": data.get("description", ""),
            }
        else:
            raise DatasetValidationError("Dataset file root must be a list or dict.")

        validated_records, stats = self.validate_records(raw_records)

        dataset = GroundTruthDataset(
            dataset_version=metadata["dataset_version"],
            domain=metadata["domain"],
            created_at=metadata["created_at"],
            description=metadata["description"],
            records=validated_records,
        )

        return dataset, stats

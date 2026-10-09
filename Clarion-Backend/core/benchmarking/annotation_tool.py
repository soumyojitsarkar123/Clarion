"""
Annotation Tooling for Closed-Domain Knowledge Graph Relation Adjudication.

Provides CLI and API utilities to:
1. Export unannotated candidate batches from relation_dataset.db with stratified sampling.
2. Validate annotated batches against the scientific annotation protocol.
3. Import validated annotations into publication-grade GroundTruthDataset schemas
   and synchronize adjudication status back to SQLite.
"""

import argparse
import datetime
import json
import logging
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from models.ground_truth import (
    ErrorCategory,
    GroundTruthDataset,
    GroundTruthRecord,
    ValidityLabel,
)

logger = logging.getLogger(__name__)

DEFAULT_GUIDELINES_VERSION = "1.0.0"
DEFAULT_INSTRUCTIONS = (
    "Closed-Domain Relation Adjudication Protocol:\n"
    "1. Read the chunk_context containing concept_a and concept_b.\n"
    "2. Determine whether the specified relation_type between concept_a (head) "
    "and concept_b (tail) is strictly supported by the text.\n"
    "3. Set 'label' to 'VALID' if explicitly supported, 'INVALID' if unsupported/contradicted/inverted, "
    "or 'AMBIGUOUS' if evidence is genuinely under-specified.\n"
    "4. If 'INVALID', select an 'error_category' from: "
    "[UNSUPPORTED_ASSERTION, DIRECTION_INVERSION, TYPE_MISMATCH, SPURIOUS_COOCCURRENCE, NOISE_ENTITY].\n"
    "5. Populate 'adjudicated_evidence_text' with the minimal sentence(s) proving the label."
)


def export_annotation_batch(
    db_path: str,
    output_path: str,
    sample_size: Optional[int] = None,
    stratify_by_doc: bool = True,
    stratify_by_type: bool = True,
    min_confidence: float = 0.0,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Export candidate relations from relation_dataset.db into an annotation batch file.
    
    Args:
        db_path: Path to relation_dataset.db
        output_path: Target path for the output JSON template
        sample_size: Maximum total samples to export (None for all eligible candidates)
        stratify_by_doc: Balance candidates across source documents
        stratify_by_type: Balance candidates across relation types
        min_confidence: Minimum LLM generator confidence threshold
        random_seed: Deterministic RNG seed
        
    Returns:
        Summary dict of exported batch metadata
    """
    p_db = Path(db_path)
    if not p_db.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(p_db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT record_id, document_id, relation_id, concept_a, concept_b,
               relation_type, llm_confidence, cooccurrence_score,
               semantic_similarity, chunk_context, source_chunk_ids, is_valid
        FROM relation_dataset
        WHERE llm_confidence >= ?
    """
    cursor.execute(query, (min_confidence,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"No records found in {db_path} matching min_confidence={min_confidence}")

    candidates: List[Dict[str, Any]] = []
    for r in rows:
        # Parse chunk ids
        src_ids = []
        if r["source_chunk_ids"]:
            try:
                parsed = json.loads(r["source_chunk_ids"])
                if isinstance(parsed, list):
                    src_ids = [str(x) for x in parsed]
                else:
                    src_ids = [str(r["source_chunk_ids"])]
            except Exception:
                src_ids = [str(r["source_chunk_ids"])]

        candidates.append({
            "record_id": str(r["record_id"]),
            "document_id": str(r["document_id"]),
            "relation_id": str(r["relation_id"]),
            "concept_a": str(r["concept_a"]),
            "concept_b": str(r["concept_b"]),
            "relation_type": str(r["relation_type"]),
            "llm_confidence": float(r["llm_confidence"]),
            "cooccurrence_score": float(r["cooccurrence_score"] or 0.0),
            "semantic_similarity": float(r["semantic_similarity"] or 0.0),
            "chunk_context": str(r["chunk_context"]),
            "source_chunk_ids": src_ids,
            "existing_db_is_valid": r["is_valid"],
            # Annotation fields to be completed by human annotators:
            "label": None,
            "error_category": None,
            "annotator_id": None,
            "confidence_rating": None,
            "adjudicated_evidence_text": None,
            "notes": None,
        })

    rng = random.Random(random_seed)

    # Perform stratified sampling if requested and sample_size is smaller than total
    if sample_size and sample_size < len(candidates):
        if stratify_by_doc and stratify_by_type:
            groups = defaultdict(list)
            for c in candidates:
                groups[(c["document_id"], c["relation_type"])].append(c)
            sampled: List[Dict[str, Any]] = []
            keys = sorted(list(groups.keys()))
            rng.shuffle(keys)
            while len(sampled) < sample_size and any(groups.values()):
                for k in keys:
                    if groups[k]:
                        rng.shuffle(groups[k])
                        sampled.append(groups[k].pop())
                        if len(sampled) >= sample_size:
                            break
            selected_records = sampled
        else:
            rng.shuffle(candidates)
            selected_records = candidates[:sample_size]
    else:
        selected_records = candidates

    # Sort deterministically for reproducibility
    selected_records.sort(key=lambda x: (x["document_id"], x["record_id"]))

    batch_manifest = {
        "manifest_version": "1.0.0",
        "guidelines_version": DEFAULT_GUIDELINES_VERSION,
        "export_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_records": len(selected_records),
        "README": {
            "purpose": (
                "This file contains candidate knowledge-graph relations extracted from "
                "technical documents. Your task is to decide, for each record, whether "
                "the stated relation between two concepts is supported by the provided text."
            ),
            "field_guide": {
                "concept_a": "The HEAD concept in the relation (e.g. 'Address')",
                "concept_b": "The TAIL concept in the relation (e.g. 'Register')",
                "relation_type": (
                    "The type of relation claimed between concept_a and concept_b. "
                    "Currently always 'explanation' — concept_a explains or describes concept_b."
                ),
                "chunk_context": (
                    "The raw text passage(s) from the source document. This is the ONLY "
                    "evidence you should use to judge the relation. Sentences are separated by '|'."
                ),
                "llm_confidence": (
                    "A score (0–1) assigned by the automated extraction system. "
                    "NOT your annotation confidence — ignore this when making your decision."
                ),
                "cooccurrence_score": "Automated signal — ignore when annotating.",
                "semantic_similarity": "Automated signal — ignore when annotating.",
                "source_chunk_ids": "Internal IDs of source passages — ignore when annotating.",
                "record_id": "Unique identifier for this candidate — do not modify.",
                "document_id": "Source document identifier — do not modify.",
                "existing_db_is_valid": "Always null — no prior label exists. Do not modify."
            },
            "annotation_workflow": [
                "1. Read 'chunk_context' carefully.",
                "2. Ask: does the text explicitly support the claim that concept_a EXPLAINS concept_b?",
                "3. Set 'label' to one of: VALID, INVALID, or AMBIGUOUS (see label_guide below).",
                "4. If INVALID, set 'error_category' to the best-fitting category (see error_guide).",
                "5. Copy the key sentence(s) that justify your decision into 'adjudicated_evidence_text'.",
                "6. Set 'annotator_id' to your initials or assigned annotator code (e.g. 'AM1').",
                "7. Set 'confidence_rating' to 1 (low) – 5 (high) reflecting YOUR confidence.",
                "8. Optionally add free-text notes in 'notes'."
            ],
            "label_guide": {
                "VALID": (
                    "The text explicitly and unambiguously supports the relation. "
                    "There is a clear explanatory or definitional link between concept_a and concept_b."
                ),
                "INVALID": (
                    "The text does NOT support the relation, contradicts it, reverses the direction, "
                    "or the two concepts only co-occur incidentally with no explanatory link."
                ),
                "AMBIGUOUS": (
                    "The text hints at a connection but does not clearly confirm or deny it. "
                    "Use sparingly — only when a reasonable expert would genuinely be uncertain."
                )
            },
            "error_guide": {
                "UNSUPPORTED_ASSERTION": "The claim is made but the text provides no evidence.",
                "DIRECTION_INVERSION": "The relation direction is reversed (B explains A, not A explains B).",
                "TYPE_MISMATCH": "The relation type is wrong — a different relation type fits better.",
                "SPURIOUS_COOCCURRENCE": "The concepts appear together but have no explanatory relationship.",
                "NOISE_ENTITY": "One or both concepts are not meaningful entities (e.g. a noise fragment)."
            },
            "submission": (
                "When complete, return this file with all 'label', 'annotator_id', and "
                "'confidence_rating' fields populated for every record. "
                "Run: python annotation_tool.py validate --file <this_file> to check for errors."
            )
        },
        "sampling_strategy": {
            "requested_sample_size": sample_size,
            "stratified_by_doc": stratify_by_doc,
            "stratified_by_type": stratify_by_type,
            "random_seed": random_seed,
            "min_confidence": min_confidence,
        },
        "instructions": DEFAULT_INSTRUCTIONS,
        "valid_labels": [e.value for e in ValidityLabel],
        "valid_error_categories": [e.value for e in ErrorCategory],
        "records": selected_records,
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(batch_manifest, f, indent=2, ensure_ascii=False)

    summary = {
        "output_path": str(out_p),
        "total_records": len(selected_records),
        "unique_documents": len(set(r["document_id"] for r in selected_records)),
        "unique_relation_types": len(set(r["relation_type"] for r in selected_records)),
    }
    logger.info(f"Exported annotation batch: {summary}")
    return summary


def validate_annotation_batch(file_path: str) -> Dict[str, Any]:
    """
    Validate the completeness and syntactic schema of an annotated batch file.
    
    Args:
        file_path: Path to the annotated JSON file
        
    Returns:
        Validation report dictionary
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    if not records:
        return {"is_valid": False, "error": "No records found in file", "total_records": 0}

    valid_labels = {e.value for e in ValidityLabel}
    valid_errs = {e.value for e in ErrorCategory}

    completed_count = 0
    label_counts = defaultdict(int)
    error_counts = defaultdict(int)
    issues: List[str] = []

    for idx, r in enumerate(records):
        rec_id = r.get("record_id", f"index_{idx}")
        lbl = r.get("label")
        if lbl is None or str(lbl).strip() == "":
            continue

        lbl_str = str(lbl).strip().upper()
        if lbl_str not in valid_labels:
            issues.append(f"Record {rec_id}: invalid label '{lbl}'. Allowed: {list(valid_labels)}")
            continue

        completed_count += 1
        label_counts[lbl_str] += 1

        err_cat = r.get("error_category")
        if lbl_str == ValidityLabel.INVALID.value:
            if not err_cat or str(err_cat).strip() in ("", "NONE"):
                issues.append(f"Record {rec_id}: label is INVALID but error_category is missing or NONE")
            elif str(err_cat).strip().upper() not in valid_errs:
                issues.append(f"Record {rec_id}: invalid error_category '{err_cat}'")
            else:
                error_counts[str(err_cat).strip().upper()] += 1

    total = len(records)
    is_complete = (completed_count == total and len(issues) == 0)

    return {
        "is_valid": len(issues) == 0,
        "is_complete": is_complete,
        "total_records": total,
        "completed_records": completed_count,
        "completion_ratio": round(completed_count / total, 4) if total > 0 else 0.0,
        "label_distribution": dict(label_counts),
        "error_category_distribution": dict(error_counts),
        "issues_count": len(issues),
        "sample_issues": issues[:10],
    }


def import_annotation_batch_to_ground_truth(
    input_path: str,
    output_gold_path: str,
    db_path: Optional[str] = None,
    dataset_version: str = "1.0.0",
    description: str = "Human-adjudicated closed-domain relation gold dataset",
) -> GroundTruthDataset:
    """
    Import fully adjudicated annotations into a publication-grade GroundTruthDataset.
    Optionally syncs adjudicated labels back into SQLite relation_dataset.db.
    """
    val_rep = validate_annotation_batch(input_path)
    if not val_rep["is_valid"]:
        raise ValueError(
            f"Annotation batch has validation issues: {val_rep['sample_issues']}"
        )
    if not val_rep["is_complete"]:
        raise ValueError(
            f"Annotation batch is incomplete ({val_rep['completed_records']}/{val_rep['total_records']} completed). "
            f"Complete all annotations before importing."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_records: List[GroundTruthRecord] = []
    db_updates: List[tuple] = []

    for r in data["records"]:
        lbl = ValidityLabel(str(r["label"]).strip().upper())
        err_cat = (
            ErrorCategory(str(r["error_category"]).strip().upper())
            if r.get("error_category")
            else ErrorCategory.NONE
        )
        ev_text = (
            r.get("adjudicated_evidence_text")
            or r.get("chunk_context")
            or ""
        )
        src_chunks = r.get("source_chunk_ids") or []
        if isinstance(src_chunks, str):
            src_chunks = [src_chunks]

        gt_rec = GroundTruthRecord(
            record_id=r["record_id"],
            document_id=r["document_id"],
            relation_id=r.get("relation_id") or r["record_id"],
            concept_a=r["concept_a"],
            concept_b=r["concept_b"],
            relation_type=r["relation_type"],
            label=lbl,
            error_category=err_cat,
            evidence_text=ev_text,
            source_chunk_ids=src_chunks,
            annotator_id=str(r.get("annotator_id") or "human_adjudicator"),
            confidence_rating=int(r.get("confidence_rating") or 5),
            notes=r.get("notes"),
        )
        gt_records.append(gt_rec)

        if db_path:
            is_valid_int = 1 if lbl == ValidityLabel.VALID else 0
            db_updates.append((is_valid_int, r["record_id"]))

    dataset = GroundTruthDataset(
        dataset_version=dataset_version,
        domain="closed-domain-technical",
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description=description,
        records=gt_records,
    )

    out_p = Path(output_gold_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(dataset.model_dump_json(indent=2))

    # Sync to SQLite if requested
    if db_path and Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.executemany(
            "UPDATE relation_dataset SET is_valid = ? WHERE record_id = ?",
            db_updates,
        )
        conn.commit()
        conn.close()
        logger.info(f"Updated {len(db_updates)} records in SQLite {db_path}")

    return dataset


def export_overlap_batch(
    primary_batch_path: str,
    output_path: str,
    overlap_size: int = 50,
    random_seed: int = 42,
    annotator_b_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export an independent overlapping subset of candidate relations for Annotator B.
    
    Stratifies selection across source documents from the primary batch (Batch 1).
    Resets all annotation fields to null to guarantee independent annotation.
    """
    p = Path(primary_batch_path)
    if not p.exists():
        raise FileNotFoundError(f"Primary batch file not found: {primary_batch_path}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    if not records:
        raise ValueError(f"No records found in {primary_batch_path}")

    # Stratified sampling across document_ids
    rng = random.Random(random_seed)
    doc_groups = defaultdict(list)
    for r in records:
        doc_groups[r["document_id"]].append(r)

    sampled = []
    keys = sorted(list(doc_groups.keys()))
    rng.shuffle(keys)
    while len(sampled) < overlap_size and any(doc_groups.values()):
        for k in keys:
            if doc_groups[k]:
                rng.shuffle(doc_groups[k])
                sampled.append(doc_groups[k].pop())
                if len(sampled) >= overlap_size:
                    break

    # Sort deterministically
    sampled.sort(key=lambda x: (x["document_id"], x["record_id"]))

    # Prepare records for Annotator B (clean slate)
    overlap_records = []
    for r in sampled:
        overlap_records.append({
            "record_id": r["record_id"],
            "document_id": r["document_id"],
            "relation_id": r["relation_id"],
            "concept_a": r["concept_a"],
            "concept_b": r["concept_b"],
            "relation_type": r["relation_type"],
            "llm_confidence": r.get("llm_confidence", 0.5),
            "cooccurrence_score": r.get("cooccurrence_score", 0.0),
            "semantic_similarity": r.get("semantic_similarity", 0.0),
            "chunk_context": r["chunk_context"],
            "source_chunk_ids": r.get("source_chunk_ids", []),
            "existing_db_is_valid": None,
            "label": None,
            "error_category": None,
            "annotator_id": annotator_b_id,
            "confidence_rating": None,
            "adjudicated_evidence_text": None,
            "notes": None,
        })

    readme = dict(data.get("README", {}))
    readme["overlap_info"] = {
        "annotator_role": "Annotator B (Independent Overlap Rater)",
        "source_batch": str(p.name),
        "total_overlap_items": len(overlap_records),
        "instructions": (
            "Annotate each candidate independently without consulting Annotator A. "
            "Your annotations will be compared with Annotator A's to measure inter-annotator agreement."
        ),
    }

    manifest = {
        "manifest_version": "1.0.0",
        "guidelines_version": data.get("guidelines_version", DEFAULT_GUIDELINES_VERSION),
        "export_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_records": len(overlap_records),
        "batch_type": "INDEPENDENT_OVERLAP_ANNOTATOR_B",
        "primary_batch_source": str(p),
        "README": readme,
        "sampling_strategy": {
            "overlap_size": overlap_size,
            "random_seed": random_seed,
            "stratified_by_doc": True,
        },
        "instructions": data.get("instructions", DEFAULT_INSTRUCTIONS),
        "valid_labels": data.get("valid_labels", [e.value for e in ValidityLabel]),
        "valid_error_categories": data.get("valid_error_categories", [e.value for e in ErrorCategory]),
        "records": overlap_records,
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    summary = {
        "output_path": str(out_p),
        "total_records": len(overlap_records),
        "unique_documents": len(set(r["document_id"] for r in overlap_records)),
        "primary_source": str(p),
    }
    logger.info(f"Exported overlap annotation batch: {summary}")
    return summary


def compute_agreement_between_files(file_a: str, file_b: str) -> Dict[str, Any]:
    """
    Compute inter-annotator agreement between two annotation batch files.
    
    If either file contains 0 completed human annotations or no overlap exists,
    reports 'HUMAN ANNOTATION PENDING — AGREEMENT NOT COMPUTED'.
    """
    from core.benchmarking.agreement import compute_cohens_kappa

    p_a, p_b = Path(file_a), Path(file_b)
    if not p_a.exists():
        raise FileNotFoundError(f"File A not found: {file_a}")
    if not p_b.exists():
        raise FileNotFoundError(f"File B not found: {file_b}")

    with open(p_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(p_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    map_a = {r["record_id"]: r for r in data_a.get("records", []) if r.get("label")}
    map_b = {r["record_id"]: r for r in data_b.get("records", []) if r.get("label")}

    common_ids = sorted(list(set(map_a.keys()) & set(map_b.keys())))

    # Guard: if human annotations are not yet completed
    if len(common_ids) == 0:
        msg = "HUMAN ANNOTATION PENDING — AGREEMENT NOT COMPUTED"
        return {
            "status": msg,
            "message": "Human annotations are not yet present in one or both files. Agreement cannot be calculated without genuine independent labels.",
            "completed_in_a": len(map_a),
            "completed_in_b": len(map_b),
            "overlapping_items": len(common_ids),
        }

    labels_a = [str(map_a[rid]["label"]).strip().upper() for rid in common_ids]
    labels_b = [str(map_b[rid]["label"]).strip().upper() for rid in common_ids]

    metrics = compute_cohens_kappa(
        rater_a=labels_a,
        rater_b=labels_b,
        item_ids=common_ids,
        categories=["VALID", "INVALID", "AMBIGUOUS"],
    )

    # Disagreement breakdown by invalid-reason category where applicable
    disagreements_by_error: Dict[str, int] = defaultdict(int)
    for rid in common_ids:
        ra, rb = map_a[rid], map_b[rid]
        la = str(ra.get("label")).strip().upper()
        lb = str(rb.get("label")).strip().upper()
        if la != lb:
            ea = str(ra.get("error_category") or "NONE").strip().upper()
            eb = str(rb.get("error_category") or "NONE").strip().upper()
            key = f"{la}({ea}) vs {lb}({eb})"
            disagreements_by_error[key] += 1

    label_dist_a = Counter(labels_a)
    label_dist_b = Counter(labels_b)

    return {
        "status": "AGREEMENT_COMPUTED",
        "overlapping_items": len(common_ids),
        "observed_agreement": metrics.observed_agreement,
        "expected_agreement": metrics.expected_agreement,
        "cohens_kappa": metrics.kappa,
        "standard_error": metrics.standard_error,
        "confidence_interval_95": metrics.confidence_interval_95,
        "interpretation": metrics.interpretation,
        "label_distribution_rater_a": dict(label_dist_a),
        "label_distribution_rater_b": dict(label_dist_b),
        "confusion_matrix": metrics.confusion_matrix,
        "disagreement_count": len(metrics.disagreement_indices),
        "disagreement_record_ids": metrics.disagreement_indices,
        "disagreements_by_error_category": dict(disagreements_by_error),
    }


def create_adjudication_queue(file_a: str, file_b: str, output_path: str) -> Dict[str, Any]:
    """
    Extract disagreements between Annotator A and Annotator B for human adjudication.
    
    Preserves both original annotations in their entirety without modifying source files.
    """
    p_a, p_b = Path(file_a), Path(file_b)
    if not p_a.exists():
        raise FileNotFoundError(f"File A not found: {file_a}")
    if not p_b.exists():
        raise FileNotFoundError(f"File B not found: {file_b}")

    with open(p_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(p_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    map_a = {r["record_id"]: r for r in data_a.get("records", [])}
    map_b = {r["record_id"]: r for r in data_b.get("records", [])}

    common_ids = sorted(list(set(map_a.keys()) & set(map_b.keys())))

    disagreements = []
    for rid in common_ids:
        ra, rb = map_a[rid], map_b[rid]
        la = ra.get("label")
        lb = rb.get("label")
        # Record disagreement if both annotated but labels differ
        if la and lb and str(la).strip().upper() != str(lb).strip().upper():
            disagreements.append({
                "record_id": rid,
                "document_id": ra["document_id"],
                "relation_id": ra["relation_id"],
                "concept_a": ra["concept_a"],
                "concept_b": ra["concept_b"],
                "relation_type": ra["relation_type"],
                "chunk_context": ra["chunk_context"],
                "source_chunk_ids": ra.get("source_chunk_ids", []),
                # Annotator A decision (preserved verbatim)
                "annotator_a": {
                    "annotator_id": ra.get("annotator_id", "Annotator_A"),
                    "label": ra.get("label"),
                    "error_category": ra.get("error_category"),
                    "confidence_rating": ra.get("confidence_rating"),
                    "evidence_text": ra.get("adjudicated_evidence_text") or ra.get("evidence_text"),
                    "notes": ra.get("notes"),
                },
                # Annotator B decision (preserved verbatim)
                "annotator_b": {
                    "annotator_id": rb.get("annotator_id", "Annotator_B"),
                    "label": rb.get("label"),
                    "error_category": rb.get("error_category"),
                    "confidence_rating": rb.get("confidence_rating"),
                    "evidence_text": rb.get("adjudicated_evidence_text") or rb.get("evidence_text"),
                    "notes": rb.get("notes"),
                },
                # Adjudication fields to be filled by human adjudicator
                "adjudicated_label": None,
                "adjudicated_error_category": None,
                "adjudicator_id": None,
                "adjudicated_evidence_text": None,
                "adjudication_rationale": None,
            })

    queue_manifest = {
        "manifest_version": "1.0.0",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_disagreements": len(disagreements),
        "source_annotator_a": str(p_a),
        "source_annotator_b": str(p_b),
        "instructions": (
            "Human Adjudication Protocol:\n"
            "1. For each disagreed candidate relation, review the candidate and chunk_context.\n"
            "2. Review the decisions of Annotator A and Annotator B.\n"
            "3. Make the final consensus determination: set 'adjudicated_label' to VALID, INVALID, or AMBIGUOUS.\n"
            "4. If INVALID, specify 'adjudicated_error_category'.\n"
            "5. Provide 'adjudicated_evidence_text' with the exact sentence(s) justifying the final verdict.\n"
            "6. Set 'adjudicator_id' and provide 'adjudication_rationale'."
        ),
        "valid_labels": [e.value for e in ValidityLabel],
        "valid_error_categories": [e.value for e in ErrorCategory],
        "disagreements": disagreements,
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(queue_manifest, f, indent=2, ensure_ascii=False)

    summary = {
        "output_path": str(out_p),
        "total_disagreements": len(disagreements),
        "source_a": str(p_a),
        "source_b": str(p_b),
    }
    logger.info(f"Created adjudication queue: {summary}")
    return summary


def assemble_gold_dataset_v1(
    primary_batch_path: str,
    output_gold_path: str,
    adjudication_path: Optional[str] = None,
    secondary_batch_path: Optional[str] = None,
    db_path: Optional[str] = None,
    dataset_version: str = "1.0.0",
    description: str = "Clarion Closed-Domain Relation Gold Dataset V1",
) -> GroundTruthDataset:
    """
    Assemble the final Gold Dataset V1 from human annotations.
    
    Distinguishes:
    - raw candidate metadata
    - annotator A label
    - annotator B label (where available)
    - adjudicated / final consensus label
    - evidence & provenance
    - annotation metadata
    
    Enforces the Gold Dataset Contract via GoldDatasetValidator and GroundTruthValidator.
    """
    from core.benchmarking.gold_dataset_contract import GoldDatasetValidator
    from core.benchmarking.dataset_validator import GroundTruthValidator

    p_prim = Path(primary_batch_path)
    if not p_prim.exists():
        raise FileNotFoundError(f"Primary batch file not found: {primary_batch_path}")

    with open(p_prim, "r", encoding="utf-8") as f:
        data_prim = json.load(f)

    # Validate primary batch completeness
    val_prim = validate_annotation_batch(primary_batch_path)
    if not val_prim["is_complete"]:
        raise ValueError(
            f"Cannot assemble Gold Dataset V1: Primary batch is incomplete "
            f"({val_prim['completed_records']}/{val_prim['total_records']} completed). "
            f"All records must be human-annotated."
        )

    # Load secondary batch (Annotator B) if provided
    map_sec = {}
    if secondary_batch_path and Path(secondary_batch_path).exists():
        with open(secondary_batch_path, "r", encoding="utf-8") as f:
            data_sec = json.load(f)
        for r in data_sec.get("records", []):
            if r.get("label"):
                map_sec[r["record_id"]] = r

    # Load adjudication if provided
    map_adj = {}
    if adjudication_path and Path(adjudication_path).exists():
        with open(adjudication_path, "r", encoding="utf-8") as f:
            data_adj = json.load(f)
        for item in data_adj.get("disagreements", []):
            if not item.get("adjudicated_label"):
                raise ValueError(
                    f"Adjudication item '{item.get('record_id')}' has null 'adjudicated_label'. "
                    f"All disagreements must be resolved by a human adjudicator."
                )
            map_adj[item["record_id"]] = item

    gt_records: List[GroundTruthRecord] = []
    db_updates: List[tuple] = []

    for r in data_prim["records"]:
        rid = r["record_id"]
        # Determine final label and provenance
        if rid in map_adj:
            # Resolved by adjudication
            adj = map_adj[rid]
            lbl = ValidityLabel(str(adj["adjudicated_label"]).strip().upper())
            err_cat = (
                ErrorCategory(str(adj["adjudicated_error_category"]).strip().upper())
                if adj.get("adjudicated_error_category")
                else ErrorCategory.NONE
            )
            ann_id = str(adj.get("adjudicator_id") or "human_adjudicator")
            ev_text = str(adj.get("adjudicated_evidence_text") or r.get("chunk_context") or "")
            conf = int(adj.get("confidence_rating") or 5)
            notes = str(adj.get("adjudication_rationale") or "")
            ann_a_lbl = str(adj["annotator_a"]["label"])
            ann_b_lbl = str(adj["annotator_b"]["label"])
            adj_status = "ADJUDICATED"
        elif rid in map_sec:
            # Dual annotated, agreed
            sec_rec = map_sec[rid]
            lbl = ValidityLabel(str(r["label"]).strip().upper())
            err_cat = (
                ErrorCategory(str(r["error_category"]).strip().upper())
                if r.get("error_category")
                else ErrorCategory.NONE
            )
            ann_id = f"{r.get('annotator_id', 'AnnA')}+{sec_rec.get('annotator_id', 'AnnB')}"
            ev_text = str(r.get("adjudicated_evidence_text") or r.get("chunk_context") or "")
            conf = int(r.get("confidence_rating") or 5)
            notes = r.get("notes")
            ann_a_lbl = str(r.get("label"))
            ann_b_lbl = str(sec_rec.get("label"))
            adj_status = "DUAL_AGREED"
        else:
            # Single annotated (from primary batch)
            lbl = ValidityLabel(str(r["label"]).strip().upper())
            err_cat = (
                ErrorCategory(str(r["error_category"]).strip().upper())
                if r.get("error_category")
                else ErrorCategory.NONE
            )
            ann_id = str(r.get("annotator_id") or "human_annotator")
            ev_text = str(r.get("adjudicated_evidence_text") or r.get("chunk_context") or "")
            conf = int(r.get("confidence_rating") or 5)
            notes = r.get("notes")
            ann_a_lbl = str(r.get("label"))
            ann_b_lbl = None
            adj_status = "SINGLE_ANNOTATED"

        raw_candidate_meta = {
            "llm_confidence": r.get("llm_confidence"),
            "cooccurrence_score": r.get("cooccurrence_score"),
            "semantic_similarity": r.get("semantic_similarity"),
            "chunk_context": r.get("chunk_context"),
            "source_chunk_ids": r.get("source_chunk_ids", []),
        }

        gt_rec = GroundTruthRecord(
            record_id=rid,
            document_id=r["document_id"],
            relation_id=r.get("relation_id") or rid,
            concept_a=r["concept_a"],
            concept_b=r["concept_b"],
            relation_type=r["relation_type"],
            label=lbl,
            error_category=err_cat,
            evidence_text=ev_text,
            source_chunk_ids=r.get("source_chunk_ids") or [],
            annotator_id=ann_id,
            confidence_rating=conf,
            notes=notes,
            annotator_a_label=ann_a_lbl,
            annotator_b_label=ann_b_lbl,
            adjudication_status=adj_status,
            raw_candidate=raw_candidate_meta,
        )
        gt_records.append(gt_rec)

        if db_path:
            is_valid_int = 1 if lbl == ValidityLabel.VALID else 0
            db_updates.append((is_valid_int, rid))

    dataset = GroundTruthDataset(
        dataset_version=dataset_version,
        domain="closed-domain-technical",
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description=description,
        records=gt_records,
    )

    # Validate against Gold Dataset Contract
    validator = GoldDatasetValidator(annotation_version=dataset_version)
    contract_res = validator.validate_records(gt_records)
    contract_res.raise_if_not_promotable()

    # Validate against GroundTruthValidator
    gt_val = GroundTruthValidator()
    raw_dicts = [json.loads(rec.model_dump_json()) for rec in gt_records]
    _, val_stats = gt_val.validate_records(raw_dicts)

    out_p = Path(output_gold_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(dataset.model_dump_json(indent=2))

    # Sync to SQLite if requested
    if db_path and Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.executemany(
            "UPDATE relation_dataset SET is_valid = ? WHERE record_id = ?",
            db_updates,
        )
        conn.commit()
        conn.close()
        logger.info(f"Updated {len(db_updates)} records in SQLite {db_path}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Clarion Relation Annotation Tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # export
    exp_p = subparsers.add_parser("export", help="Export candidate records to annotation template")
    exp_p.add_argument("--db-path", default="Clarion-Backend/data/relation_dataset.db")
    exp_p.add_argument("--output", default="Clarion-Backend/data/ground_truth/annotation_queue_batch1.json")
    exp_p.add_argument("--sample-size", type=int, default=None)
    exp_p.add_argument("--min-confidence", type=float, default=0.0)
    exp_p.add_argument("--seed", type=int, default=42)

    # export-overlap
    ovl_p = subparsers.add_parser("export-overlap", help="Export independent overlap batch for Annotator B")
    ovl_p.add_argument("--primary-batch", default="Clarion-Backend/data/ground_truth/annotation_queue_batch1.json")
    ovl_p.add_argument("--output", default="Clarion-Backend/data/ground_truth/annotation_queue_batch1_overlap_b.json")
    ovl_p.add_argument("--overlap-size", type=int, default=50)
    ovl_p.add_argument("--seed", type=int, default=42)
    ovl_p.add_argument("--annotator-b-id", default=None)

    # validate
    val_p = subparsers.add_parser("validate", help="Validate an annotated batch file")
    val_p.add_argument("--file", required=True)

    # agreement
    agr_p = subparsers.add_parser("agreement", help="Compute inter-annotator agreement between two files")
    agr_p.add_argument("--file-a", required=True)
    agr_p.add_argument("--file-b", required=True)

    # adjudicate
    adj_p = subparsers.add_parser("adjudicate", help="Extract disagreements between two annotators for adjudication")
    adj_p.add_argument("--file-a", required=True)
    adj_p.add_argument("--file-b", required=True)
    adj_p.add_argument("--output", default="Clarion-Backend/data/ground_truth/adjudication_queue_batch1.json")

    # assemble-gold
    gld_p = subparsers.add_parser("assemble-gold", help="Assemble final Gold Dataset V1 with provenance and validation")
    gld_p.add_argument("--primary", required=True, help="Full primary batch annotated by Annotator A")
    gld_p.add_argument("--adjudication", default=None, help="Completed adjudication file if disagreements existed")
    gld_p.add_argument("--secondary", default=None, help="Annotator B overlap batch if available")
    gld_p.add_argument("--output-gold", default="Clarion-Backend/data/ground_truth/human_gold_v1.json")
    gld_p.add_argument("--db-path", default="Clarion-Backend/data/relation_dataset.db")
    gld_p.add_argument("--version", default="1.0.0")

    # import
    imp_p = subparsers.add_parser("import", help="Import complete annotations into ground truth dataset")
    imp_p.add_argument("--input", required=True)
    imp_p.add_argument("--output-gold", default="Clarion-Backend/data/ground_truth/human_gold_v1.json")
    imp_p.add_argument("--db-path", default="Clarion-Backend/data/relation_dataset.db")

    # annotate (terminal interactive)
    ann_p = subparsers.add_parser("annotate", help="Launch interactive terminal annotation session")
    ann_p.add_argument("--file", required=True, help="Path to annotation queue JSON file")
    ann_p.add_argument("--annotator-id", default=None, help="Annotator ID (e.g. AM1)")

    # serve-ui (browser web interface)
    ui_p = subparsers.add_parser("serve-ui", help="Launch local browser-based annotation interface")
    ui_p.add_argument("--file", required=True, help="Path to annotation queue JSON file")
    ui_p.add_argument("--port", type=int, default=8080, help="Local port (default: 8080)")
    ui_p.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")

    args = parser.parse_args()

    if args.command == "export":
        summary = export_annotation_batch(
            db_path=args.db_path,
            output_path=args.output,
            sample_size=args.sample_size,
            min_confidence=args.min_confidence,
            random_seed=args.seed,
        )
        print(f"Successfully exported annotation batch: {summary}")
    elif args.command == "export-overlap":
        summary = export_overlap_batch(
            primary_batch_path=args.primary_batch,
            output_path=args.output,
            overlap_size=args.overlap_size,
            random_seed=args.seed,
            annotator_b_id=args.annotator_b_id,
        )
        print(f"Successfully exported independent overlap batch for Annotator B: {summary}")
    elif args.command == "validate":
        report = validate_annotation_batch(args.file)
        print(json.dumps(report, indent=2))
    elif args.command == "agreement":
        report = compute_agreement_between_files(args.file_a, args.file_b)
        if report.get("status") == "HUMAN ANNOTATION PENDING — AGREEMENT NOT COMPUTED":
            print("HUMAN ANNOTATION PENDING — AGREEMENT NOT COMPUTED")
            print(report.get("message"))
        else:
            print(json.dumps(report, indent=2))
    elif args.command == "adjudicate":
        summary = create_adjudication_queue(args.file_a, args.file_b, args.output)
        print(f"Successfully created adjudication queue: {summary}")
    elif args.command == "assemble-gold":
        ds = assemble_gold_dataset_v1(
            primary_batch_path=args.primary,
            output_gold_path=args.output_gold,
            adjudication_path=args.adjudication,
            secondary_batch_path=args.secondary,
            db_path=args.db_path,
            dataset_version=args.version,
        )
        print(f"Successfully assembled Gold Dataset V1 with {len(ds.records)} records: {args.output_gold}")
    elif args.command == "import":
        ds = import_annotation_batch_to_ground_truth(
            input_path=args.input,
            output_gold_path=args.output_gold,
            db_path=args.db_path,
        )
        print(f"Successfully imported {len(ds.records)} ground-truth records to {args.output_gold}")
    elif args.command == "annotate":
        from core.benchmarking.annotation_ui import InteractiveTerminalAnnotator
        session = InteractiveTerminalAnnotator(args.file, annotator_id=args.annotator_id)
        session.run()
    elif args.command == "serve-ui":
        from core.benchmarking.annotation_ui import run_web_server
        run_web_server(args.file, port=args.port, host=args.host)


if __name__ == "__main__":
    main()

"""
Unit tests for Clarion Interactive Human Annotation Interface (annotation_ui.py).

Verifies:
1. Backup creation before file modification.
2. Atomic saving mechanism.
3. Human annotation field validation (labels, error categories, annotator IDs, confidence, evidence).
4. Web server REST endpoints and safety guarantees (immutable fields cannot be mutated).
"""

import json
from pathlib import Path
import pytest

from core.benchmarking.annotation_ui import (
    create_backup,
    atomic_save_json,
    validate_single_annotation,
    LocalAnnotationHTTPHandler,
    ReusableThreadingServer,
    IMMUTABLE_FIELDS,
)


def test_create_backup(tmp_path):
    """Verify backup file is created with timestamped name."""
    test_file = tmp_path / "test_queue.json"
    test_file.write_text(json.dumps({"records": [{"id": 1}]}))

    backup = create_backup(test_file)
    assert backup.exists()
    assert ".bak_" in backup.name
    assert json.loads(backup.read_text()) == {"records": [{"id": 1}]}


def test_atomic_save_json(tmp_path):
    """Verify atomic write leaves valid JSON and cleans up temp files."""
    test_file = tmp_path / "atomic_target.json"
    data = {"status": "ok", "count": 42}
    atomic_save_json(test_file, data)

    assert test_file.exists()
    assert json.loads(test_file.read_text()) == data
    # No dangling .tmp file
    assert not (tmp_path / "atomic_target.json.tmp").exists()


def test_validate_single_annotation_valid():
    """Verify clean annotation inputs pass validation."""
    # VALID record
    ok, errs = validate_single_annotation(
        label="VALID",
        error_category="NONE",
        annotator_id="AM1",
        confidence_rating=5,
        evidence_text="Direct quote from document passage.",
    )
    assert ok is True
    assert len(errs) == 0

    # INVALID record with valid error category
    ok, errs = validate_single_annotation(
        label="INVALID",
        error_category="DIRECTION_INVERSION",
        annotator_id="AnnB",
        confidence_rating=4,
        evidence_text="Passage contradicts the direction.",
    )
    assert ok is True
    assert len(errs) == 0

    # AMBIGUOUS record
    ok, errs = validate_single_annotation(
        label="AMBIGUOUS",
        error_category="NONE",
        annotator_id="AM1",
        confidence_rating=3,
        evidence_text="Evidence mentions both but does not confirm relation.",
    )
    assert ok is True
    assert len(errs) == 0


def test_validate_single_annotation_failures():
    """Verify invalid inputs are caught and rejected."""
    # Invalid label
    ok, errs = validate_single_annotation("MAYBE", "NONE", "AM1", 5, "Quote")
    assert ok is False
    assert any("Label must be one of" in e for e in errs)

    # INVALID without error category
    ok, errs = validate_single_annotation("INVALID", "NONE", "AM1", 5, "Quote")
    assert ok is False
    assert any("error category" in e for e in errs)

    # Placeholder annotator ID
    for bad_ann in ("", "   ", "unknown", "none", "placeholder", "null"):
        ok, errs = validate_single_annotation("VALID", "NONE", bad_ann, 5, "Quote")
        assert ok is False

    # Out of range confidence rating
    for bad_conf in (0, 6, -1, 10):
        ok, errs = validate_single_annotation("VALID", "NONE", "AM1", bad_conf, "Quote")
        assert ok is False

    # Empty evidence text
    ok, errs = validate_single_annotation("VALID", "NONE", "AM1", 5, "")
    assert ok is False
    assert any("evidence text" in e for e in errs)


def test_immutable_fields_preserved_on_save(tmp_path):
    """Verify that saving human annotations NEVER alters candidate metadata."""
    sample_record = {
        "record_id": "rec-12345",
        "document_id": "doc-99999",
        "relation_id": "rel-88888",
        "concept_a": "Cache",
        "concept_b": "Memory",
        "relation_type": "explanation",
        "chunk_context": "The cache stores frequently accessed memory blocks.",
        "source_chunk_ids": ["c1", "c2"],
        "llm_confidence": 0.88,
        "cooccurrence_score": 0.65,
        "semantic_similarity": 0.78,
        "existing_db_is_valid": None,
        # Human fields:
        "label": None,
        "error_category": None,
        "annotator_id": None,
        "confidence_rating": None,
        "adjudicated_evidence_text": None,
        "notes": None,
    }

    test_file = tmp_path / "test_queue.json"
    data = {"manifest_version": "1.0.0", "records": [dict(sample_record)]}
    with open(test_file, "w") as f:
        json.dump(data, f)

    # Apply human annotation
    with open(test_file, "r") as f:
        loaded = json.load(f)

    r = loaded["records"][0]
    r["label"] = "VALID"
    r["error_category"] = "NONE"
    r["annotator_id"] = "HumanAnnotator1"
    r["confidence_rating"] = 5
    r["adjudicated_evidence_text"] = "The cache stores frequently accessed memory blocks."
    r["notes"] = "Clear explanatory definition."

    atomic_save_json(test_file, loaded)

    with open(test_file, "r") as f:
        saved = json.load(f)["records"][0]

    # Verify immutable fields remain strictly identical
    for field_name in IMMUTABLE_FIELDS:
        assert saved[field_name] == sample_record[field_name], f"Field {field_name} was mutated!"

    # Verify human fields were updated
    assert saved["label"] == "VALID"
    assert saved["annotator_id"] == "HumanAnnotator1"
    assert saved["confidence_rating"] == 5


def test_http_server_get_endpoints(tmp_path):
    """Verify live HTTP server handles GET / and GET /api/records with correct headers and Content-Length."""
    import threading
    import urllib.request

    test_file = tmp_path / "test_batch.json"
    dummy_records = [
        {
            "record_id": f"rec-{i}",
            "document_id": f"doc-{i}",
            "relation_id": f"rel-{i}",
            "concept_a": f"Head{i}",
            "concept_b": f"Tail{i}",
            "relation_type": "explanation",
            "chunk_context": f"Evidence context {i}",
            "source_chunk_ids": ["c1"],
            "llm_confidence": 0.9,
            "cooccurrence_score": 0.5,
            "semantic_similarity": 0.7,
            "existing_db_is_valid": None,
            "label": None,
            "error_category": None,
            "annotator_id": None,
            "confidence_rating": None,
            "adjudicated_evidence_text": None,
            "notes": None,
        }
        for i in range(3)
    ]
    test_file.write_text(json.dumps({"manifest_version": "1.0.0", "records": dummy_records}))

    handler_factory = lambda *args, **kwargs: LocalAnnotationHTTPHandler(*args, file_path=test_file, **kwargs)

    # Bind to port 0 to let OS assign an available ephemeral port
    with ReusableThreadingServer(("127.0.0.1", 0), handler_factory) as server:
        host, port = server.server_address
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            # Test GET /
            req_root = urllib.request.Request(f"http://{host}:{port}/")
            with urllib.request.urlopen(req_root) as res:
                assert res.status == 200
                assert "text/html" in res.headers.get("Content-Type")
                assert "Content-Length" in res.headers
                content = res.read().decode("utf-8")
                assert "Clarion Relation Annotation Console" in content

            # Test GET /api/records
            req_api = urllib.request.Request(f"http://{host}:{port}/api/records")
            with urllib.request.urlopen(req_api) as res:
                assert res.status == 200
                assert "application/json" in res.headers.get("Content-Type")
                assert "Content-Length" in res.headers
                data = json.loads(res.read().decode("utf-8"))
                assert data["total"] == 3
                assert len(data["records"]) == 3
                assert data["filename"] == test_file.name
        finally:
            server.shutdown()
            t.join(timeout=2.0)
            server.server_close()


def test_http_server_post_endpoint_and_safety(tmp_path):
    """Verify live HTTP server POST /api/record/<idx> updates human fields and leaves immutable fields intact."""
    import threading
    import urllib.request

    test_file = tmp_path / "test_batch.json"
    initial_record = {
        "record_id": "rec-immutable-1",
        "document_id": "doc-immutable-1",
        "relation_id": "rel-immutable-1",
        "concept_a": "Knowledge",
        "concept_b": "Graph",
        "relation_type": "composition",
        "chunk_context": "A knowledge graph organizes structured information.",
        "source_chunk_ids": ["c100"],
        "llm_confidence": 0.85,
        "cooccurrence_score": 0.62,
        "semantic_similarity": 0.77,
        "existing_db_is_valid": None,
        "label": None,
        "error_category": None,
        "annotator_id": None,
        "confidence_rating": None,
        "adjudicated_evidence_text": None,
        "notes": None,
    }
    test_file.write_text(json.dumps({"manifest_version": "1.0.0", "records": [dict(initial_record)]}))

    handler_factory = lambda *args, **kwargs: LocalAnnotationHTTPHandler(*args, file_path=test_file, **kwargs)

    with ReusableThreadingServer(("127.0.0.1", 0), handler_factory) as server:
        host, port = server.server_address
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            # POST /api/record/0
            post_payload = json.dumps({
                "label": "VALID",
                "error_category": "NONE",
                "annotator_id": "TestAnnotator",
                "confidence_rating": 5,
                "adjudicated_evidence_text": "A knowledge graph organizes structured information.",
                "notes": "Verified against chunk text.",
            }).encode("utf-8")

            req = urllib.request.Request(
                f"http://{host}:{port}/api/record/0",
                data=post_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as res:
                assert res.status == 200
                assert "Content-Length" in res.headers
                resp_data = json.loads(res.read().decode("utf-8"))
                assert resp_data["success"] is True
                assert resp_data["record"]["label"] == "VALID"
                assert resp_data["record"]["annotator_id"] == "TestAnnotator"

            # Verify disk persistence
            disk_data = json.loads(test_file.read_text())
            saved_rec = disk_data["records"][0]
            for field in IMMUTABLE_FIELDS:
                assert saved_rec[field] == initial_record[field], f"Immutable field {field} was altered!"
            assert saved_rec["label"] == "VALID"
            assert saved_rec["annotator_id"] == "TestAnnotator"
            assert saved_rec["confidence_rating"] == 5
        finally:
            server.shutdown()
            t.join(timeout=2.0)
            server.server_close()


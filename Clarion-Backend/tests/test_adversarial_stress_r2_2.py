"""
Adversarial Stress Verification Suite 2 (Round 2).
Focus:
1. Ingress robustness: POST /upload error containment
   - Invalid extensions (.exe, .txt)
   - 0-byte uploads
   - Corrupted zip / docx payload
   - Sub-20-word extractable text
   Confirm HTTP 400 with explicit error details.
2. Pipeline error containment & database hygiene:
   - Simulate pipeline stage failure (Stage 4: mapping, Stage 5: graph)
   - Verify document status transitions to DocumentStatus.FAILED
   - Verify error message is recorded
   - Verify analysis report is logged
   - Verify no orphaned/corrupted records remain in SQLite
3. Reprocessing deduplication:
   - Verify reprocessing purges prior chunks, embedding_segments, embeddings, and FAISS vectors
   - Verify final counts match initial counts with 0 duplicate records
   - Verify public API route POST /analyze/{document_id} with force_reanalyze=True
   - Verify parameter interface for ProcessingPipeline.execute(..., force_reanalyze=True vs skip_existing=False)
"""

import asyncio
import io
import json
import sqlite3
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import faiss

from main import app
from models.document import DocumentStatus
from services.document_service import DocumentService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.knowledge_map_service import KnowledgeMapService, LLMInterface
from services.background_service import BackgroundService
from services.processing_pipeline import ProcessingPipeline, PipelineStageError
from services.relation_dataset_service import RelationDatasetService
from services.retrieval_service import RetrievalService
from services.rag_audit_service import RagAuditService
from utils.config import settings
from tests.conftest import make_docx_bytes


def controlled_fake_llm_generate(self, prompt: str) -> str:
    """Deterministic controlled fake LLM generator for fast, isolated pipeline testing."""
    prompt_lower = prompt.lower()
    if (
        "extract key concepts" in prompt
        or "significant knowledge concepts" in prompt
        or "concepts in this text" in prompt
        or "most significant" in prompt
    ):
        return json.dumps([
            {"name": "Concept A", "definition": "Foundational test concept A."},
            {"name": "Concept B", "definition": "Test concept B derived from concept A."},
        ])
    elif "analyze the relationship between" in prompt or "relationship between these two" in prompt:
        return json.dumps({
            "relation_type": "prerequisite",
            "description": "Concept A is prerequisite for Concept B.",
            "confidence": 0.95
        })
    elif "main topics and their hierarchy" in prompt or "identify the main topics" in prompt:
        return json.dumps({
            "main_topics": [{"id": "topic_1", "title": "Main Topic", "description": "Top level topic", "subtopic_ids": ["sub_1"]}],
            "subtopics": [{"id": "sub_1", "title": "Subtopic 1", "parent_topic_id": "topic_1"}]
        })
    elif "executive summary" in prompt_lower or "structured summary" in prompt_lower:
        return "Executive summary of test document content with sufficient detail for pipeline verification."
    else:
        return "Deterministic controlled fake LLM response for test suite."


@pytest.fixture
def isolated_adversarial_env(tmp_path, monkeypatch):
    """Isolate storage and rebind router singletons for clean testing."""
    temp_data = tmp_path / "data"
    temp_vectorstore = temp_data / "vectorstore"
    temp_logs = tmp_path / "logs"
    temp_datasets = temp_data / "datasets"
    temp_graphs = temp_data / "graphs"
    temp_reports = temp_data / "reports"

    for d in (temp_data, temp_vectorstore, temp_logs, temp_datasets, temp_graphs, temp_reports):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "data_dir", temp_data)
    monkeypatch.setattr(settings, "vectorstore_dir", temp_vectorstore)
    monkeypatch.setattr(settings, "logs_dir", temp_logs)
    monkeypatch.setattr(settings, "dataset_dir", temp_datasets)
    isolated_db_path = temp_data / "clarion.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{isolated_db_path}")

    # Rebind router singletons
    import routers.analyze as analyze_mod
    import routers.query as query_mod
    import routers.upload as upload_mod

    monkeypatch.setattr(upload_mod, "document_service", DocumentService())
    monkeypatch.setattr(analyze_mod, "document_service", DocumentService())
    monkeypatch.setattr(analyze_mod, "background_service", BackgroundService(db_path=isolated_db_path))
    monkeypatch.setattr(analyze_mod, "pipeline", ProcessingPipeline())
    monkeypatch.setattr(query_mod, "document_service", DocumentService())
    monkeypatch.setattr(query_mod, "retrieval_service", RetrievalService())
    monkeypatch.setattr(query_mod, "knowledge_map_service", KnowledgeMapService())
    monkeypatch.setattr(query_mod, "rag_audit_service", RagAuditService(db_path=isolated_db_path))

    # Controlled LLM
    monkeypatch.setattr(LLMInterface, "_is_service_available", lambda self: True)
    monkeypatch.setattr(LLMInterface, "generate", controlled_fake_llm_generate)
    monkeypatch.setattr(RelationDatasetService, "_workspace_dataset_dir", lambda self: None)

    client = TestClient(app)
    return {
        "client": client,
        "db_path": isolated_db_path,
        "vectorstore_dir": temp_vectorstore,
        "tmp_path": tmp_path,
        "analyze_mod": analyze_mod,
    }


def make_valid_sample_docx(word_count: int = 150) -> bytes:
    """Build a valid DOCX with specified word count."""
    text = (
        "Statistical machine learning establishes algorithmic frameworks for discovering patterns "
        "and drawing probabilistic inferences from empirical data collections. "
        "Through iterative optimization algorithms and objective function regularization, "
        "predictive systems generalize effectively across previously unseen observations while "
        "minimizing empirical risk and controlling hypothesis class variance under strict convergence bounds. "
    )
    # repeat until reaching word_count
    words = text.split()
    repeated_words = []
    while len(repeated_words) < word_count:
        repeated_words.extend(words)
    content = " ".join(repeated_words[:word_count])
    return make_docx_bytes([content])


# ============================================================================
# 1. Ingress Robustness: POST /upload Error Containment
# ============================================================================

class TestUploadIngressRobustness:
    """Adversarially challenge POST /upload with invalid, empty, corrupt, and sub-20-word inputs."""

    def test_upload_invalid_extension_exe(self, isolated_adversarial_env):
        client = isolated_adversarial_env["client"]
        response = client.post(
            "/upload",
            files={"file": ("malicious_payload.exe", b"\x4d\x5a\x90\x00\x03\x00\x00\x00", "application/x-msdownload")}
        )
        assert response.status_code == 400, f"Expected 400 for .exe, got {response.status_code}"
        data = response.json()
        assert "File type not allowed" in data["detail"]
        assert ".pdf" in data["detail"] and ".docx" in data["detail"]

    def test_upload_invalid_extension_txt(self, isolated_adversarial_env):
        client = isolated_adversarial_env["client"]
        response = client.post(
            "/upload",
            files={"file": ("raw_notes.txt", b"This is a plain text file not allowed by upload.", "text/plain")}
        )
        assert response.status_code == 400, f"Expected 400 for .txt, got {response.status_code}"
        data = response.json()
        assert "File type not allowed" in data["detail"]

    def test_upload_zero_byte_docx(self, isolated_adversarial_env):
        client = isolated_adversarial_env["client"]
        response = client.post(
            "/upload",
            files={"file": ("empty.docx", b"", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        )
        assert response.status_code == 400, f"Expected 400 for 0-byte docx, got {response.status_code}"
        data = response.json()
        assert "empty (0 bytes)" in data["detail"]

    def test_upload_zero_byte_pdf(self, isolated_adversarial_env):
        client = isolated_adversarial_env["client"]
        response = client.post(
            "/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")}
        )
        assert response.status_code == 400, f"Expected 400 for 0-byte pdf, got {response.status_code}"
        data = response.json()
        assert "empty (0 bytes)" in data["detail"]

    def test_upload_corrupted_zip_payload_for_docx(self, isolated_adversarial_env):
        client = isolated_adversarial_env["client"]
        corrupted_zip_bytes = b"PK\x03\x04\x14\x00\x00\x00\x08\x00corrupted_non_zip_data_payload_stream_here"
        response = client.post(
            "/upload",
            files={"file": ("broken_archive.docx", corrupted_zip_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        )
        assert response.status_code == 400, f"Expected 400 for corrupt docx, got {response.status_code}"
        data = response.json()
        detail = data["detail"].lower()
        assert "corrupt" in detail or "invalid docx" in detail

    def test_upload_sub_20_word_extractable_text(self, isolated_adversarial_env):
        client = isolated_adversarial_env["client"]
        # Valid DOCX structure containing only 6 words (< 20 words)
        sub_20_bytes = make_docx_bytes(["Only six words in this document."])
        response = client.post(
            "/upload",
            files={"file": ("sub_20_words.docx", sub_20_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        )
        assert response.status_code == 400, f"Expected 400 for sub-20 words, got {response.status_code}"
        data = response.json()
        assert "No extractable text found" in data["detail"]
        assert "at least 20" in data["detail"] or "OCR first" in data["detail"]


# ============================================================================
# 2. Pipeline Error Containment and Database Hygiene
# ============================================================================

class TestPipelineErrorContainmentAndDatabaseHygiene:
    """Adversarially simulate stage failure and verify clean status, logging, and database hygiene."""

    def test_simulated_stage_4_mapping_failure(self, isolated_adversarial_env, monkeypatch):
        db_path = isolated_adversarial_env["db_path"]
        docx_bytes = make_valid_sample_docx(120)
        doc_service = DocumentService()
        doc = doc_service.upload_document(docx_bytes, "stage4_fail_test.docx")
        assert doc.status == DocumentStatus.UPLOADED

        # Inject failure into Stage 4 (mapping)
        async def failing_stage_mapping(document_id, chunks, use_existing=True):
            raise RuntimeError("CRITICAL_STAGE4_MAPPING_FAILURE: Synthetic LLM service fault")

        pipeline = ProcessingPipeline()
        monkeypatch.setattr(pipeline, "_stage_mapping", failing_stage_mapping)

        with pytest.raises(PipelineStageError) as exc_info:
            asyncio.run(pipeline.execute(doc.id, skip_existing=False))
        assert "CRITICAL_STAGE4_MAPPING_FAILURE" in str(exc_info.value)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # 1. Document status must be 'failed'
        doc_row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc.id,)).fetchone()
        assert doc_row is not None
        assert doc_row["status"] == DocumentStatus.FAILED.value
        assert "CRITICAL_STAGE4_MAPPING_FAILURE" in doc_row["error_message"]

        # 2. Analysis report must be saved with success=False
        report_row = conn.execute("SELECT * FROM analysis_reports WHERE document_id = ?", (doc.id,)).fetchone()
        assert report_row is not None
        report_data = json.loads(report_row["data"])
        assert report_data.get("success") is False
        assert "CRITICAL_STAGE4_MAPPING_FAILURE" in report_data.get("error", "")

        # 3. Database hygiene: Stage 4 failed before writing knowledge_maps, so NO corrupt map exists
        km_count = conn.execute("SELECT COUNT(*) FROM knowledge_maps WHERE document_id = ?", (doc.id,)).fetchone()[0]
        assert km_count == 0, f"Expected 0 knowledge_maps records for failed stage 4, found {km_count}"

        conn.close()

    def test_simulated_stage_5_graph_failure(self, isolated_adversarial_env, monkeypatch):
        db_path = isolated_adversarial_env["db_path"]
        docx_bytes = make_valid_sample_docx(120)
        doc_service = DocumentService()
        doc = doc_service.upload_document(docx_bytes, "stage5_fail_test.docx")

        # Inject failure into Stage 5 (graph building)
        async def failing_stage_graph(document_id, knowledge_map, use_existing=True):
            raise RuntimeError("CRITICAL_STAGE5_GRAPH_FAILURE: Graph builder synthetic exception")

        pipeline = ProcessingPipeline()
        monkeypatch.setattr(pipeline, "_stage_graph_building", failing_stage_graph)

        with pytest.raises(PipelineStageError) as exc_info:
            asyncio.run(pipeline.execute(doc.id, skip_existing=False))
        assert "CRITICAL_STAGE5_GRAPH_FAILURE" in str(exc_info.value)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        doc_row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc.id,)).fetchone()
        assert doc_row["status"] == DocumentStatus.FAILED.value
        assert "CRITICAL_STAGE5_GRAPH_FAILURE" in doc_row["error_message"]

        report_row = conn.execute("SELECT * FROM analysis_reports WHERE document_id = ?", (doc.id,)).fetchone()
        assert report_row is not None
        report_data = json.loads(report_row["data"])
        assert report_data.get("success") is False

        conn.close()


# ============================================================================
# 3. Reprocessing Deduplication
# ============================================================================

class TestReprocessingDeduplication:
    """Verify purge and deduplication on reprocessing."""

    def test_pipeline_reprocessing_with_skip_existing_false(self, isolated_adversarial_env):
        """Verify that execute with skip_existing=False purges prior records and avoids duplicates."""
        db_path = isolated_adversarial_env["db_path"]
        vectorstore_dir = isolated_adversarial_env["vectorstore_dir"]

        docx_bytes = make_valid_sample_docx(200)
        doc_service = DocumentService()
        doc = doc_service.upload_document(docx_bytes, "reprocess_verify.docx")

        pipeline = ProcessingPipeline()

        # Pass 1: Initial execution
        res1 = asyncio.run(pipeline.execute(doc.id, skip_existing=False))
        assert res1["success"] is True

        conn = sqlite3.connect(db_path)
        pass1_chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc.id,)).fetchone()[0]
        pass1_segs = conn.execute("SELECT COUNT(*) FROM embedding_segments WHERE document_id = ?", (doc.id,)).fetchone()[0]
        pass1_embs = conn.execute("SELECT COUNT(*) FROM embeddings WHERE document_id = ?", (doc.id,)).fetchone()[0]
        conn.close()

        vector_store = EmbeddingService().get_vector_store(doc.id)
        assert vector_store is not None
        index_file = vector_store.index_path
        assert index_file.exists()
        pass1_faiss_ntotal = faiss.read_index(str(index_file)).ntotal
        assert pass1_chunks > 0
        assert pass1_segs > 0
        assert pass1_embs > 0
        assert pass1_faiss_ntotal == pass1_segs

        # Pass 2: Reprocess with skip_existing=False (equivalent to force_reanalyze=True)
        res2 = asyncio.run(pipeline.execute(doc.id, skip_existing=False))
        assert res2["success"] is True

        conn = sqlite3.connect(db_path)
        pass2_chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc.id,)).fetchone()[0]
        pass2_segs = conn.execute("SELECT COUNT(*) FROM embedding_segments WHERE document_id = ?", (doc.id,)).fetchone()[0]
        pass2_embs = conn.execute("SELECT COUNT(*) FROM embeddings WHERE document_id = ?", (doc.id,)).fetchone()[0]
        conn.close()

        pass2_faiss_ntotal = faiss.read_index(str(index_file)).ntotal

        # STRICT DEDUPLICATION ASSERTIONS:
        assert pass2_chunks == pass1_chunks, f"Chunks duplicated: {pass2_chunks} != {pass1_chunks}"
        assert pass2_segs == pass1_segs, f"Segments duplicated: {pass2_segs} != {pass1_segs}"
        assert pass2_embs == pass1_embs, f"Embeddings duplicated: {pass2_embs} != {pass1_embs}"
        assert pass2_faiss_ntotal == pass1_faiss_ntotal, f"FAISS vectors duplicated: {pass2_faiss_ntotal} != {pass1_faiss_ntotal}"

        # Pass 3: A third reprocess run to stress test multi-cycle purge
        res3 = asyncio.run(pipeline.execute(doc.id, skip_existing=False))
        assert res3["success"] is True

        conn = sqlite3.connect(db_path)
        pass3_chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc.id,)).fetchone()[0]
        pass3_segs = conn.execute("SELECT COUNT(*) FROM embedding_segments WHERE document_id = ?", (doc.id,)).fetchone()[0]
        pass3_embs = conn.execute("SELECT COUNT(*) FROM embeddings WHERE document_id = ?", (doc.id,)).fetchone()[0]
        conn.close()

        pass3_faiss_ntotal = faiss.read_index(str(index_file)).ntotal
        assert pass3_chunks == pass1_chunks
        assert pass3_segs == pass1_segs
        assert pass3_embs == pass1_embs
        assert pass3_faiss_ntotal == pass1_faiss_ntotal

    def test_public_api_post_analyze_force_reanalyze(self, isolated_adversarial_env):
        """Verify that POST /analyze/{document_id} with force_reanalyze=True triggers reprocessing deduplication."""
        client = isolated_adversarial_env["client"]
        db_path = isolated_adversarial_env["db_path"]
        vectorstore_dir = isolated_adversarial_env["vectorstore_dir"]

        docx_bytes = make_valid_sample_docx(160)
        # Upload via public route
        upload_resp = client.post(
            "/upload",
            files={"file": ("api_reprocess_doc.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        )
        assert upload_resp.status_code == 200
        doc_id = upload_resp.json()["document_id"]

        # Run 1 via API
        analyze_resp1 = client.post(
            f"/analyze/{doc_id}",
            json={"force_reanalyze": False, "wait_for_completion": True}
        )
        assert analyze_resp1.status_code == 200
        assert analyze_resp1.json()["status"] in ("completed", "degraded")

        conn = sqlite3.connect(db_path)
        init_chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,)).fetchone()[0]
        init_segs = conn.execute("SELECT COUNT(*) FROM embedding_segments WHERE document_id = ?", (doc_id,)).fetchone()[0]
        init_embs = conn.execute("SELECT COUNT(*) FROM embeddings WHERE document_id = ?", (doc_id,)).fetchone()[0]
        conn.close()

        vector_store = EmbeddingService().get_vector_store(doc_id)
        assert vector_store is not None
        index_file = vector_store.index_path
        assert index_file.exists()
        init_faiss = faiss.read_index(str(index_file)).ntotal
        assert init_faiss == init_segs

        # Calling without force_reanalyze should return already_analyzed
        analyze_resp_cached = client.post(
            f"/analyze/{doc_id}",
            json={"force_reanalyze": False, "wait_for_completion": True}
        )
        assert analyze_resp_cached.status_code == 200
        assert analyze_resp_cached.json()["status"] == "already_analyzed"

        # Calling with force_reanalyze=True must reprocess and deduplicate
        analyze_resp2 = client.post(
            f"/analyze/{doc_id}",
            json={"force_reanalyze": True, "wait_for_completion": True}
        )
        assert analyze_resp2.status_code == 200
        assert analyze_resp2.json()["status"] in ("completed", "degraded")

        conn = sqlite3.connect(db_path)
        final_chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,)).fetchone()[0]
        final_segs = conn.execute("SELECT COUNT(*) FROM embedding_segments WHERE document_id = ?", (doc_id,)).fetchone()[0]
        final_embs = conn.execute("SELECT COUNT(*) FROM embeddings WHERE document_id = ?", (doc_id,)).fetchone()[0]
        conn.close()

        final_faiss = faiss.read_index(str(index_file)).ntotal

        assert final_chunks == init_chunks, f"Chunks duplicated via API: {final_chunks} != {init_chunks}"
        assert final_segs == init_segs, f"Segments duplicated via API: {final_segs} != {init_segs}"
        assert final_embs == init_embs, f"Embeddings duplicated via API: {final_embs} != {init_embs}"
        assert final_faiss == init_faiss, f"FAISS vectors duplicated via API: {final_faiss} != {init_faiss}"

    def test_execute_force_reanalyze_parameter_interface(self, isolated_adversarial_env):
        """
        Adversarially probe ProcessingPipeline.execute parameter signature.
        Demonstrates that execute() accepts skip_existing (bool) rather than force_reanalyze,
        and verifies behavior when called with unexpected kwargs.
        """
        pipeline = ProcessingPipeline()
        import inspect
        sig = inspect.signature(pipeline.execute)
        params = list(sig.parameters.keys())

        # The service layer contract uses 'skip_existing'
        assert "skip_existing" in params
        # Document that force_reanalyze is a router/API parameter, mapped to skip_existing=not force_reanalyze
        if "force_reanalyze" not in params:
            with pytest.raises(TypeError) as exc:
                asyncio.run(pipeline.execute("dummy_doc_id", force_reanalyze=True))
            assert "unexpected keyword argument 'force_reanalyze'" in str(exc.value)

"""Independently reconcile the isolated live demo artifacts without writing to its DB."""
import hashlib
import json
import sqlite3
from pathlib import Path

from PyPDF2 import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "demo_runs" / "clarion-final-demo-2026-11-12"
SOURCE = Path(__file__).resolve().parents[2] / "docs" / "Unit-4 Advanced OLAP Concepts.pptx.pdf"
doc_id = (RUN / "document-id.txt").read_text(encoding="utf-8").strip()
data = RUN / "data"
db = data / "clarion.db"
metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
km = json.loads((RUN / "knowledge-map.json").read_text(encoding="utf-8"))
query = json.loads((RUN / "query-response.json").read_text(encoding="utf-8"))
graph = json.loads((data / "graphs" / f"{doc_id}_cytoscape.json").read_text(encoding="utf-8"))

reader = PdfReader(str(SOURCE))
page_text = [page.extract_text() or "" for page in reader.pages]
independent = {
    "physical_pages": len(reader.pages),
    "extracted_pages": len(page_text),
    "text_bearing_pages": sum(bool(text.strip()) for text in page_text),
    "raw_whitespace_words": sum(len(text.split()) for text in page_text),
}

conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
doc = conn.execute("SELECT status, page_count, word_count, raw_word_count, source_file_path, source_file_sha256 FROM documents WHERE id=?", (doc_id,)).fetchone()
assert doc, f"document {doc_id} absent"
doc_status, doc_pages, doc_words, doc_raw_words, source_path, source_hash = doc
db_chunk_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id=?", (doc_id,)).fetchone()[0]
tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
segment_table = "embedding_segments_v2" if "embedding_segments_v2" in tables else "embedding_segments"
segments = conn.execute(f"SELECT COUNT(*) FROM {segment_table} WHERE document_id=?", (doc_id,)).fetchone()[0]
conn.close()

nodes = graph["elements"]["nodes"]
edges = graph["elements"]["edges"]
node_ids = {node["data"]["id"] for node in nodes}
orphan_edges = [edge["data"]["id"] for edge in edges if edge["data"].get("source") not in node_ids or edge["data"].get("target") not in node_ids]
concept_names = {c["name"].casefold() for c in km.get("concepts", [])}
relation_pairs = [(r["from_concept"].casefold(), r["to_concept"].casefold()) for r in km.get("relations", [])]
supported_relations = [r for r in km.get("relations", []) if r.get("source_evidence") and any(e.get("chunk_id") for e in r["source_evidence"])]
retrieved = query.get("results", [])
source_file = Path(source_path) if source_path else None
retained_hash = hashlib.sha256(source_file.read_bytes()).hexdigest() if source_file and source_file.is_file() else None

count_fields = metrics["counts"]
reconciliation = {
    "document_id_matches": metrics["document_id"] == doc_id == km["document_id"],
    "physical_pages_api_db_independent": [count_fields["physical_pages"], doc_pages, independent["physical_pages"]],
    "extracted_pages_api_independent": [count_fields["extracted_pages"], independent["extracted_pages"]],
    "text_bearing_pages_api_independent": [count_fields["text_bearing_pages"], independent["text_bearing_pages"]],
    "raw_words_api_db_independent": [count_fields["raw_words"], doc_raw_words, independent["raw_whitespace_words"]],
    "normalized_words_api_db": [count_fields["normalized_words"], doc_words],
    "retained_chunks_api_db": [count_fields["retained_chunks"], db_chunk_count],
    "segments_api_db": [count_fields["embedding_segments"], segments],
    "vectors_api": [count_fields["mapped_vectors"], count_fields["faiss_vectors"]],
    "source_hash_matches_record": bool(retained_hash and retained_hash == source_hash == metrics.get("source_file_sha256")),
    "retrieval_results_have_document_and_page_provenance": all(r.get("document_id") == doc_id and r.get("page_references") for r in retrieved),
}
result = {
    "run_id": "clarion-final-demo-2026-11-12",
    "document_id": doc_id,
    "source_document": str(SOURCE.relative_to(SOURCE.parents[1])).replace("\\", "/"),
    "independent_pdf_inspection": independent,
    "document_status": str(doc_status),
    "api_counts": count_fields,
    "database_retained_chunks": db_chunk_count,
    "database_embedding_segments": segments,
    "knowledge_map": {"concepts": len(km.get("concepts", [])), "candidate_relations": len(km.get("relations", [])), "evidence_backed_relations": len(supported_relations), "duplicate_concept_names": len(km.get("concepts", [])) - len(concept_names), "duplicate_directed_relation_pairs": len(relation_pairs) - len(set(relation_pairs))},
    "graph": {"nodes": len(nodes), "edges": len(edges), "orphan_edges": orphan_edges},
    "retrieval": {"results": len(retrieved), "generated_answer": bool(query.get("response")), "fallback_used": query.get("fallback_used"), "results_with_page_citations": sum(bool(r.get("page_references")) for r in retrieved)},
    "reconciliation": reconciliation,
    "limitations": ["Browser automation did not reach page rendering: Chromium and Edge exited during process startup in this Windows session; no screenshot or frontend interaction is claimed as verified.", "Only one PDF was run through live processing in this session; slide PDF behavior is shown, while long-text, table-heavy, and image-only/low-text documents remain for the next validation pass.", "Rejected chunk count is not exposed by the current API; it is null, not inferred.", "The dataset-export mirror was copied into this isolated run and the uniquely named workspace mirror was removed; set CLARION_DISABLE_WORKSPACE_DATASET_MIRROR=1 for isolated runs."],
}
result["all_reconciliations_pass"] = all([
    result["reconciliation"]["document_id_matches"],
    result["reconciliation"]["physical_pages_api_db_independent"][0] == result["reconciliation"]["physical_pages_api_db_independent"][1] == result["reconciliation"]["physical_pages_api_db_independent"][2],
    result["reconciliation"]["extracted_pages_api_independent"][0] == result["reconciliation"]["extracted_pages_api_independent"][1],
    result["reconciliation"]["text_bearing_pages_api_independent"][0] == result["reconciliation"]["text_bearing_pages_api_independent"][1],
    result["reconciliation"]["raw_words_api_db_independent"][0] == result["reconciliation"]["raw_words_api_db_independent"][1] == result["reconciliation"]["raw_words_api_db_independent"][2],
    result["reconciliation"]["normalized_words_api_db"][0] == result["reconciliation"]["normalized_words_api_db"][1],
    result["reconciliation"]["retained_chunks_api_db"][0] == result["reconciliation"]["retained_chunks_api_db"][1],
    result["reconciliation"]["segments_api_db"][0] == result["reconciliation"]["segments_api_db"][1],
    result["reconciliation"]["vectors_api"][0] == result["reconciliation"]["vectors_api"][1],
    result["reconciliation"]["source_hash_matches_record"],
    result["reconciliation"]["retrieval_results_have_document_and_page_provenance"],
    not orphan_edges,
])
(RUN / "live-demo-verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))

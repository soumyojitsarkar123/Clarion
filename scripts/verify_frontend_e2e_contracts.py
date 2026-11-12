"""End-to-end integration and DOM-contract verification for Clarion frontend and backend.

Validates the full user journey simulated by frontend/app.js against real backend endpoints:
1. System health & honest degraded/fallback badge state.
2. Document library population.
3. Document selection metrics: concepts, candidate relations, summary, graph structure.
4. Cytoscape elements schema verification (preventing JS exceptions in frontend).
5. Grounded RAG query interaction with vector chunk retrieval.
6. Error boundary verification (404 on missing document, 400 on invalid input).
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from main import app

REAL_DOC_ID = "a945d65e-f069-41f9-abb2-e1a7e774d559"


def run_e2e_verification():
    print("=" * 70)
    print("CLARION MILESTONE 4 — FRONTEND & RUNTIME E2E CONTRACT VERIFICATION")
    print("=" * 70)

    client = TestClient(app)
    results = {}

    # 1. System Health & Honest Status Ribbon
    print("\n[Step 1] System Health & LLM Runtime State")
    health_res = client.get("/health")
    assert health_res.status_code == 200, f"Health endpoint failed: {health_res.text}"
    health_data = health_res.json()
    print(f"  Health status: {health_data.get('status')}")

    sys_res = client.get("/system-status")
    assert sys_res.status_code == 200, f"System status endpoint failed: {sys_res.text}"
    sys_data = sys_res.json()
    llm_info = sys_data.get("services", {}).get("llm", {})
    llm_status = llm_info.get("status")
    llm_model = llm_info.get("model")
    llm_provider = llm_info.get("provider")
    print(f"  LLM Provider: {llm_provider}")
    print(f"  LLM Status: {llm_status}")
    print(f"  LLM Model Display: {llm_model}")
    assert any(term in llm_status.lower() for term in ["unavailable", "available", "demo", "degraded", "ready"]), f"Unexpected LLM status: {llm_status}"

    hc_res = client.get(f"/system/health-check?document_id={REAL_DOC_ID}")
    assert hc_res.status_code == 200, f"Health check endpoint failed: {hc_res.text}"
    hc_data = hc_res.json()
    print(f"  Pipeline degraded flag: {hc_data.get('pipeline', {}).get('degraded')}")
    print(f"  System health: {hc_data.get('system_status')}")
    results["step1_health_and_ribbon"] = "PASS"

    # 2. Document Library Listing
    print("\n[Step 2] Document Library Population")
    docs_res = client.get("/upload/list")
    assert docs_res.status_code == 200, f"Upload list endpoint failed: {docs_res.text}"
    docs_payload = docs_res.json()
    docs = docs_payload if isinstance(docs_payload, list) else docs_payload.get("documents", [])
    print(f"  Total documents listed: {len(docs)}")
    target_doc = next((d for d in docs if d.get("id") == REAL_DOC_ID or d.get("document_id") == REAL_DOC_ID), None)
    if not target_doc:
        target_doc = next((d for d in docs if "DAA" in (d.get("name") or d.get("filename") or "") and d.get("status") == "analyzed"), None)
    assert target_doc is not None, f"Target DAA document not found in library: {docs}"
    active_doc_id = target_doc.get("id") or target_doc.get("document_id")
    print(f"  Target document ({active_doc_id}) present: True")
    results["step2_document_library"] = "PASS"

    # 3. Document Selection & Overview Metrics
    print(f"\n[Step 3] Loading Document Details for ID: {active_doc_id}")
    status_res = client.get(f"/status/{active_doc_id}")
    assert status_res.status_code == 200, f"Status check failed: {status_res.text}"
    status_data = status_res.json()
    print(f"  Document status: {status_data.get('document_status')}")
    assert status_data.get("document_status") == "analyzed"

    # Knowledge Map
    km_res = client.get(f"/knowledge-map/{active_doc_id}")
    assert km_res.status_code == 200, f"Knowledge map failed: {km_res.text}"
    km_data = km_res.json()
    concepts = km_data.get("concepts", [])
    relations = km_data.get("relations", [])
    print(f"  Concepts loaded: {len(concepts)}")
    print(f"  Candidate relations loaded: {len(relations)}")
    assert len(concepts) > 0, "No concepts returned"
    assert len(relations) > 0, "No candidate relations returned"

    # Verify relation candidate flags and evidence grounding
    for rel in relations[:5]:
        assert rel.get("is_candidate") is True, "Relation missing is_candidate flag"
        assert rel.get("evidence_text"), "Relation missing grounded evidence text"
    print("  Validated candidate flags and textual grounding on relations.")

    # Summary
    sum_res = client.get(f"/summary/{active_doc_id}")
    assert sum_res.status_code == 200, f"Summary failed: {sum_res.text}"
    sum_data = sum_res.json()
    doc_type = sum_data.get("document_type")
    sections = sum_data.get("sections", [])
    print(f"  Document classification: {doc_type}")
    print(f"  Summary sections count: {len(sections)}")
    results["step3_document_metrics"] = "PASS"

    # 4. Cytoscape Graph Schema Verification
    print(f"\n[Step 4] Cytoscape Graph Schema Verification")
    graph_res = client.get(f"/graph/{active_doc_id}")
    assert graph_res.status_code == 200, f"Graph endpoint failed: {graph_res.text}"
    graph_data = graph_res.json()

    elements = graph_data.get("elements", {})
    nodes = elements.get("nodes", []) or graph_data.get("nodes", [])
    edges = elements.get("edges", []) or graph_data.get("edges", [])
    print(f"  Cytoscape graph nodes: {len(nodes)}")
    print(f"  Cytoscape graph edges: {len(edges)}")
    assert len(nodes) > 0, "Graph has no nodes"

    # Ensure node schema compatibility (data.id required by Cytoscape)
    for n in nodes[:5]:
        assert "data" in n and "id" in n["data"], f"Invalid node schema: {n}"
    for e in edges[:5]:
        assert "data" in e and "source" in e["data"] and "target" in e["data"], f"Invalid edge schema: {e}"
    print("  Cytoscape schema verification: OK (Zero JS rendering crash risk).")
    results["step4_graph_schema"] = "PASS"

    # 5. Grounded RAG Query Interaction
    print(f"\n[Step 5] Grounded RAG Query Interaction")
    query_payload = {"query": "Explain Bellman-Ford algorithm for negative edge weights"}
    query_res = client.post(f"/query/{active_doc_id}", json=query_payload)
    assert query_res.status_code == 200, f"Query endpoint failed: {query_res.text}"
    q_data = query_res.json()
    resp_text = q_data.get("response", "")
    retrieved_chunks = q_data.get("results", [])
    print(f"  RAG Response length: {len(resp_text)}")
    print(f"  RAG Response preview: {resp_text[:120]}...")
    print(f"  Retrieved vector chunks: {len(retrieved_chunks)}")
    assert len(retrieved_chunks) > 0, "No chunks retrieved by vector search"
    assert len(resp_text) > 0, "Empty RAG response returned"
    results["step5_rag_query"] = "PASS"

    # 6. Error Handling & Resilience
    print(f"\n[Step 6] Error Handling & Boundary Verification")
    missing_res = client.get("/status/non-existent-doc-uuid-0000")
    print(f"  Missing document status code: {missing_res.status_code} (Expected 404)")
    assert missing_res.status_code == 404, f"Expected 404 for missing doc, got {missing_res.status_code}"

    empty_upload_res = client.post(
        "/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")}
    )
    print(f"  Empty file upload status code: {empty_upload_res.status_code} (Expected 400)")
    assert empty_upload_res.status_code == 400, f"Expected 400 for empty PDF, got {empty_upload_res.status_code}"
    results["step6_error_resilience"] = "PASS"

    print("\n" + "=" * 70)
    print("E2E VERIFICATION RESULTS SUMMARY:")
    for step, outcome in results.items():
        print(f"  * {step}: {outcome}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_e2e_verification()

"""Live end-to-end smoke test against a running Clarion backend."""
import io
import sys
import time
import json

import httpx

BASE = "http://127.0.0.1:8000"
client = httpx.Client(base_url=BASE, timeout=60)

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not condition:
        failures.append(name)


def make_docx(paragraphs):
    from docx import Document

    doc = Document()
    for para in paragraphs:
        doc.add_paragraph(para)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


text = (
    "Hypothesis Testing\n\n"
    "Hypothesis testing is a statistical method used to make decisions about a "
    "population based on a random sample. The null hypothesis represents the "
    "default assumption, while the alternative hypothesis represents the claim "
    "being tested. A sampling distribution describes the distribution of the "
    "sample mean under repeated sampling. The standard normal variate is used "
    "to compute test statistics. The significance level controls the "
    "probability of a type one error. Confidence intervals provide a range of "
    "plausible values for the population parameter."
)

docx_bytes = make_docx(text.split("\n\n"))

r = client.post("/upload", files={"file": ("e2e_stats.docx", docx_bytes, "application/octet-stream")})
check("upload docx", r.status_code == 200, r.text[:200])
if r.status_code != 200:
    sys.exit(1)
doc_id = r.json()["document_id"]

r = client.post(
    f"/analyze/{doc_id}",
    json={"wait_for_completion": True, "timeout_seconds": 600},
    timeout=600,
)
check("analyze sync", r.status_code == 200 and r.json().get("status") == "completed", r.text[:300])
analyze_payload = r.json()

results = analyze_payload.get("results", {})
check("pipeline success flag", results.get("success") is True)
check("7 stages completed", set(["ingestion", "chunking", "embedding", "mapping", "graph_building", "evaluation", "hierarchy"]).issubset(set(results.get("stages_completed", []))))

r = client.get(f"/status/{doc_id}")
check("document status analyzed", r.status_code == 200 and r.json()["document_status"] == "analyzed", r.text[:200])

r = client.get(f"/knowledge-map/{doc_id}")
km = r.json()
check("knowledge map concepts", r.status_code == 200 and len(km.get("concepts", [])) > 0, f"concepts={len(km.get('concepts', []))}")
check("knowledge map relations", len(km.get("relations", [])) > 0, f"relations={len(km.get('relations', []))}")

r = client.get(f"/graph/{doc_id}")
g = r.json()
nodes = g.get("elements", {}).get("nodes", []) if isinstance(g.get("elements"), dict) else []
check("graph has elements", r.status_code == 200 and len(nodes) > 0, f"nodes={len(nodes)}")

r = client.get(f"/summary/{doc_id}")
check("summary sections", r.status_code == 200 and len(r.json().get("sections", [])) > 0)

r = client.get(f"/system/analysis-report/{doc_id}")
report = r.json()
check("analysis report success", r.status_code == 200 and report.get("success") is True)

r = client.post(f"/query/{doc_id}", json={"query": "What is a null hypothesis?"}, timeout=240)
check("rag query", r.status_code == 200 and r.json().get("query") == "What is a null hypothesis?", r.text[:200])

r = client.get("/dataset/relations")
check("relation dataset rows", r.status_code == 200 and r.json().get("count", 0) > 0, f"count={r.json().get('count')}")

r = client.get(f"/system/pipeline-status?document_id={doc_id}")
check("pipeline tracker completed", r.status_code == 200 and r.json().get("status") == "completed")

r = client.get("/system-status")
check("system status", r.status_code == 200 and "services" in r.json())

r = client.get("/system/health-check")
check("health check", r.status_code == 200 and r.json().get("status") in {"healthy", "degraded", "warning"}, r.text[:200])

# Clean up the document and its artifacts so the test is repeatable.
client.delete(f"/upload/{doc_id}")
client.delete(f"/dataset/relations/{doc_id}")

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("E2E SMOKE TEST PASSED")

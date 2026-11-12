# Full application readiness report

**Classification: PARTIALLY VERIFIED — NOT RELEASE READY.** An isolated real PDF extraction/chunk/BGE/FAISS/search path passed. Full backend regression stalls in Windows async event-loop initialization; live provider generation and browser citation display remain unverified.

| Workflow step | Status | Evidence / limit |
|---|---|---|
| PDF page extraction and persistence | **PASS** in isolated services | 87 page records from a real PDF; extraction warnings identify unverified layout/order. |
| Normalization and source offsets | **PASS** for extracted-text mapping | Versioned v2 map; OCR and visual glyph coordinates unavailable. |
| Chunk/segment provenance | **PASS** in isolated run | 44 chunks and 100 BGE segments carry source spans and model identity. |
| FAISS mapping/search | **PASS** in isolated run | 100 vectors match model-scoped DB map; search returned page-bounded spans. |
| Retrieval/API citation fields | **PARTIALLY VERIFIED** | Retrieval service returns source fields in isolated tests; API endpoint did not complete through TestClient. |
| Frontend citation display | **NOT VERIFIED** | Static renderer updated; no browser executable or Playwright. |
| Query response generation | **NOT VERIFIED** | Ollama loopback unavailable; exact provider usage not observed. |
| Dataset output budgeting | **PARTIALLY VERIFIED** | Oversize rejection/audit tests pass; async generated dataset path stalls before the mock provider. |
| Backend full regression | **BLOCKED** | 14 tests pass, test 15 stalls in AnyIO/asyncio socketpair initialization. |
| Migration rollback/recovery | **PARTIALLY VERIFIED** | Isolated migration rollback passes; production migration and backup restore not run. |

Use the final integration reports for source provenance, token budgets, runtime recovery, and live application/browser evidence.

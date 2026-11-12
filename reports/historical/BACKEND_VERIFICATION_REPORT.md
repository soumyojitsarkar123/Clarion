# Backend Verification Report

## Status: FAILED

Interpreter: `D:\Clarion\Clarion-Backend\.venv-validation\Scripts\python.exe` (Python 3.12.14).

Commands executed:

```powershell
.\.venv-validation\Scripts\python.exe -m pytest -q
.\.venv-validation\Scripts\python.exe -m pytest tests\test_embedding_segments.py tests\test_embedding_service.py tests\test_challenger1_m1_adversarial.py -q
```

The focused embedding/segment suite passed: **33 passed**. The complete suite reached 100% but reported three failures and one skipped test. The failures are outside the segmentation change and reflect live local LLM/provider behavior where tests expected deterministic fallback/heuristic metadata:

1. `tests/test_document_intelligence_overhaul.py::test_summary_deterministic_fallback_notice` expected `fallback_used=True`, received `False`.
2. `tests/test_knowledge_map_service.py::test_build_knowledge_map_is_heuristic` expected `extraction_method="heuristic"`, received `"llm"`.
3. `tests/test_m3_fallback_quality.py::test_fallback_overall_summary_has_no_template_boilerplate` expected `fallback_used=True`, received `False`.

Pytest also emitted existing Windows socket-thread warnings in annotation UI tests and cache-write warnings for `.pytest_cache` access. No segmentation-focused test failed.

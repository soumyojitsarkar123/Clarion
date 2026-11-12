# Token budget architecture

**Status:** PARTIALLY VERIFIED. A central service now exists for the active document-query path; complete workflow integration is not finished.

## Runtime terms

For the active query request, the service counts the composed single user prompt (fixed instructions, query, ranked evidence and evidence IDs/section labels). There is no separate chat history or schema in this route. Runtime context comes from `LLM_CONTEXT_WINDOW` (default 4,096), not an advertised model maximum. Reserved output is provider-aware: Ollama preserves the existing 800-token cap; other providers reserve up to configured `LLM_MAX_TOKENS` (default 2,000). `LLM_TOKEN_SAFETY_TOKENS` defaults to 256 for chat template and uncounted formatting overhead.

\[
B_{available}=C_{configured}-T_{prompt}-T_{reserved\ output}-T_{safety}
\]

`T_prompt` is exact only when an explicitly configured, matching local tokenizer is available via `LLM_TOKENIZER_NAME`. The tokenizer is loaded cache-only. Otherwise the implementation uses UTF-8 byte length as a conservative preflight estimate and sets `preflight_exact=false`, `preflight_method=utf8_byte_upper_bound_estimate`. Provider prompt/output counts, when supplied, are recorded separately and never substituted with this estimate.

## Packing and refusal behavior

`TokenBudgetService.pack_ranked_evidence` evaluates complete evidence items in retrieval rank order, keeps only items that fit, removes normalized duplicate content, and returns IDs plus per-item omission reasons (`duplicate_content` or `context_budget`). It never cuts a passage mid-text. `QueryResponse` returns omission information and preflight/provider usage fields. If even the fixed prompt cannot fit, the route rejects with HTTP 413. The LLM adapter rejects over-budget prompts before provider access. Ollama `done_reason=length`, cloud `finish_reason=length`, and Gemini `MAX_TOKENS` are recorded as truncation; a partial completion is not accepted as a normal answer.

## Current limitations

- The runtime Qwen3.5 tokenizer is unavailable in this sandbox; default counts are estimates.
- Exact system-template token overhead is not measured; safety reserve is configurable but must be validated per provider/model.
- Query route has no separate system/history/schema fields today; these terms must be added before those features are introduced.
- Dataset factory uses its own character-derived estimates and is not yet wired to this service.
- Pipeline-stage token telemetry is not persisted end-to-end; query response is the current reporting surface.
- Provider live usage/truncation was not exercised because the Ollama daemon is unavailable in this environment.

## Configuration

`LLM_CONTEXT_WINDOW=4096`, `LLM_RESERVED_OUTPUT_TOKENS=2000`, `LLM_TOKEN_SAFETY_TOKENS=256`, optional `LLM_TOKENIZER_NAME`. Defaults preserve the previous 4K Ollama context and 800-token Ollama generation cap. Model/provider-specific limits must be set to the actual served configuration.

## Code and verification

Implementation: `Clarion-Backend/services/token_budget_service.py`, `services/knowledge_map_service.py`, `routers/query.py`, `utils/config.py`, `models/response.py`. Targeted tests cover exact boundary arithmetic, UTF-8 estimate labeling, duplicate/whole-item omission, and refusal before provider access. The targeted query API contract passed with local pytest temp storage. Full regression status is recorded in the final verification report after that run.

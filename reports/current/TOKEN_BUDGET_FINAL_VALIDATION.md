# Token budget validation summary

**Current classification: PARTIALLY VERIFIED.** This summary supersedes the earlier baseline assessment. The current provider and end-to-end call-path results are in [`TOKEN_BUDGET_PROVIDER_INTEGRATION_REPORT.md`](TOKEN_BUDGET_PROVIDER_INTEGRATION_REPORT.md) and [`TOKEN_BUDGET_END_TO_END_REPORT.md`](TOKEN_BUDGET_END_TO_END_REPORT.md).

Focused tests cover exact fake-tokenizer/chat-template measurement, UTF-8 fallback estimates, Unicode byte lengths, special tokens/schema, context and output reservations, safe whole-passage packing, dataset preflight rejection, provider usage capture and incomplete-output reporting. The four first-party providers, query/RAG, dataset, structured extraction, summary and knowledge-map call paths route through the shared budget controls.

Live provider usage and deployed Qwen tokenizer counts remain unverified. Ollama's runtime endpoint is blocked by this host, and full async application tests stall during Windows socket-pair initialization.

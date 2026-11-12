# Token budget end-to-end report

**Classification: PARTIALLY VERIFIED.** Unit/provider boundary and safe passage-packing checks pass. A live query-generation or dataset-generation request did not complete because Windows async event-loop socket-pair initialization stalls before the mocked/live provider call.

## Call-path review

- Query/RAG: `routers/query.py` uses `LLMInterface` and `TokenBudgetService.pack_ranked_evidence`; whole evidence passages are packed against the prompt, output reservation, and safety margin. Omitted evidence and counting method are included in query response metrics.
- Dataset generation: service-level preflight includes structured schema and uses a task-appropriate output cap; the provider performs a second check over its final structured prompt before network activity. Oversized requests are rejected without retry, and token usage/audit fields are additive.
- Knowledge-map and summary calls use `LLMInterface`, whose API boundary performs budget preflight. Structured extraction paths use provider `generate_structured`, then pass through provider preflight.
- No LLM streaming endpoint was found. Existing WebSocket progress/log updates do not send LLM prompts.

## Test evidence and limits

The final focused run passed 21 cases across token budgeting, provider boundaries, embedding segmentation, dataset rejection/migration, and provenance tests. Whole-passage packing reports `duplicate_content` and `context_budget` omissions. The dataset workflow test that proceeds past preflight reaches `asyncio.run()`/AnyIO loop creation and stalls in this Windows host; it was not treated as a provider pass. No live Qwen, OpenAI, Kimi, or GLM usage metrics were collected. The full regression suite also stalls before query/RAG API tests execute.

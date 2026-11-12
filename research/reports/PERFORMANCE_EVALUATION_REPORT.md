# Performance evaluation report

**Status: NOT EVALUATED as a controlled benchmark.** The earlier isolated indexing run and two live Ollama requests were functional smoke measurements. They do not provide repeated cold/warm samples, p50/p95, throughput, peak memory, or controlled component timings. Do not extrapolate those durations.

No controlled latency results are reported here. A future run must freeze host CPU/GPU, RAM, OS, Python and library versions, model revision/provider, input sizes, concurrency, and run counts; separately measure extraction, chunking, embedding, index write, retrieval, reranking, LLM first-token and generation; include cold starts and warm requests; retain raw samples; report p50/p95 and peak RSS/VRAM. If access to consistent hardware/provider conditions is unavailable, mark the affected stage BLOCKED.

# Token budget provider integration report

**Classification: PARTIALLY VERIFIED.** Shared preflight and accounting are wired through the four first-party provider classes and the first-party query/dataset interfaces. Provider-boundary tests pass, but no live LLM provider request or exact Qwen tokenizer count was obtained in this environment.

## Enforcement and accounting

- Ollama, OpenAI, Kimi, and GLM generation entrypoints call the common preflight before issuing a request. Direct construction of these provider classes uses the same check.
- Ollama preflight uses the effective `num_ctx` sent in the request. Other providers use the explicit configured context override or model registry limit. Dataset generation uses a bounded per-chunk output reservation of at most 1024 tokens and passes it through to the provider.
- If a matching model tokenizer and chat template are configured, preflight counts the serialized messages/template. Without that, the request uses an explicit UTF-8 byte upper-bound estimate plus safety tokens. Schema material, role framing, and request JSON framing are included conservatively; an estimate is not labeled exact.
- Metrics distinguish model-token count when exact, estimated count, UTF-8 byte length, special-token overhead, context limit, output/safety reservations, and provider-reported input/output/total usage. Output-limit finish reasons are surfaced as incomplete responses rather than silently accepted.
- The existing RAG evidence packer retains complete ranked passages, removes normalized exact duplicates, and reports every omitted passage and reason.

## Verification

The focused provider/token suite passed in the final run: exact fake-tokenizer/template accounting, special tokens and schema, UTF-8 fallback and Unicode byte length, context/output boundary rejection, evidence deduplication/omission, and response usage/output-limit handling. Dataset oversized-input rejection and additive audit migration passed in isolation. The entire affected focused set is recorded in `research_artifacts/raw/backend-full-suite-final-2026-11-12.log` for the full run; the 21-case focused command passed.

Actual prompt usage from a live Qwen/Ollama or remote-provider response remains unavailable. Ollama's locally installed model is `qwen3.5:4b` with manifest model-layer digest `sha256:81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490`, but loopback port 11434 is denied by this execution environment. Exact remote deployment context limits and provider framing equivalence therefore remain deployment-specific.

# Model compatibility and selection report

**Decision: retain the configured controls. Challenger quality is NOT EVALUATED.** No model default was changed.

| Role | Control / candidate | Evidence available in this checkout | Decision |
|---|---|---|---|
| Generation | `qwen3.5:4b` | Configured baseline. Earlier same-prompt latency smoke is historical and has no quality labels or pinned digest in this validation run. | Retain; quality and current digest NOT VERIFIED. |
| Generation | `qwen3.5:2b` | No controlled same-input comparison, digest, output-quality labels, or current resource sample. | NOT EVALUATED. |
| Generation | `qwen3.5:9b` | Earlier descriptive latency smoke only; no controlled quality labels or current memory/KV-cache sample. | NOT EVALUATED as a challenger. |
| Embedding | `BAAI/bge-large-en-v1.5` | Configured baseline. Existing historical isolated run recorded tokenizer-safe segments and aligned FAISS vectors; this turn's 22-case requirements matrix also passed its retrieval/index feature checks. No judged qrels were used. | Retain; retrieval quality and current full workflow PARTIALLY VERIFIED. |
| Embedding | BGE-M3 | No separate index, fixed query/corpus/qrels run, latency/memory sample, or model digest captured here. | NOT EVALUATED. |
| Embedding | Qwen3-Embedding-0.6B | No separate index, fixed query/corpus/qrels run, latency/memory sample, or model digest captured here. | NOT EVALUATED. |

The 22-case requirements matrix run took 31.86 seconds and reported all cases passing. It loaded the configured BGE model and exercised feature/integration contracts, but it did not compare semantic retrieval quality and emitted legacy word-to-subword warnings for chunks; the embedding service's actual tokenizer segmentation is a separate boundary and was not measured by that matrix.

The reported development hardware (16 GB RAM, RTX 3050 6 GB VRAM) is a project note, not a measured allocation in this turn. No peak RAM/VRAM, KV-cache size, throughput, index size, or pinned model checksums were captured. Do not infer model suitability or a production choice from compatibility by download size or the earlier latency-only observations.

## Reproducible decision gate

Keep `qwen3.5:4b` and `BAAI/bge-large-en-v1.5` as experimental controls. A selection change requires exact model digests, identical fixed datasets/prompts/parameters, separate embedding indexes, human relevance/quality labels, measured failure/latency/resource data, and repeatable paired results. No candidate met that gate in this run.

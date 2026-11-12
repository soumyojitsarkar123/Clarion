# Model selection benchmark

**Status: PARTIALLY VERIFIED for execution/resource smoke; NOT EVALUATED for comparative quality. No model switch made.** The configured controls remain `BAAI/bge-large-en-v1.5` for embeddings and `qwen3.5:4b` for Ollama generation. Local Ollama models were queried after granting access to the daemon. A six-request same-prompt smoke compared installed Qwen 4B and 9B; no independent quality labels were available. PyTorch in the validation environment reports CPU-only despite the host having an RTX 3050 6 GB; Ollama itself reported the 4B model at 100% GPU. The result describes this Ollama runtime, not the PyTorch embedding runtime.

| Role | Baseline | Candidate | Quality | Latency/memory | Decision |
|---|---|---|---|---|---|
| Generation | `qwen3.5:4b` (digest `47daeecdce2ea0908bc2ec011ea496ee08fe811763bbb37c7c1e71a42a224e2c`) | `qwen3.5:9b` (digest `c3167a8a9ead9f750d5f4c97fef1843716d7ca5cb2f97d7d793659e227569f53`); 2B not installed | Quality not evaluated. Provider counts: 41 prompt, 57 (4B) / 71 (9B) output tokens per fixed prompt. | Three runs/model; 4B 1.57–1.90 s (warm); 9B 9.27–17.69 s (first includes load); GPU sampled peak 4,451 MiB across run. | Keep 4B; no quality evidence supports promotion |
| Dense embedding | `BAAI/bge-large-en-v1.5` | `BAAI/bge-m3`, `Qwen3-Embedding-0.6B` | No qrels; no comparison | Candidate loads not attempted | Keep baseline; isolated corpus index remains model-specific |
| Reranker | None in active query path | BGE/Qwen reranker candidates | No qrels; not run | Not measured | Do not add without paired evidence |

The timing observations are cold/warm mixed by design: the 4B model was already loaded before the run, while the initial Ollama process snapshot did not include 9B; 9B run 1 includes load. `prompt_eval_count` and `eval_count` are provider-native. GPU memory was sampled every 0.75 seconds and is only a host-level aggregate peak, not model-specific VRAM attribution. All six requests completed with `done_reason=stop`. Raw observations and exact parameters are in `research_artifacts/processed/generation_model_smoke_2026-11-11.json` and `.csv`; regenerate using `research_artifacts/scripts/benchmark_generation_models.py`.

## Required benchmark controls

For comparative quality, pin model repository revision or Ollama digest; use fresh isolated vector indexes per embedding model; preserve query preprocessing and model-specific instruction templates; use identical frozen qrels; report nDCG@k/MRR/P@k/Recall@k, schema/evidence quality, latency, throughput, peak RAM/VRAM, and failures. Do not mix vectors from different models. More output records or a larger model is not a quality metric. The present latency smoke is not the complete model-selection benchmark.

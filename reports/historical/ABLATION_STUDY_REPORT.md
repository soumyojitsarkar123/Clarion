# Ablation Study Report

Current canonical backend relation DB: 1562 candidates, 1562 unlabeled (read-only query). Root `data/relation_dataset.db` is a separate database and is not mixed into this count. Annotation queues have 200 primary and 50 overlap entries; 50 record IDs overlap by design. They are queues, not completed human labels.

| Ablation | Single component change | Outcome | Status |
|---|---|---|---|
| Full pipeline | No ablation | P@k, Recall@k, MRR, latency | NOT RUN |
| No graph augmentation | Disable graph expansion only | Ranking/evidence support/cost | NOT RUN |
| Sparse only | BM25 only | Ranking/latency | NOT RUN |
| Dense only | Dense only | Ranking/latency | NOT RUN |
| No reranking | Skip reranker only | Ranking/latency | NOT RUN |
| No relationship validation | Disable filter only | Human edge precision/recall | BLOCKED: no gold labels |

Use same frozen query set, corpus, versions, top-k and seed, and paired source-document-aware analysis. **Ablations BLOCKED** until valid relevance judgments and human relation labels exist. Synthetic harness results are not empirical quality results.

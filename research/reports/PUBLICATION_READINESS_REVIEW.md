# Publication readiness review

**Decision: NOT READY FOR SUBMISSION.** The project is a functioning engineering prototype with partial pipeline measurements; evidence does not currently support a scientific performance claim or a validated novelty claim.

| Criterion | Status | Evidence/gap |
|---|---|---|
| Defensible contribution and literature gap | PARTIALLY VERIFIED | Candidate provenance direction; not novel/implemented; competing provenance work exists. |
| Algorithm and math verification | PARTIALLY VERIFIED | Chunk and segment checks exist; no source-coordinate composition proof. |
| PDF extraction adequacy | PARTIALLY VERIFIED | 10/11 text yield; no corpus-wide fidelity check; one book unreadable without OCR. |
| Provenance-preserving segmentation | NOT IMPLEMENTED at source page level | Only chunk-local segment spans. |
| Independent human evaluation | BLOCKED | No labels or annotator agreement. |
| Retrieval/graph benchmark | BLOCKED | No valid qrels; graph ablation absent. |
| Fair baselines and ablations | NOT EVALUATED | Chunk counts only; no relevance comparison. |
| Statistical analysis | NOT EVALUATED | No inferential outcomes. |
| Performance characterization | NOT EVALUATED | Smoke runs only. |
| Reproducible figures/tables | PARTIALLY VERIFIED | Corpus/chunk charts have source data/scripts; no retrieval or human-result charts. |
| Verified citations | PARTIALLY VERIFIED | Selected papers linked to authoritative venue pages; review is not exhaustive. |
| Venue and submission conditions | NOT EVALUATED | No official venue/deadline/fee/template survey. |
| Recoverable repository | PARTIALLY VERIFIED | Snapshot and cleanup manifest exist; large pre-existing modified worktree remains. |

## Minimum path to reconsideration

Implement and audit page/source interval propagation; run a licensed, suitable retrieval benchmark plus a document-held-out internal study; collect independent relation labels if relation quality remains in scope; execute paired ablations and controlled performance profiling; complete a broader primary-source review; rerun all analyses from retained raw data; then write a full paper with limitations and a real venue shortlist. If no novel method or empirical finding survives that work, target a systems report or technical artifact rather than claim a research advance.

No venue is recommended at this stage. Selection needs verified fit, recent accepted work, page/template requirements, review model, deadline, open-access policy and fees from official venue pages.

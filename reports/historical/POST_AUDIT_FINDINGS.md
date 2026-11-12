# Post-Package Independent Findings

| Issue | Classification | Finding |
|---|---|---|
| ISS-04 token safety | PARTIALLY FIXED | The prior warning-only implementation was unsafe. Tokenizer-aware segmentation, segment provenance, retrieval support, and regression tests were added. Runtime execution against the real model is still blocked locally. |
| Graph reconciliation | VERIFIED FIXED | `GRAPH_ARTIFACT_REGISTRY.md` directly inventories 46 canonical backend graph artifacts. BSM UUID `65430b13-d3d6-49b8-8386-f34762f0ae9c` is 10/9/34/43/34, not the appendix's 11/10/9/19/34. The latter refers to a non-canonical root-level artifact (`8e355...`) and was an unsupported substitution. |
| Complete manuscript | BLOCKED BY MISSING EVIDENCE | No main manuscript source exists; appendix alone is not a manuscript. |
| Latency | UNRESOLVED | Historical five-query figures are preliminary and non-additive; controlled profiling was not runnable in this environment. |
| Metric terminology | VERIFIED FIXED | `1/(1+d²)` is now described as a bounded monotonic distance-derived ranking score, not calibrated similarity/probability. |
| Synthetic benchmark | PARTIALLY FIXED | The registry correctly designates `bench_7138429ad4e2.json` as a five-seed synthetic regression check and rejects real-world claims. Its 20 records are five clusters of four; bootstrap inference over five clusters is too small for generalizable uncertainty claims. |
| Test/document alignment | PARTIALLY FIXED | `TEST_READY.md` no longer reports conflicting BSM/Thevenin topology. Full affected-suite execution is blocked by the inaccessible project interpreter. |

## Artifact facts established directly from disk

- Canonical BSM run `65430b13-d3d6-49b8-8386-f34762f0ae9c`: 10 nodes, 9 structural edges, 34 semantic edges, 43 total edges, 34 candidate records.
- Canonical Thevenin run `827d8b9c-4444-4d35-ae6a-4a48dc81f0ec`: 16 nodes, 15 structural, 1 semantic, 16 total, 1 candidate record.
- Canonical Thevenin run `fc5f2c9b-85ab-437b-8700-b23ae982ee08`: 18 nodes, 17 structural, 2 semantic, 19 total, 2 candidate records.

Do not claim a CLEAN audit or academic-submission readiness until the blocked runtime validations, controlled latency study, and complete manuscript evidence are supplied.

# Latency Validation Report

## Status: UNRESOLVED

The historical five-query script `scripts/verify_rag_reviewer2.py` is not a controlled production benchmark. It mixes separately timed calls with a separate HTTP request, so its component means are not additive. In particular, `retrieve_with_context()` performs query embedding and vector search again after those stages have already been timed. The reported 125.03 ms gap between the listed 655.28 ms component means and 780.31 ms HTTP mean therefore cannot be assigned to one uninstrumented stage from the stored figures.

The script also hard-codes WSL paths and uses five different queries, omits median/p95/throughput/raw samples, and does not separately report cold versus warm full requests. Existing numbers are classified as preliminary observations only.

Before performance claims, add request-scoped timing at the API boundary and at embedding, index load/search, chunk lookup, context assembly, generation, serialization, and middleware stages; run fixed-query warmups plus repeated cold and warm trials; and save raw JSON/CSV samples with hardware, model, corpus UUID, code revision, median, p95, TTFT, total latency, and throughput.

# Retrieval Evaluation Dataset Protocol

No validated query relevance dataset is present. `research_artifacts/processed/retrieval_queries_template.csv` is header-only. Source PDFs can seed question drafting, but final relevance judgments must be independent of system predictions.

Freeze a source-document-level development/test split before annotation; keep pages and near-duplicate chunks from one source document together. For each query record query ID/text, source document, relevant passage IDs, relevance labels, page/character location, supporting excerpt, assessor ID, timestamp, adjudication, split and run ID. Use two independent assessors for an overlap subset, blinded to system identity and model confidence. Adjudicate disagreements without tuning labels to a kappa target.

Evaluate sparse, dense, hybrid, reranked and graph-augmented approaches on the same frozen corpus, queries and judgments. Record versions, top-k and seeds. For each q: P@k = relevant retrieved/k; Recall@k = relevant retrieved/all relevant; RR(q)=1/rank of first relevant, or 0 if none; MRR=mean RR. Define treatment of queries with zero relevant passages before analysis. Use nDCG only for graded labels. Compare paired query results and cluster uncertainty by source document. No scores or confidence intervals exist yet.

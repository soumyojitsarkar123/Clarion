# PDF Research Evidence

No corpus PDF reports Clarion experiments or usable retrieval/graph/model benchmark values. Relevant IR pages 39-40 were rendered and visually inspected.

| Evidence ID | Source PDF | Page | Table/Figure | Extracted observation | Relevance |
|---|---|---:|---|---|---|
| EVD-001 | UNIT IV Information Retrieval Overview.pdf | 39 | Slide/page 39; no numbered table | Recommends test collections with documents, queries and relevance judgments; compare retrieved output against judgments; use same queries/judgments for system comparisons and repeat experiments. | Protocol guidance only; no Clarion result. |
| EVD-002 | UNIT IV Information Retrieval Overview.pdf | 40 | Slide/page 40; no numbered table | Precision = relevant retrieved / total retrieved. Recall = relevant retrieved / total relevant. F1 = 2 × P × R / (P + R). No metric values are reported. | Definitions apply only after independent relevance judgments exist. |
| EVD-003 | AI_Book.pdf | 1-1151 | Not extractable | Clarion parser returned zero words for all 1,151 pages; pypdf recovery-limit warning at page 1002; rendered pages 1-2 blank. Identity/content cannot be verified. | Cannot support claims before source recovery or reviewed OCR. |
| EVD-004 | All corpus PDFs | per-page report | N/A | 10 text-bearing PDFs produced 86 retained chunks, 170 tokenizer-safe segments and 170 FAISS vectors with SQLite mappings aligned in an isolated run. | Processing evidence only; not retrieval quality. |

Figures/tables were not exhaustively extracted across the corpus. The relevant slides have no numbered data tables or underlying numerical results. OCR and image-based digitization were not performed. Machine-readable evidence is in `research_artifacts/processed/research_evidence.csv` and `.json`.

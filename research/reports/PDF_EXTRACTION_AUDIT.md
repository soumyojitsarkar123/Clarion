# PDF extraction audit

**Status:** PARTIALLY VERIFIED; extraction statistics are measured, extraction fidelity is not.

The current corpus evidence enumerates 11 PDFs and 1,495 pages. The current PyPDF2-based Clarion extractor yields text for 10 files (36,073 words in the prior measured run) and zero words for `AI_Book.pdf` (1,151 pages). The old counts should be treated as that run's measurements; source files must be hash-compared before reusing them. Detailed source inventory and page-level extraction data are in `research_artifacts/processed/pdf_source_register.json` and `pdf_page_extraction.csv`.

`AI_Book.pdf` is image-bearing (1,282 image XObjects), no useful title/author metadata, zero extracted words, and the first two rendered pages were reported blank. A pypdf recovery comparison and parser warnings were recorded, including a page recovery limit. This does not establish that the file is corrupt or recoverable: no trusted alternate copy was located, complete visual review was not performed, OCR was not run, and no OCR output is treated as truth.

The corpus contains mixed text/image slides and technical notes. Existing page-level data detects sparse/blank pages by extracted text count but does not validate reading order, table completeness, figure labels, equations, column ordering, or text against page images across all pages. Therefore extraction *yield* is measured; extraction *accuracy/completeness* is not established.

| Check | Result/status |
|---|---|
| Source count/pages and extraction yield | PARTIALLY VERIFIED in captured run; re-hash before reuse |
| Page-level sparse-text screening | PARTIALLY VERIFIED; heuristic, not visual ground truth |
| Tables/figures/equations and reading order | NOT EVALUATED corpus-wide |
| `AI_Book.pdf` recovery | BLOCKED: no verified alternative source |
| OCR and numerical verification | NOT EVALUATED |
| Corpus suitability for general IR/IE claims | INSUFFICIENT: small, heterogeneous course-material collection without relevance/label data |

No additional public data was downloaded. BEIR offers heterogeneous IR tasks but is not a single corpus license; each dataset's original license and use conditions must be checked. SciERC, DocRED and SciREX have task-specific annotations, domains and schemas and should be evaluated separately, not pooled into a synthetic Clarion score.

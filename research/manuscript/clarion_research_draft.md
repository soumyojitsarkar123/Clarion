# Auditable Evidence Provenance for Document-Centric Retrieval: A Research Agenda and System Audit of Clarion

**Working draft — not submission-ready.** This evidence-limited manuscript describes an audited system and candidate research question. It intentionally makes no novel-method or retrieval-quality claim. Major results remain unavailable; do not replace these status statements with estimates.

## Abstract

Clarion is a document intelligence prototype that extracts PDF text, forms section-aware chunks, segments text to fit a BGE embedding model, indexes vectors in FAISS with SQLite metadata, and includes graph and retrieval components. We audited an 11-PDF internal corpus and the current processing path as a first step toward a research evaluation. Ten PDFs yielded extractable text; one image-bearing 1,151-page PDF yielded no words. In an isolated run, 86 retained chunks produced 170 tokenizer-bounded embedding segments and 170 aligned indexed vectors. These measurements establish processing and index alignment, not retrieval relevance, graph quality, or answer faithfulness. Code inspection shows that segment character offsets are local to chunks; PDF page and source-character coordinates are not propagated through extraction and normalization. The existing relation set contains 1,562 unlabeled candidates, and no independent retrieval judgments are available. We therefore find no supported claim that graph augmentation or Clarion improves retrieval. We identify end-to-end source-coordinate provenance as a testable candidate direction, while related attribution and provenance literature prevents a novelty claim without further comparison. We provide falsifiable research questions, protocols, and explicit blockers for a subsequent study.

## Keywords

Document retrieval; retrieval-augmented generation; provenance; evidence attribution; chunking; knowledge graphs; evaluation.

## 1. Introduction

Document-grounded retrieval systems need to retrieve relevant evidence and make the path from that evidence back to the source inspectable. The Clarion codebase combines PDF processing, token-aware embedding segmentation, vector search, graph-oriented relation components, and answer generation. A combination of known components is not by itself a scientific contribution. This paper reports what the current artifacts establish and defines a candidate evaluation program rather than claiming unmeasured gains.

The audited corpus contains 11 PDFs and 1,495 pages. A prior isolated run extracted text from 10 PDFs and indexed 170 segments. One 1,151-page file produced no text. The critical source-traceability issue is structural: chunk-local character offsets do not identify a PDF page or original source interval after normalization. The system can resolve a vector to its segment and parent chunk, but not yet to a verified source-page quote.

Our contributions in this draft are deliberately limited: (1) an evidence-bounded audit of a real document pipeline; (2) a distinction between local segment integrity and source provenance; and (3) falsifiable protocols for evaluating source-coordinate traceability, retrieval, relation labels, and resource costs. Whether source-coordinate propagation constitutes a publishable methodological contribution remains open.

## 2. Problem formulation

Let a source document consist of pages `P={p_1,...,p_n}`. Extraction yields strings `X`; normalization transforms these into `N`; chunking yields retained chunks `C`; tokenizer-safe segmentation yields segments `S`; embedding/indexing maps each segment to vector ID `v`; retrieval returns ranked evidence `R(q)`. The desired provenance function maps every returned segment to a set of half-open source intervals `(document_hash, page, start, end)` and a text verification status. Transformations may be non-injective: a rewrite can map several source characters to normalized output or remove characters. Therefore a single scalar offset is not generally sufficient; the mapping must preserve interval relations and mark ambiguous/lossy transforms.

Retrieval quality is a distinct objective. Given query `q` and judged relevant passages `G_q`, retrieval metrics such as Recall@k and reciprocal rank assess ranking; valid source coordinates assess traceability. Neither implies the other.

## 3. Related work

RAG was established as retrieval over non-parametric memory for knowledge-intensive NLP (Lewis et al., 2020). GraphRAG and related systems study graph-mediated context construction; their existence means graph+RAG is not a standalone Clarion novelty. BEIR provides heterogeneous zero-shot IR tasks (Thakur et al., 2021), while SciERC, DocRED, and SciREX provide relation or scientific document extraction benchmarks with distinct schemas and domains. LongRAG studies long-context retrieval and multi-hop QA. Provenance (EMNLP Industry 2024), VISA (ACL 2025), and TROVE (ACL 2025) address source attribution or fine-grained provenance and are close prior work. The selected citations are not a complete systematic review.

## 4. System and implementation audit

The pipeline uses PyPDF2 extraction and regex normalization, section-aware word chunking, BAAI/bge-large-en-v1.5 token-offset segmentation, SQLite metadata, and FAISS vectors. The measured model output is 1,024-dimensional; current configuration exposes a 512-token maximum. Segment bounds are local to a chunk. Retrieval resolves segment IDs to parent chunks and returns segment text and section/position metadata.

The isolated corpus run reports 86 retained chunks, 170 segments, 170 vectors. The index-count alignment is a storage invariant, not a relevance result. `AI_Book.pdf` returned no extracted text and was not OCRed. No production database was used for that indexing run.

## 5. Experimental protocol

The next study should test RQ1 source-coordinate validity, RQ2 chunking effects on retrieval and provenance, RQ3 graph augmentation effects, RQ4 relation quality using independent human labels, and RQ5 controlled resource costs. Baselines include BM25, dense BGE, hybrid, current retrieval, and graph-on/off under identical candidate budgets. Use document-disjoint splits; query is the paired retrieval unit with clustering by source document. Use document-clustered bootstrap confidence intervals and correct multiple planned comparisons. Full protocols are in `../protocols/RESEARCH_QUESTIONS.md` and `../protocols/RETRIEVAL_EVALUATION_PROTOCOL.md`.

## 6. Results available now

Corpus processing and chunk-size sensitivity are described in the existing raw and processed files under `research_artifacts/`. The extraction yield and segmentation counts are descriptive. No retrieval qrels or independent judgments exist, so P@k, Recall@k, MRR and nDCG are unavailable. Relation labels are absent, so relation precision/recall/F1 and annotator agreement are unavailable. Controlled latency and memory samples are also unavailable. These are missing results, not zero-valued results.

## 7. Discussion

The current evidence supports a limited systems statement: Clarion can process the readable internal PDFs through chunking, local BGE segmentation, and FAISS indexing with aligned vector metadata. It does not support an effectiveness claim. Source provenance is an identifiable engineering and research gap because the chain stops at chunk-local offsets. Related work on claim attribution, visual source attribution, and sentence-level provenance means any contribution must specify what new coordinate propagation, correctness guarantee, benchmark, or empirical finding Clarion adds.

## 8. Limitations and threats to validity

The corpus is small, heterogeneous, and mostly course material. One file is not text-extractable, and extraction fidelity has not been validated against page images across the corpus. There are no independent retrieval labels, relation annotations, or human agreement results. Prior chunk-count experiments do not evaluate retrieval. A single embedding model and parser configuration limit generalization. The literature search is selected rather than systematic. Results from the isolated corpus run are not a public benchmark comparison.

## 9. Reproducibility statement

The prior processing outputs and scripts are retained under `research_artifacts/`; raw, processed, and derived files are separated there. The source PDFs remain under `docs/`. Record exact source hashes, configuration, commit/worktree patch, model revision, package versions, and random seeds in any new run. Existing outputs are point-in-time evidence; verify hashes before reuse. No new benchmark dataset is redistributed here.

## 10. Conclusion

Clarion's current engineering evidence is not scientific validation. No supported result shows graph or retrieval improvement, relation validity, or end-to-end PDF provenance. The next credible step is an implemented source-coordinate mapping with audited references and an independently judged retrieval evaluation, followed by a renewed novelty review. Submission readiness is not established.

## References

Lewis, P. et al. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS. https://proceedings.neurips.cc/paper/2020/hash/6b493230-Abstract.html

Edge, D. et al. 2024. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv preprint. https://arxiv.org/abs/2404.16130

Thakur, N. et al. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS Datasets and Benchmarks. https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html

Luan, Y. et al. 2018. Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction. EMNLP. https://aclanthology.org/D18-1360/

Yao, Y. et al. 2019. DocRED: A Large-Scale Document-Level Relation Extraction Dataset. ACL. https://aclanthology.org/P19-1074/

Jain, S. et al. 2020. SciREX: A Challenge Dataset for Document-Level Information Extraction. ACL. https://aclanthology.org/2020.acl-main.670/

Zhao, Q. et al. 2024. LongRAG: A Dual-Perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering. EMNLP. https://aclanthology.org/2024.emnlp-main.1259/

Sankararaman, K. et al. 2024. Provenance: A Light-weight Fact-checker for Retrieval Augmented LLM Generation Output. EMNLP Industry. https://aclanthology.org/2024.emnlp-industry.97/

Kim, H. et al. 2025. VISA: Retrieval Augmented Generation with Visual Source Attribution. ACL. https://aclanthology.org/2025.acl-long.1456/

TROVE. 2025. A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification. ACL. https://aclanthology.org/2025.acl-long.577/

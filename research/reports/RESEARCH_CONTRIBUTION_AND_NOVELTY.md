# Research contribution and novelty assessment

**Assessment date:** 2026-11-11. **Status:** PARTIALLY VERIFIED; novelty not established.

## Decision

Clarion as currently evidenced is an engineering integration of PDF ingestion, section-aware word chunking, tokenizer-offset segmentation, BGE embeddings, FAISS/SQLite persistence, graph/relation components, and retrieval. The combination alone is not a defensible novelty claim. The most testable candidate is **auditable source-coordinate provenance from PDF page text through normalization, retained chunk, embedding segment, vector, and returned evidence**, evaluated jointly for coordinate correctness, retrieval usefulness, and overhead. This is a candidate research direction, not a claim of invention: fine-grained source tracing and RAG attribution are active areas, and the literature review is not exhaustive.

The current implementation does not preserve this end-to-end chain. Extraction returns normalized whole-document text; chunk position is word-based and can be cumulative after overlap/merge; embedding `char_start/end` are local to each chunk; retrieval returns segment text plus parent chunk/section/position but no PDF page or original source interval. Therefore the candidate contribution is **not implemented or empirically validated**.

## Novelty matrix

| Claim area | Established work / current Clarion | Classification | Evidence |
|---|---|---|---|
| RAG and dense retrieval | Foundational RAG combines parametric generation with retrieved non-parametric memory; Clarion implements retrieval + generation components. | Established method / integration | Lewis et al. 2020. |
| Graph-assisted retrieval | GraphRAG and other graph-augmented methods already combine extracted entities/relations and retrieval. Clarion has graph construction and retrieval components. | Engineering integration; no measured advantage | Edge et al. 2024 preprint; empirical comparison absent. |
| Tokenizer-aware segmentation | Clarion splits over-limit chunks with fast-tokenizer offsets and preserves segment-to-chunk text spans. This is useful implementation work, not by itself a new segmentation algorithm. | Existing engineering technique | Prior tests and isolated corpus run. |
| PDF-source coordinate continuity | Current pipeline lacks page/source interval mapping. Proposed research implementation would add it. Related attribution/provenance literature means novelty must be narrowly compared. | Candidate method contribution, unproven | Provenance audit; TROVE, VISA, Provenance (EMNLP Industry). |
| Graph relation validity | 1,562 candidates are unlabeled in the canonical DB; no precision/recall or inter-annotator agreement exists. | Hypothesis only | Annotation audit. |
| Graph/RAG gains | No independent retrieval relevance labels or controlled graph ablation. | Not evaluated | Retrieval dataset absent. |
| Chunk sensitivity | Eight chunk configurations were run for count/segmentation outputs, not retrieval quality. | Descriptive engineering result | `research_artifacts/processed/chunking_configuration_sensitivity.csv`. |

## Literature comparison (selected, not exhaustive)

| Study | Year / venue | Problem and method | Dataset/evaluation | Limitations / relevance |
|---|---|---|---|---|
| Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks” | 2020, NeurIPS | RAG with retrieved non-parametric memory and seq2seq generator. | Knowledge-intensive NLP tasks; task metrics. | Foundational baseline, not fine-grained PDF coordinate provenance. DOI: 10.5555/3495724.3496517. |
| Edge et al., “From Local to Global: A Graph RAG Approach to Query-Focused Summarization” | 2024, arXiv preprint | Entity graph/community summaries for global sensemaking. | Query-focused summarization evaluation. | Preprint; graph-RAG combination is established, so not Clarion novelty. DOI: none assigned. |
| Thakur et al., “BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models” | 2021, NeurIPS Datasets & Benchmarks | Broad zero-shot IR evaluation. | 18 heterogeneous retrieval datasets. | Dataset licenses vary; evaluate each task separately. DOI: 10.48550/arXiv.2104.08663. |
| Luan et al., “Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction” (SciERC) | 2018, EMNLP | Joint scientific IE. | SciERC annotations. | Scientific abstracts; not a PDF source-coordinate benchmark. DOI: 10.18653/v1/D18-1360. |
| Yao et al., “DocRED: A Large-Scale Document-Level Relation Extraction Dataset” | 2019, ACL | Document-level relation extraction with evidence. | DocRED/Wikipedia documents. | Different domain and entity/relation schema; no automatic transfer to Clarion labels. DOI: 10.18653/v1/P19-1074. |
| Jain et al., “SciREX: A Challenge Dataset for Document-Level Information Extraction” | 2020, ACL | Scientific document-level extraction and n-ary relations. | Scientific papers. | Stronger long-document relation benchmark candidate; licensing and task compatibility require dataset-specific review. DOI: 10.18653/v1/2020.acl-main.670. |
| Zhao et al., “LongRAG: A Dual-Perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering” | 2024, EMNLP | Long-context + factual retrieval perspectives, with ablations. | Three multi-hop datasets. | More directly relevant to long-context evaluation than internal slides. DOI: 10.18653/v1/2024.emnlp-main.1259. |
| Sankararaman et al., “Provenance: A Light-weight Fact-checker for Retrieval Augmented LLM Generation Output” | 2024, EMNLP Industry | Trace claims to retrieved context chunks. | RAG factuality evaluation. | Closely related to evidence attribution; narrows novelty. DOI: 10.18653/v1/2024.emnlp-industry.97. |
| Kim et al., “VISA: Retrieval Augmented Generation with Visual Source Attribution” | 2025, ACL | Visual-document RAG with source attribution. | Visual document QA. | PDF/page attribution is not unclaimed territory. DOI: 10.18653/v1/2025.acl-long.1456. |
| “TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification” | 2025, ACL | Fine-grained source sentence tracing and relationship labels. | Provenance benchmark. | Closest conceptual overlap; Clarion must distinguish coordinate propagation through its transformations and evaluate against this prior work. DOI: 10.18653/v1/2025.acl-long.577. |
| “The Bottleneck Is Retrieval, Not Reading…” | 2026, Findings of EMNLP | Long-context QA diagnosis with evidence-based retrieval. | Published ACL 2026 benchmark/evaluation. | Very recent; shows empirical retrieval diagnosis is expected. DOI: 10.18653/v1/2026.findings-emnlp.2222. |

## Contribution threshold

Before claiming a contribution: (1) implement source intervals including page identity and transform mapping/ambiguity; (2) compare against chunk-local baseline and manual audited spans; (3) quantify correctness, coverage, storage/latency overhead, and retrieval effect; (4) compare with TROVE/VISA/Provenance and other source attribution methods; (5) publish scripts/raw data and failure cases. If this yields only correct metadata plumbing without a generalizable method or useful empirical finding, present it as a systems artifact rather than a methodological contribution.

## References

- Lewis et al. (2020), [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/6b493230-Abstract.html).
- Edge et al. (2024), [arXiv preprint](https://arxiv.org/abs/2404.16130).
- Thakur et al. (2021), [NeurIPS Datasets & Benchmarks](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html).
- Luan et al. (2018), [ACL Anthology](https://aclanthology.org/D18-1360/).
- Yao et al. (2019), [ACL Anthology](https://aclanthology.org/P19-1074/).
- Jain et al. (2020), [ACL Anthology](https://aclanthology.org/2020.acl-main.670/).
- Zhao et al. (2024), [ACL Anthology](https://aclanthology.org/2024.emnlp-main.1259/).
- Sankararaman et al. (2024), [ACL Anthology](https://aclanthology.org/2024.emnlp-industry.97/).
- Kim et al. (2025), [ACL Anthology](https://aclanthology.org/2025.acl-long.1456/).
- TROVE (2025), [ACL Anthology](https://aclanthology.org/2025.acl-long.577/).
- (2026), [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.2222/).

Publication metadata was checked against the linked publisher/venue pages; Edge et al. is explicitly an arXiv preprint. Venue selection and manuscript submission deadlines were not investigated and remain not evaluated.

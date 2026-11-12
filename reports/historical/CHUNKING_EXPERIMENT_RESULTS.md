# Chunking Experiment Results

The isolated sweep used the same extracted text, real Clarion chunking/merge methods and local BGE tokenizer for chunk sizes 256, 384, 512 and 768 words, with overlaps 0 or 50 words; minimum size stayed 100. It measured counts and token lengths, not vector quality or retrieval.

On DAA FINAL (87 pages, 19,830 words), at overlap 50 the sweep produced retained counts 96, 61, 44 and 30 and embedding segment counts 129, 111, 100 and 91 for chunk sizes 256, 384, 512 and 768 respectively. At overlap 0, retained counts were 77, 53, 40 and 28; segment counts were 105, 106, 93 and 88. Raw results for all PDFs are in `research_artifacts/processed/chunking_configuration_sensitivity.csv`.

Larger chunks reduce raw records but tokenizer segmentation keeps segment counts relatively close. This does not establish better retrieval or justify changing the production default. Retrieval relevance, source coverage, latency and storage effects were not measured.

Charts: `research_artifacts/figures/chunking_config_sensitivity_daa.png`, `embedding_segment_token_lengths.png`, and `pdf_chunk_counts.png`. **Count sensitivity VERIFIED; quality comparison NOT EVALUATED.**

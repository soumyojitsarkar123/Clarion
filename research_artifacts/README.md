# Clarion research artifacts

Generated 2026-11-11 from the current checkout. `raw/` contains page-level extraction text and retained PDF renders; `processed/` contains machine-readable counts, annotation/retrieval templates and inventory; `tables/` contains convenient CSV copies; `figures/` contains charts; `scripts/` contains the extraction, exact-code chunk simulation, isolated indexing, report and chart generators.

The full embedding/index run used `Clarion-Backend/.venv-validation/Scripts/python.exe`, the current Clarion services, local `BAAI/bge-large-en-v1.5`, and an isolated root at `isolated_pdf_index_2026-11-11/`. To rerun the indexing workflow without overwriting this evidence, pass a new empty output directory as the first argument to `run_isolated_pdf_index.py`. The application may refuse an unavailable embedding model; never label deterministic fallback vectors as BGE vectors.

Run the parser, chunk, report and chart scripts from the repository root with the validation interpreter. Source PDFs remain under `docs/`. The checks performed here did not establish retrieval quality, human relation quality, statistical significance or controlled performance.

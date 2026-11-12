# Dependency and model manifest

The exact validation environment's `pip freeze`, Python/platform details, BGE cache revision and tokenizer/config file hashes, Ollama model digests, host GPU, and runtime caveats are recorded in [`research_artifacts/processed/software_dependency_model_manifest_2026-11-11.json`](../../research_artifacts/processed/software_dependency_model_manifest_2026-11-11.json).

The generation smoke raw data and regeneration script are `research_artifacts/processed/generation_model_smoke_2026-11-11.json` and `research_artifacts/scripts/benchmark_generation_models.py`. The output quality comparison is **NOT EVALUATED**. BGE weight files were cached; this manifest records the Hugging Face revision and critical tokenizer/config hashes rather than every weight shard hash. Ollama digests identify the local generation models. These are local-environment manifests, not a universal lockfile.

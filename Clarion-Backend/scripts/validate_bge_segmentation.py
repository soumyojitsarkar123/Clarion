"""Controlled real-model validation of Clarion's embedding segmentation path."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.chunk import Chunk
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.retrieval_service import RetrievalService
from utils.config import settings


def source_for_token_count(tokenizer, target: int) -> str:
    words = []
    while True:
        candidate = " ".join(words + ["validation"])
        count = len(tokenizer.encode(candidate, add_special_tokens=True, truncation=False))
        if count > target:
            return " ".join(words)
        words.append("validation")


def main() -> None:
    temporary_root = Path(tempfile.mkdtemp(prefix="clarion-bge-validation-"))
    original = settings.data_dir, settings.vectorstore_dir, settings.dataset_dir
    settings.data_dir = temporary_root / "data"
    settings.vectorstore_dir = settings.data_dir / "vectorstore"
    settings.dataset_dir = settings.data_dir / "datasets"
    settings.data_dir.mkdir(parents=True)
    settings.vectorstore_dir.mkdir(parents=True)
    try:
        service = EmbeddingService()
        service._load_model()
        tokenizer = service.model.tokenizer
        limit = service._model_token_limit()
        exact = source_for_token_count(tokenizer, limit)
        cases = {
            "under_limit": source_for_token_count(tokenizer, limit - 1),
            "exact_limit": exact,
            "over_limit": exact + " validation",
            "multi_limit": ("Unicode—punctuation: café naïve; αβγ. " * 100).strip(),
            "whitespace": "   \n\t  ",
            "empty": "",
        }
        report = {"limit": limit, "dimension": service.model.get_sentence_embedding_dimension(), "cases": {}}
        saved_chunks = []
        for index, (name, source) in enumerate(cases.items()):
            chunk = Chunk(chunk_id=f"source-{name}", document_id="validation-doc", content=source, position_index=index, word_count=len(source.split()))
            segments, vectors = service.generate_embedding_segments([chunk])
            token_counts = [len(tokenizer.encode(s.content, add_special_tokens=True, truncation=False)) for s in segments]
            reconstructs = "".join(s.content for s in segments) == source
            report["cases"][name] = {
                "source_tokens": len(tokenizer.encode(source, add_special_tokens=True, truncation=False)),
                "segments": len(segments), "segment_tokens": token_counts,
                "reconstructs_exactly": reconstructs, "vectors": len(vectors),
                "vector_dimension": len(vectors[0]) if vectors else 0,
            }
            assert all(count <= limit for count in token_counts), name
            assert reconstructs or not source.strip(), name
            assert len(segments) == len(vectors)
            if source.strip():
                saved_chunks.append(chunk)

        # Integration: persist parent chunks, persist segment mappings/index, then retrieve.
        integration_chunk = Chunk(chunk_id="source-integration", document_id="integration-doc", content=cases["multi_limit"], position_index=0, word_count=len(cases["multi_limit"].split()))
        ChunkingService()._save_chunks([integration_chunk])
        segments, vectors = service.generate_embedding_segments([integration_chunk])
        service.create_vector_index("integration-doc", segments, vectors)
        results = RetrievalService().retrieve("integration-doc", "café unicode punctuation", top_k=3)
        assert results
        assert all(result.chunk_id == integration_chunk.chunk_id for result in results)
        assert all(result.content in integration_chunk.content for result in results)
        report["integration"] = {"segments": len(segments), "retrieved": len(results), "source_chunk_ids": [r.chunk_id for r in results]}
        print(json.dumps(report, indent=2))
    finally:
        settings.data_dir, settings.vectorstore_dir, settings.dataset_dir = original
        shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    main()

"""Exercise an isolated SQLite/FAISS backup, restore, retrieval, and interruption path."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "Clarion-Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from models.chunk import Chunk
from services.embedding_service import EmbeddingService
from utils.config import settings
from vectorstore import VectorStore


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    run_id = "recovery-rehearsal-final-2026-11-12"
    base = BACKEND / ".venv-validation" / run_id
    if base.exists():
        raise SystemExit(f"Refusing to overwrite existing rehearsal data: {base}")

    source_data = base / "source" / "data"
    source_vectors = source_data / "vectorstore"
    source_logs = base / "source" / "logs"
    source_dataset = source_data / "datasets"
    for directory in (source_data, source_vectors, source_logs, source_dataset):
        directory.mkdir(parents=True, exist_ok=True)

    settings.data_dir = source_data
    settings.vectorstore_dir = source_vectors
    settings.logs_dir = source_logs
    settings.dataset_dir = source_dataset
    settings.database_url = f"sqlite:///{source_data / 'clarion.db'}"

    document_id = "backup-restore-doc"
    chunk = Chunk(
        chunk_id="backup-restore-chunk-0",
        document_id=document_id,
        content="Isolated recovery rehearsal evidence for vector mapping.",
        position_index=0,
        word_count=7,
        source_provenance={},
        raw_chunk_ids=["backup-restore-chunk-0"],
    )
    service = EmbeddingService()
    service.active_model_id = "clarion-hash-fallback-v1"
    vector = service._fallback_embed_text(chunk.content)
    source_store = service.create_vector_index(document_id, [chunk], [vector])

    backup_db = base / "backup" / "clarion.db"
    backup_db.parent.mkdir(parents=True)
    with sqlite3.connect(service.db_path) as source, sqlite3.connect(backup_db) as backup:
        source.backup(backup)

    restored_data = base / "restored" / "data"
    restored_vectors = restored_data / "vectorstore"
    restored_data.mkdir(parents=True)
    restored_vectors.mkdir(parents=True)
    restored_db = restored_data / "clarion.db"
    shutil.copy2(backup_db, restored_db)
    shutil.copy2(source_store.index_path, restored_vectors / source_store.index_path.name)
    shutil.copy2(source_store.metadata_path, restored_vectors / source_store.metadata_path.name)

    settings.data_dir = restored_data
    settings.vectorstore_dir = restored_vectors
    settings.logs_dir = base / "restored" / "logs"
    settings.dataset_dir = restored_data / "datasets"
    settings.database_url = f"sqlite:///{restored_db}"
    settings.logs_dir.mkdir(parents=True)
    settings.dataset_dir.mkdir(parents=True)

    with sqlite3.connect(restored_db) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        mapped = connection.execute(
            "SELECT vector_index, vector_id FROM embedding_vector_map "
            "WHERE document_id=? ORDER BY vector_index",
            (document_id,),
        ).fetchall()
    if integrity != "ok":
        raise AssertionError(f"SQLite integrity check failed: {integrity}")

    restored_service = EmbeddingService()
    restored_service._load_model = lambda: None
    restored_store = restored_service.get_vector_store(document_id)
    if restored_store is None:
        raise AssertionError("Restored FAISS index did not match the restored database map")
    hits = restored_store.search(vector, k=1)
    if not hits or hits[0][0] != chunk.chunk_id:
        raise AssertionError(f"Restored retrieval returned unexpected result: {hits}")
    if len(mapped) != restored_store.index.ntotal:
        raise AssertionError("Restored SQLite vector map count differs from FAISS")

    interrupted_vectors = base / "interrupted" / "vectorstore"
    interrupted_vectors.mkdir(parents=True)
    settings.vectorstore_dir = interrupted_vectors
    interrupted_store = VectorStore(
        document_id,
        model_id="clarion-hash-fallback-v1",
        model_revision="1",
    )
    import faiss

    faiss.write_index(restored_store.index, str(interrupted_store.index_path))
    interrupted_index_rejected = not interrupted_store._load_index()
    if not interrupted_index_rejected:
        raise AssertionError("Index without its metadata was incorrectly accepted")

    output = ROOT / "research_artifacts" / "processed" / "backup_restore_rehearsal_2026-11-12.json"
    artifacts = [backup_db, restored_db, restored_store.index_path, restored_store.metadata_path]
    result = {
        "status": "PASS",
        "scope": "isolated SQLite and FAISS files only",
        "production_data_accessed": False,
        "database_integrity": integrity,
        "database_backup_sha256": sha256(backup_db),
        "database_restore_sha256": sha256(restored_db),
        "mapped_vector_ids": [row[1] for row in mapped],
        "mapped_vector_count": len(mapped),
        "faiss_vector_count": int(restored_store.index.ntotal),
        "faiss_dimension": int(restored_store.index.d),
        "retrieval_top1_vector_id": hits[0][0],
        "interrupted_index_without_metadata_rejected": interrupted_index_rejected,
        "interrupted_recovery_action": "rebuild index and metadata from durable segments and vector map",
        "artifact_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in artifacts},
        "isolated_directory": str(base.relative_to(ROOT)),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

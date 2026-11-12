"""Capture a recoverable worktree snapshot without creating a ZIP archive."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "research_artifacts" / "engineering-final-blocker-resolution-2026-11-12"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str, binary: bool = False) -> bytes | str:
    result = subprocess.check_output(["git", *args], cwd=ROOT)
    return result if binary else result.decode("utf-8", errors="replace")


def main() -> int:
    if SNAPSHOT.exists():
        raise SystemExit(f"Refusing to overwrite existing snapshot: {SNAPSHOT}")
    SNAPSHOT.mkdir(parents=True)
    (SNAPSHOT / "untracked-files").mkdir()
    (SNAPSHOT / "preserved-databases" / "Clarion-Backend" / "data").mkdir(parents=True)

    head = git("rev-parse", "HEAD").strip()
    status = git("status", "--short", "--untracked-files=all")
    (SNAPSHOT / "head.txt").write_text(head + "\n", encoding="utf-8")
    (SNAPSHOT / "git-status.txt").write_text(status, encoding="utf-8")
    patch_path = SNAPSHOT / "tracked-worktree.patch"
    patch_path.write_bytes(git("diff", "--binary", "HEAD", binary=True))

    untracked_raw = git("ls-files", "--others", "--exclude-standard", "-z", binary=True)
    untracked = [item.decode("utf-8", errors="surrogateescape") for item in untracked_raw.split(b"\0") if item]
    copied_files = []
    for relative_text in untracked:
        relative = Path(relative_text)
        source = ROOT / relative
        if not source.is_file():
            continue
        target = SNAPSHOT / "untracked-files" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_files.append({
            "path": relative_text,
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        })

    database_files = []
    db_root = ROOT / "Clarion-Backend" / "data"
    for name in ("clarion.db", "clarion.db-wal", "clarion.db-shm"):
        source = db_root / name
        if not source.is_file():
            continue
        target = SNAPSHOT / "preserved-databases" / "Clarion-Backend" / "data" / name
        shutil.copy2(source, target)
        source_hash, copy_hash = sha256(source), sha256(target)
        if source_hash != copy_hash:
            raise RuntimeError(f"Database preservation copy hash mismatch: {name}")
        database_files.append({
            "source": str(source), "snapshot_path": str(target.relative_to(SNAPSHOT)),
            "bytes": target.stat().st_size, "sha256": copy_hash, "copy_matches_source": True,
        })

    existing_readiness_zip = ROOT / "reports" / "current" / "Clarion_Software_Readiness_Work_2026-11-12.zip"
    existing_zip = None
    if existing_readiness_zip.exists():
        existing_zip = {"path": str(existing_readiness_zip), "sha256": sha256(existing_readiness_zip)}

    readme = """# Clarion blocker-resolution snapshot

This snapshot preserves the current worktree without a ZIP archive. To restore tracked edits, apply `tracked-worktree.patch` against the recorded HEAD. Restore files under `untracked-files/` to their corresponding repository-relative paths. `preserved-databases/` contains byte-for-byte copies of the Clarion SQLite database, WAL, and SHM files captured with matching hashes. The Hugging Face and Ollama model files remain in their original local caches; their revision/digests are recorded in the verification reports.

The pre-implementation snapshots and existing readiness ZIP remain preserved in their original locations. This snapshot does not overwrite them.
"""
    (SNAPSHOT / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(ROOT),
        "git_head": head,
        "tracked_patch": {
            "path": "tracked-worktree.patch", "bytes": patch_path.stat().st_size,
            "sha256": sha256(patch_path),
        },
        "git_status_entries": len([line for line in status.splitlines() if line.strip()]),
        "untracked_file_count": len(copied_files),
        "untracked_files": copied_files,
        "database_copies": database_files,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "dependency_manifest": "research_artifacts/processed/backend-validation-requirements-2026-11-12.txt",
        },
        "model_identities": {
            "generation_default": "qwen3.5:4b",
            "ollama_model_layer_digest": "sha256:81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490",
            "embedding_default": "BAAI/bge-large-en-v1.5",
            "embedding_revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
            "chunk_baseline_words": 512,
        },
        "existing_readiness_zip": existing_zip,
        "archive_created": False,
    }
    manifest_path = SNAPSHOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "snapshot": str(SNAPSHOT), "git_head": head,
        "untracked_file_count": len(copied_files), "database_files": database_files,
        "tracked_patch_sha256": manifest["tracked_patch"]["sha256"],
        "archive_created": False,
    }, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

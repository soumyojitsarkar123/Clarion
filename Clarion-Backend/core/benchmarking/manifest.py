"""
Experiment manifest models and dataset versioning.
"""

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunManifest:
    run_id: str
    timestamp: str
    seed: int
    embedding_model: str
    llm_provider: str
    use_graph_features: bool
    dataset_path: str
    dataset_version: str
    split: Dict[str, float]
    stratify_key: Optional[str]
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentManifest:
    experiment_id: str
    created_at: str
    description: str
    num_runs: int
    seeds: List[int]
    embedding_model: str
    llm_provider: str
    use_graph_features: bool
    dataset_path: str
    dataset_version: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_dataset_version(dataset_path: Path) -> str:
    """
    Compute a deterministic version hash for relation dataset records.
    """
    if not dataset_path.exists():
        return "missing-dataset"

    conn = sqlite3.connect(str(dataset_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT record_id, document_id, relation_id, relation_type, llm_confidence,
               cooccurrence_score, semantic_similarity, is_valid, created_at
        FROM relation_dataset
        ORDER BY record_id
        """
    )
    rows = cursor.fetchall()
    conn.close()

    payload = [dict(row) for row in rows]
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest[:16]}"


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

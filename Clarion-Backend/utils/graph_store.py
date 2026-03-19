"""
Safe graph serialization using JSON node-link format.
"""

import json
from pathlib import Path
from typing import Optional

import networkx as nx
from networkx.readwrite import json_graph

from utils.config import settings


def graph_json_path(document_id: str) -> Path:
    return settings.data_dir / "graphs" / f"{document_id}.json"


def graph_pickle_path(document_id: str) -> Path:
    return settings.data_dir / "graphs" / f"{document_id}.pickle"


def save_graph_json(graph: nx.DiGraph, document_id: str) -> Path:
    path = graph_json_path(document_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_graph.node_link_data(graph)
    path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
    return path


def load_graph_json(document_id: str) -> Optional[nx.DiGraph]:
    path = graph_json_path(document_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return nx.DiGraph(json_graph.node_link_graph(data, directed=True, multigraph=False))

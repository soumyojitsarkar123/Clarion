"""
Graph router - serves graph data for visualization.
"""

import logging
from fastapi import APIRouter, HTTPException

from services.document_service import DocumentService
from utils.graph_store import graph_json_path, graph_pickle_path
from utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])

document_service = DocumentService()


@router.get("/{document_id}")
async def get_graph(document_id: str):
    """
    Get the knowledge graph for a document in Cytoscape-compatible format.

    - **document_id**: ID of the document
    """
    try:
        document = document_service.get_document(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        cytoscape_path = settings.data_dir / "graphs" / f"{document_id}_cytoscape.json"
        if cytoscape_path.exists():
            import json

            data = json.loads(cytoscape_path.read_text(encoding="utf-8"))
            return data

        json_path = graph_json_path(document_id)

        if json_path.exists():
            import json

            data = json.loads(json_path.read_text(encoding="utf-8"))
            return data

        if settings.allow_legacy_pickle_loading:
            pickle_path = graph_pickle_path(document_id)
            if pickle_path.exists():
                import networkx as nx
                from networkx.readwrite import json_graph

                graph = nx.read_gpickle(pickle_path)
                data = json_graph.node_link_data(graph)
                return data

        raise HTTPException(
            status_code=404,
            detail="Graph not found. Please analyze the document first.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving graph: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve graph")

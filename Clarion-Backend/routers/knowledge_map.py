"""
Knowledge Map router - handles knowledge map retrieval.
"""

import logging
from fastapi import APIRouter, HTTPException

from models.response import KnowledgeMapResponse
from services.knowledge_map_service import KnowledgeMapService
from services.document_service import DocumentService
from services.chunking_service import ChunkingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-map", tags=["knowledge-map"])

knowledge_map_service = KnowledgeMapService()
document_service = DocumentService()
chunking_service = ChunkingService()


def _source_evidence(chunk_ids: list[str], document_id: str) -> list[dict]:
    evidence = []
    for chunk_id in list(dict.fromkeys(chunk_ids or []))[:8]:
        chunk = chunking_service.get_chunk(chunk_id)
        if not chunk or chunk.document_id != document_id:
            continue
        provenance = chunk.source_provenance or {}
        evidence.append({
            "chunk_id": chunk.chunk_id,
            "excerpt": chunk.content[:1200],
            "section_title": chunk.section_title,
            "page_references": provenance.get("page_references", []),
            "source_intervals": provenance.get("source_intervals", []),
            "location_available": bool(provenance.get("location_available")),
        })
    return evidence


@router.get("/{document_id}", response_model=KnowledgeMapResponse)
async def get_knowledge_map(document_id: str):
    """
    Get the knowledge map for a document.
    
    - **document_id**: ID of the document
    """
    try:
        document = document_service.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        knowledge_map = knowledge_map_service.get_knowledge_map(document_id)
        
        if not knowledge_map:
            raise HTTPException(
                status_code=404, 
                detail="Knowledge map not found. Please analyze the document first."
            )
        
        return KnowledgeMapResponse(
            document_id=knowledge_map.document_id,
            extraction_method=(knowledge_map.metadata or {}).get("extraction_method", "unknown"),
            analysis_mode=(knowledge_map.metadata or {}).get("analysis_mode", "unknown"),
            main_topics=[
                {
                    "id": topic.id,
                    "title": topic.title,
                    "description": topic.description,
                    "concept_ids": topic.concept_ids,
                    "subtopic_ids": topic.subtopic_ids
                }
                for topic in knowledge_map.main_topics
            ],
            subtopics=[
                {
                    "id": sub.id,
                    "title": sub.title,
                    "description": sub.description,
                    "parent_topic_id": sub.parent_topic_id,
                    "concept_ids": sub.concept_ids
                }
                for sub in knowledge_map.subtopics
            ],
            dependencies=knowledge_map.get_dependencies(),
            concepts=[
                {
                    "id": concept.id,
                    "name": concept.name,
                    "frequency": getattr(concept, "frequency", 1),
                    "supporting_sections": getattr(concept, "supporting_sections", []),
                    "definition": concept.definition,
                    "context": concept.context,
                    "chunk_ids": concept.chunk_ids,
                    "node_type": "concept",
                    "extraction_status": "extracted_candidate",
                    "source_evidence": _source_evidence(concept.chunk_ids, document_id),
                }
                for concept in knowledge_map.concepts
            ],
            relations=[
                {
                    "id": rel.id,
                    "from_concept": rel.from_concept,
                    "to_concept": rel.to_concept,
                    "relation_type": rel.relation_type.value,
                    "is_candidate": getattr(rel, "is_candidate", True),
                    "source_chunk_id": getattr(rel, "source_chunk_id", None) or (rel.source_chunk_ids[0] if getattr(rel, "source_chunk_ids", None) else (rel.chunk_ids[0] if getattr(rel, "chunk_ids", None) else None)),
                    "source_chunk_ids": getattr(rel, "source_chunk_ids", []) or ([rel.source_chunk_id] if getattr(rel, "source_chunk_id", None) else []),
                    "evidence_text": getattr(rel, "evidence_text", None),
                    "description": rel.description,
                    "confidence": rel.confidence,
                    "validation_status": "candidate_unverified" if getattr(rel, "is_candidate", True) else "validation_state_unknown",
                    "source_evidence": _source_evidence(
                        getattr(rel, "source_chunk_ids", []) or ([getattr(rel, "source_chunk_id", None)] if getattr(rel, "source_chunk_id", None) else []),
                        document_id,
                    ),
                }
                for rel in knowledge_map.relations
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving knowledge map: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

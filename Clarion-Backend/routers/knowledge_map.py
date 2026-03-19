"""
Knowledge Map router - handles knowledge map retrieval.
"""

import logging
from fastapi import APIRouter, HTTPException

from models.response import KnowledgeMapResponse
from services.knowledge_map_service import KnowledgeMapService
from services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-map", tags=["knowledge-map"])

knowledge_map_service = KnowledgeMapService()
document_service = DocumentService()


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
                    "definition": concept.definition,
                    "context": concept.context,
                    "chunk_ids": concept.chunk_ids
                }
                for concept in knowledge_map.concepts
            ],
            relations=[
                {
                    "id": rel.id,
                    "from_concept": rel.from_concept,
                    "to_concept": rel.to_concept,
                    "relation_type": rel.relation_type.value,
                    "description": rel.description,
                    "confidence": rel.confidence
                }
                for rel in knowledge_map.relations
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving knowledge map: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

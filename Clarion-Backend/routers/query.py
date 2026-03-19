"""
Query router - handles RAG-based querying.
"""

import logging
from fastapi import APIRouter, HTTPException, Body

from models.retrieval import QueryRequest
from models.response import QueryResponse
from services.document_service import DocumentService
from services.retrieval_service import RetrievalService
from services.knowledge_map_service import KnowledgeMapService
from services.knowledge_map_service import LLMInterface
from services.rag_audit_service import get_rag_audit_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

document_service = DocumentService()
retrieval_service = RetrievalService()
knowledge_map_service = KnowledgeMapService()
llm = LLMInterface()
rag_audit_service = get_rag_audit_service()


@router.post("/{document_id}", response_model=QueryResponse)
async def query_document(
    document_id: str,
    request: QueryRequest = Body(...)
):
    """
    Query a document using RAG.
    
    - **document_id**: ID of the document to query
    - **request**: Query request with query text, top_k, and options
    """
    try:
        document = document_service.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not document.text_content:
            raise HTTPException(
                status_code=400, 
                detail="Document has no content"
            )
        
        results, context = retrieval_service.retrieve_with_context(
            document_id,
            request.query,
            request.top_k
        )
        
        if not results:
            rag_audit_service.audit_query(
                document_id=document_id,
                query=request.query,
                top_k=request.top_k,
                response_text="No relevant information found in the document.",
                retrieval_results=[],
            )
            return QueryResponse(
                query=request.query,
                results=[],
                response="No relevant information found in the document.",
                knowledge_map=None
            )
        
        prompt = f"""Based on the following context from the document, answer the query.
If the context doesn't contain relevant information to answer the question, say so.

Query: {request.query}

Context:
{context}

Answer:"""
        
        try:
            response_text = llm.generate(prompt)
        except Exception as e:
            logger.warning(f"LLM generation failed: {str(e)}")
            response_text = "Unable to generate response at this time."

        rag_audit_service.audit_query(
            document_id=document_id,
            query=request.query,
            top_k=request.top_k,
            response_text=response_text,
            retrieval_results=results,
        )
        
        result_dicts = [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "score": r.score,
                "section_title": r.section_title,
                "position_index": r.position_index
            }
            for r in results
        ]
        
        knowledge_map = None
        if request.include_knowledge_map:
            km = knowledge_map_service.get_knowledge_map(document_id)
            if km:
                knowledge_map = {
                    "main_topics": [
                        {"id": t.id, "title": t.title}
                        for t in km.main_topics
                    ],
                    "concepts": [
                        {"id": c.id, "name": c.name}
                        for c in km.concepts[:10]
                    ]
                }
        
        return QueryResponse(
            query=request.query,
            results=result_dicts,
            response=response_text,
            knowledge_map=knowledge_map
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

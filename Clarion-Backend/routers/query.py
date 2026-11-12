"""
Query router - handles RAG-based querying.
"""

import asyncio
import logging
import re
from typing import List
from fastapi import APIRouter, HTTPException, Body

from models.retrieval import QueryRequest
from models.response import QueryResponse
from services.document_service import DocumentService
from services.retrieval_service import RetrievalService
from services.knowledge_map_service import KnowledgeMapService
from services.knowledge_map_service import LLMInterface
from services.rag_audit_service import get_rag_audit_service
from services.token_budget_service import TokenBudgetExceeded
from core.llm.base import sanitize_llm_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

document_service = DocumentService()
retrieval_service = RetrievalService()
knowledge_map_service = KnowledgeMapService()
llm = LLMInterface()
rag_audit_service = get_rag_audit_service()


def synthesize_grounded_fallback_answer(query: str, results: list) -> str:
    """Extract and synthesize direct factual answers from retrieved evidence chunks."""
    if not results:
        return "No relevant information found in the document."

    query_terms = set(re.findall(r"\w+", query.lower())) - {
        "who", "what", "where", "when", "why", "how", "is", "are", "the", "a",
        "an", "in", "on", "of", "to", "for", "and", "or", "tell", "me", "about"
    }

    scored_sentences = []
    for r in results:
        raw_text = r.content or ""
        sentences = re.split(r"(?<=[.!?\n])\s+", raw_text)
        for s in sentences:
            cleaned = s.strip()
            if len(cleaned) < 25 or len(cleaned) > 350:
                continue
            words = set(re.findall(r"\w+", cleaned.lower()))
            overlap = len(words & query_terms)
            if overlap > 0:
                scored_sentences.append((overlap, cleaned))

    if scored_sentences:
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        selected = []
        seen = set()
        for _, s in scored_sentences:
            norm = s.lower().strip()
            if norm not in seen:
                seen.add(norm)
                selected.append(s)
            if len(selected) >= 3:
                break
        return " ".join(selected)

    top_chunk = results[0].content.strip()
    sents = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", top_chunk) if len(s.strip()) > 25]
    return " ".join(sents[:2]) if sents else top_chunk[:300]


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
        
        results, _ = retrieval_service.retrieve_with_context(
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
        
        fixed_prompt = f"""Based on the following context from the document, answer the query.
If the context doesn't contain relevant information to answer the question, say so.

Query: {request.query}

Context:
{{context}}

Answer:"""

        packer = llm.token_budget
        try:
            packed = packer.pack_ranked_evidence(fixed_prompt, results)
        except TokenBudgetExceeded as e:
            raise HTTPException(status_code=413, detail=str(e)) from e
        prompt = fixed_prompt.replace("{context}", packed.context)
        
        fallback_used = False
        try:
            llm.last_token_metrics = {}
            is_avail = llm._is_service_available()
            raw_text = await asyncio.to_thread(llm.generate, prompt)
            response_text = sanitize_llm_text(raw_text)
            if (not is_avail) or ("LLM unavailable" in response_text) or ("fallback analysis" in response_text.lower()) or not response_text:
                fallback_used = True
        except Exception as e:
            logger.warning(f"LLM generation failed: {str(e)}")
            fallback_used = True
            response_text = ""

        if fallback_used or not response_text or "LLM unavailable" in response_text or "fallback analysis" in response_text.lower():
            response_text = synthesize_grounded_fallback_answer(request.query, results)
            fallback_used = True

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
                "position_index": r.position_index,
                "document_id": r.document_id,
                "document_name": document.metadata.filename,
                "page_references": r.page_references,
                "source_intervals": r.source_intervals,
                "segment_id": r.segment_id,
                "raw_chunk_ids": r.raw_chunk_ids,
                "embedding_model_id": r.embedding_model_id,
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
            knowledge_map=knowledge_map,
            fallback_used=fallback_used,
            omitted_evidence=list(packed.omitted),
            token_metrics={
                **(llm.last_token_metrics or {}),
                "context_pack_input_tokens": packed.prompt_plan.input_tokens,
                "context_pack_exact": packed.prompt_plan.exact,
                "context_pack_method": packed.prompt_plan.counting_method,
                "included_evidence_count": len(packed.included_ids),
                "omitted_evidence_count": len(packed.omitted),
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

"""
Intelligent Document Structure and Knowledge Mapping System
Main application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.logger import setup_logging
from utils.config import settings, ensure_directories
from routers import (
    upload_router,
    analyze_router,
    knowledge_map_router,
    query_router,
    summary_router,
    status_router,
    dataset_router,
    graph_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    ensure_directories()
    setup_logging(log_level="INFO", log_file=str(settings.logs_dir / "clarion.log"))
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    yield
    logger.info("Application shutdown")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    Intelligent Document Structure and Knowledge Mapping System using RAG and LLM.
    
    This system provides:
    - Document ingestion (PDF, DOCX)
    - Structure-aware chunking
    - Semantic embedding with FAISS
    - Knowledge map extraction with concept relationships
    - Structured hierarchical summaries
    - RAG-based querying
    """,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(analyze_router)
app.include_router(knowledge_map_router)
app.include_router(query_router)
app.include_router(summary_router)
app.include_router(status_router)
app.include_router(dataset_router)
app.include_router(graph_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from services.background_service import BackgroundService

    bg_service = BackgroundService()
    active_jobs = len(bg_service.list_active_jobs())

    return {
        "status": "healthy",
        "services": {
            "database": "ok",
            "embeddings": "ready",
            "graph_engine": "ready",
            "background_processor": "ready",
            "active_jobs": active_jobs,
        },
    }


@app.get("/system-status")
async def system_status():
    """Get detailed system status with job information."""
    from services.background_service import BackgroundService
    from services.knowledge_map_service import LLMInterface
    import psutil
    import os

    bg_service = BackgroundService()
    llm = LLMInterface()
    
    active_jobs = bg_service.list_active_jobs()
    
    # Check LLM availability
    llm_available = llm._is_ollama_available()

    # Get system metrics
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024

        cpu_percent = process.cpu_percent(interval=0.1)
    except Exception:
        memory_mb = 0
        cpu_percent = 0

    return {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "app_status": "running",
        "services": {
            "llm": {
                "status": "available" if llm_available else "unavailable (demo mode)",
                "model": "Ollama" if llm_available else "Demo",
            },
            "database": "ok",
            "embeddings": "ready",
            "graph_engine": "ready",
        },
        "background_jobs": {
            "active": len([j for j in active_jobs if hasattr(j, 'status') and j.status == "processing"]),
            "total_active": len(active_jobs),
            "jobs": [{"id": j.job_id, "status": j.status} if hasattr(j, 'job_id') else {} for j in active_jobs[:5]],
        },
        "system": {
            "memory_mb": round(memory_mb, 2),
            "cpu_percent": cpu_percent,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)

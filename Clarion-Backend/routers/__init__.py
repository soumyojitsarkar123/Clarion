"""
API routers for the application.
"""

from .upload import router as upload_router
from .analyze import router as analyze_router
from .knowledge_map import router as knowledge_map_router
from .query import router as query_router
from .summary import router as summary_router
from .status import router as status_router
from .dataset import router as dataset_router
from .graph import router as graph_router
from .logs import router as logs_router
from .system import router as system_router

__all__ = [
    "upload_router",
    "analyze_router",
    "knowledge_map_router",
    "query_router",
    "summary_router",
    "status_router",
    "dataset_router",
    "graph_router",
    "logs_router",
    "system_router",
]

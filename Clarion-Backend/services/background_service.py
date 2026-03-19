"""
Background Processing Service - Orchestrates document processing pipeline.

Provides persistent job tracking, async execution, and progress monitoring.
"""

import uuid
import sqlite3
import logging
import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from utils.config import settings
from utils.graph_store import graph_json_path, graph_pickle_path
from utils.sqlite import connect as sqlite_connect

logger = logging.getLogger(__name__)

_instance = None


def get_background_service() -> "BackgroundService":
    """Get singleton instance of BackgroundService."""
    global _instance
    if _instance is None:
        _instance = BackgroundService()
    return _instance


class JobStatus(str, Enum):
    """Processing job states."""

    PENDING = "pending"
    PROCESSING = "processing"
    INGESTING = "ingesting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    MAPPING = "mapping"
    GRAPH_BUILDING = "graph_building"
    EVALUATING = "evaluating"
    HIERARCHY = "hierarchy"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactInfo(BaseModel):
    """Information about a processing artifact."""

    stage: str
    name: str
    path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None


class ProcessingJob(BaseModel):
    """Represents a document processing job."""

    job_id: str = Field(..., description="Unique job identifier")
    document_id: str = Field(..., description="Associated document ID")
    status: JobStatus = Field(default=JobStatus.PENDING)
    current_stage: str = Field(default="queued", description="Current processing stage")
    progress_percent: int = Field(default=0, ge=0, le=100)

    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)

    error_message: Optional[str] = Field(None)
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)

    metadata: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, ArtifactInfo] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class JobRepository:
    """
    Repository for job persistence using SQLite.

    Provides CRUD operations for processing jobs with proper
    transaction handling and error recovery.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (settings.data_dir / "clarion.db")
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure jobs table exists."""
        conn = sqlite_connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_jobs (
                job_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                status TEXT NOT NULL,
                current_stage TEXT NOT NULL,
                progress_percent INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                metadata TEXT,
                artifacts TEXT
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_document 
            ON processing_jobs(document_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status 
            ON processing_jobs(status)
        """)

        conn.commit()
        conn.close()

        logger.debug("Job repository initialized")

    def create(self, job: ProcessingJob) -> ProcessingJob:
        """Create a new job record."""
        conn = sqlite_connect(str(self.db_path))
        cursor = conn.cursor()

        artifacts_json = (
            json.dumps(
                {
                    k: {
                        "stage": v.stage,
                        "name": v.name,
                        "path": v.path,
                        "created_at": v.created_at.isoformat(),
                        "size_bytes": v.size_bytes,
                        "checksum": v.checksum,
                    }
                    for k, v in job.artifacts.items()
                }
            )
            if job.artifacts
            else "{}"
        )

        cursor.execute(
            """
            INSERT INTO processing_jobs 
            (job_id, document_id, status, current_stage, progress_percent,
             created_at, started_at, completed_at, error_message, 
             retry_count, max_retries, metadata, artifacts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                job.job_id,
                job.document_id,
                job.status,
                job.current_stage,
                job.progress_percent,
                job.created_at.isoformat(),
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.error_message,
                job.retry_count,
                job.max_retries,
                json.dumps(job.metadata),
                artifacts_json,
            ),
        )

        conn.commit()
        conn.close()

        logger.info(f"Created job {job.job_id} for document {job.document_id}")
        return job

    def get(self, job_id: str) -> Optional[ProcessingJob]:
        """Retrieve job by ID."""
        conn = sqlite_connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM processing_jobs WHERE job_id = ?", (job_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_job(row)

    def get_by_document(self, document_id: str) -> List[ProcessingJob]:
        """Get all jobs for a document."""
        conn = sqlite_connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM processing_jobs WHERE document_id = ? ORDER BY created_at DESC",
            (document_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_job(row) for row in rows]

    def get_latest_by_document(self, document_id: str) -> Optional[ProcessingJob]:
        """Get most recent job for a document."""
        conn = sqlite_connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """SELECT * FROM processing_jobs 
               WHERE document_id = ? 
               ORDER BY created_at DESC 
               LIMIT 1""",
            (document_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_job(row)

    def update(self, job: ProcessingJob) -> ProcessingJob:
        """Update job record."""
        conn = sqlite_connect(str(self.db_path))
        cursor = conn.cursor()

        artifacts_json = (
            json.dumps(
                {
                    k: {
                        "stage": v.stage,
                        "name": v.name,
                        "path": v.path,
                        "created_at": v.created_at.isoformat(),
                        "size_bytes": v.size_bytes,
                        "checksum": v.checksum,
                    }
                    for k, v in job.artifacts.items()
                }
            )
            if job.artifacts
            else "{}"
        )

        cursor.execute(
            """
            UPDATE processing_jobs SET
                status = ?,
                current_stage = ?,
                progress_percent = ?,
                started_at = ?,
                completed_at = ?,
                error_message = ?,
                retry_count = ?,
                metadata = ?,
                artifacts = ?
            WHERE job_id = ?
        """,
            (
                job.status,
                job.current_stage,
                job.progress_percent,
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.error_message,
                job.retry_count,
                json.dumps(job.metadata),
                artifacts_json,
                job.job_id,
            ),
        )

        conn.commit()
        conn.close()

        return job

    def update_progress(self, job_id: str, stage: str, percent: int) -> None:
        """Update job progress efficiently."""
        conn = sqlite_connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE processing_jobs 
            SET current_stage = ?, progress_percent = ?
            WHERE job_id = ?
        """,
            (stage, percent, job_id),
        )

        conn.commit()
        conn.close()

    def add_artifact(
        self, job_id: str, artifact_key: str, artifact: ArtifactInfo
    ) -> None:
        """Add or update an artifact for a job."""
        job = self.get(job_id)
        if not job:
            return

        job.artifacts[artifact_key] = artifact
        self.update(job)

    def get_artifact(self, job_id: str, artifact_key: str) -> Optional[ArtifactInfo]:
        """Get a specific artifact from a job."""
        job = self.get(job_id)
        if not job:
            return None
        return job.artifacts.get(artifact_key)

    def list_active(self) -> List[ProcessingJob]:
        """List all active (non-terminal) jobs."""
        conn = sqlite_connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM processing_jobs 
            WHERE status NOT IN (?, ?, ?)
            ORDER BY created_at DESC
        """,
            (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED),
        )

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_job(row) for row in rows]

    def delete(self, job_id: str) -> bool:
        """Delete a job."""
        conn = sqlite_connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("DELETE FROM processing_jobs WHERE job_id = ?", (job_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted

    def _row_to_job(self, row: sqlite3.Row) -> ProcessingJob:
        """Convert database row to ProcessingJob."""
        metadata = {}
        if row["metadata"]:
            try:
                metadata = (
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                )
            except:
                metadata = {}

        artifacts = {}
        if row["artifacts"]:
            try:
                artifacts_data = (
                    json.loads(row["artifacts"])
                    if isinstance(row["artifacts"], str)
                    else row["artifacts"]
                )
                for k, v in artifacts_data.items():
                    artifacts[k] = ArtifactInfo(
                        stage=v["stage"],
                        name=v["name"],
                        path=v.get("path"),
                        created_at=datetime.fromisoformat(v["created_at"])
                        if v.get("created_at")
                        else datetime.now(),
                        size_bytes=v.get("size_bytes"),
                        checksum=v.get("checksum"),
                    )
            except Exception as e:
                logger.warning(f"Failed to parse artifacts: {e}")

        return ProcessingJob(
            job_id=row["job_id"],
            document_id=row["document_id"],
            status=JobStatus(row["status"]),
            current_stage=row["current_stage"],
            progress_percent=row["progress_percent"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"])
            if row["started_at"]
            else None,
            completed_at=datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None,
            error_message=row["error_message"],
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            metadata=metadata,
            artifacts=artifacts,
        )


class BackgroundService:
    """
    Service for managing background document processing.

    Features:
    - Persistent job tracking
    - Async pipeline execution
    - Progress monitoring
    - Retry with exponential backoff
    - Concurrent processing limits
    - Artifact tracking for partial outputs

    Example:
        service = BackgroundService()
        job_id = await service.submit_job(document_id)

        # Check status
        job = service.get_job(job_id)
        print(f"Progress: {job.progress_percent}%")
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        retry_delay_base: float = 5.0,
        db_path: Optional[Path] = None,
    ):
        """
        Initialize background service.

        Args:
            max_concurrent: Maximum concurrent processing jobs
            retry_delay_base: Base delay for exponential backoff (seconds)
            db_path: Path to SQLite database
        """
        self.max_concurrent = max_concurrent
        self.retry_delay_base = retry_delay_base
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._repository = JobRepository(db_path)
        self._active_tasks: Dict[str, asyncio.Task] = {}

        # logger.info(f"Background service initialized (max_concurrent={max_concurrent})")

    async def submit_job(
        self,
        document_id: str,
        process_func: Callable,
        background_tasks: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """
        Submit a document processing job.

        Args:
            document_id: Document to process
            process_func: Async function to execute pipeline
            background_tasks: Optional FastAPI BackgroundTasks for integration
            **kwargs: Additional arguments for process_func

        Returns:
            job_id for tracking
        """
        existing_job = self._repository.get_latest_by_document(document_id)
        if existing_job and existing_job.status in [
            JobStatus.PENDING,
            JobStatus.PROCESSING,
            JobStatus.INGESTING,
            JobStatus.CHUNKING,
            JobStatus.EMBEDDING,
            JobStatus.MAPPING,
            JobStatus.GRAPH_BUILDING,
            JobStatus.EVALUATING,
            JobStatus.HIERARCHY,
        ]:
            logger.info(
                f"Document {document_id} already has active job: {existing_job.job_id}"
            )
            return existing_job.job_id

        job = ProcessingJob(
            job_id=str(uuid.uuid4()),
            document_id=document_id,
            status=JobStatus.PENDING,
            current_stage="queued",
        )

        self._repository.create(job)

        if background_tasks:
            background_tasks.add_task(
                self._execute_job_safe, job, process_func, **kwargs
            )
        else:
            task = asyncio.create_task(
                self._execute_job_safe(job, process_func, **kwargs)
            )
            self._active_tasks[job.job_id] = task

        logger.info(f"Submitted job {job.job_id} for document {document_id}")
        return job.job_id

    async def submit_job_with_callback(
        self,
        document_id: str,
        process_func: Callable,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        **kwargs,
    ) -> str:
        """
        Submit a job with completion/error callbacks.

        Args:
            document_id: Document to process
            process_func: Async function to execute pipeline
            on_complete: Callback function on successful completion
            on_error: Callback function on error
            **kwargs: Additional arguments for process_func

        Returns:
            job_id for tracking
        """
        existing_job = self._repository.get_latest_by_document(document_id)
        if existing_job and existing_job.status in [
            JobStatus.PENDING,
            JobStatus.PROCESSING,
            JobStatus.INGESTING,
            JobStatus.CHUNKING,
            JobStatus.EMBEDDING,
            JobStatus.MAPPING,
            JobStatus.GRAPH_BUILDING,
            JobStatus.EVALUATING,
            JobStatus.HIERARCHY,
        ]:
            return existing_job.job_id

        job = ProcessingJob(
            job_id=str(uuid.uuid4()),
            document_id=document_id,
            status=JobStatus.PENDING,
            current_stage="queued",
        )

        self._repository.create(job)

        task = asyncio.create_task(
            self._execute_job_with_callbacks(
                job, process_func, on_complete, on_error, **kwargs
            )
        )
        self._active_tasks[job.job_id] = task

        logger.info(f"Submitted job {job.job_id} for document {document_id}")
        return job.job_id

    async def _execute_job_safe(
        self, job: ProcessingJob, process_func: Callable, **kwargs
    ) -> None:
        """Execute job with error handling."""
        try:
            await self._execute_job(job, process_func, **kwargs)
        except Exception as e:
            logger.error(f"Job {job.job_id} execution error: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            self._repository.update(job)
            if job.job_id in self._active_tasks:
                del self._active_tasks[job.job_id]

    async def _execute_job_with_callbacks(
        self,
        job: ProcessingJob,
        process_func: Callable,
        on_complete: Optional[Callable],
        on_error: Optional[Callable],
        **kwargs,
    ) -> None:
        """Execute job with callbacks."""
        try:
            await self._execute_job(job, process_func, **kwargs)
            if on_complete:
                try:
                    await on_complete(job)
                except Exception as e:
                    logger.error(f"Complete callback error: {e}")
        except Exception as e:
            if on_error:
                try:
                    await on_error(job, e)
                except Exception as cb_err:
                    logger.error(f"Error callback error: {cb_err}")
        finally:
            if job.job_id in self._active_tasks:
                del self._active_tasks[job.job_id]

    async def _execute_job(
        self, job: ProcessingJob, process_func: Callable, **kwargs
    ) -> None:
        """
        Execute a job with concurrency control and error handling.

        This method runs within the semaphore to limit concurrent execution.
        """
        async with self._semaphore:
            try:
                # Mark as started
                job.status = JobStatus.PROCESSING
                job.started_at = datetime.now()
                self._repository.update(job)

                logger.info(f"Starting job {job.job_id}")

                # Create progress callback
                async def progress_callback(stage: str, percent: int):
                    job.current_stage = stage
                    job.progress_percent = percent
                    self._repository.update_progress(job.job_id, stage, percent)
                    logger.debug(f"Job {job.job_id}: {stage} ({percent}%)")

                # Execute pipeline
                await process_func(
                    job.document_id, progress_callback=progress_callback, **kwargs
                )

                # Mark as completed
                job.status = JobStatus.COMPLETED
                job.progress_percent = 100
                job.completed_at = datetime.now()
                job.current_stage = "completed"

                logger.info(f"Job {job.job_id} completed successfully")

            except Exception as e:
                logger.error(f"Job {job.job_id} failed: {e}")

                if job.retry_count < job.max_retries:
                    # Schedule retry
                    job.retry_count += 1
                    job.status = JobStatus.PENDING
                    job.error_message = str(e)
                    job.current_stage = f"retry_scheduled (attempt {job.retry_count})"

                    self._repository.update(job)

                    # Exponential backoff
                    delay = self.retry_delay_base * (2 ** (job.retry_count - 1))
                    logger.info(
                        f"Retrying job {job.job_id} in {delay}s (attempt {job.retry_count})"
                    )

                    await asyncio.sleep(delay)
                    asyncio.create_task(self._execute_job(job, process_func, **kwargs))
                else:
                    # Max retries reached
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    job.completed_at = datetime.now()
                    logger.error(
                        f"Job {job.job_id} failed permanently after {job.max_retries} retries"
                    )

            finally:
                # Always update final state
                self._repository.update(job)

    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job by ID."""
        return self._repository.get(job_id)

    def get_document_jobs(self, document_id: str) -> List[ProcessingJob]:
        """Get all jobs for a document."""
        return self._repository.get_by_document(document_id)

    def get_latest_job(self, document_id: str) -> Optional[ProcessingJob]:
        """Get most recent job for a document."""
        return self._repository.get_latest_by_document(document_id)

    def list_active_jobs(self) -> List[ProcessingJob]:
        """List all active jobs."""
        return self._repository.list_active()

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or processing job.

        Note: Can only cancel if job hasn't started yet or is between stages.
        """
        job = self._repository.get(job_id)

        if not job:
            return False

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            logger.warning(f"Cannot cancel job {job_id} - already in terminal state")
            return False

        # Mark as cancelled
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        job.current_stage = "cancelled"
        self._repository.update(job)

        logger.info(f"Cancelled job {job_id}")
        return True

    def get_job_history(self, document_id: str) -> Dict[str, Any]:
        """
        Get complete processing history for a document.

        Returns:
            Dictionary with job history and current status
        """
        jobs = self._repository.get_by_document(document_id)

        if not jobs:
            return {"document_id": document_id, "has_history": False, "jobs": []}

        latest = jobs[0]

        return {
            "document_id": document_id,
            "has_history": True,
            "latest_job_id": latest.job_id,
            "latest_status": latest.status,
            "latest_progress": latest.progress_percent,
            "total_jobs": len(jobs),
            "completed_jobs": sum(1 for j in jobs if j.status == JobStatus.COMPLETED),
            "failed_jobs": sum(1 for j in jobs if j.status == JobStatus.FAILED),
            "jobs": [
                {
                    "job_id": j.job_id,
                    "status": j.status,
                    "current_stage": j.current_stage,
                    "progress_percent": j.progress_percent,
                    "created_at": j.created_at.isoformat(),
                    "started_at": j.started_at.isoformat() if j.started_at else None,
                    "completed_at": j.completed_at.isoformat()
                    if j.completed_at
                    else None,
                    "error_message": j.error_message,
                    "retry_count": j.retry_count,
                }
                for j in jobs
            ],
        }

    def check_recoverable(self, document_id: str) -> Dict[str, Any]:
        """
        Check if a failed job can be recovered based on partial outputs.

        Returns:
            Dictionary with recovery info and available artifacts
        """
        job = self._repository.get_latest_by_document(document_id)

        if not job:
            return {"recoverable": False, "reason": "No job found"}

        if job.status == JobStatus.COMPLETED:
            return {"recoverable": False, "reason": "Job already completed"}

        if job.status == JobStatus.CANCELLED:
            return {"recoverable": False, "reason": "Job was cancelled"}

        artifacts = {}
        available_stages = []

        from pathlib import Path
        from utils.config import settings

        doc_path = settings.data_dir / "documents" / f"{document_id}.txt"
        if doc_path.exists():
            artifacts["document"] = {"path": str(doc_path), "stage": "ingestion"}
            available_stages.append("ingestion")

        chunks_path = settings.data_dir / "chunks" / f"{document_id}.json"
        if chunks_path.exists():
            artifacts["chunks"] = {"path": str(chunks_path), "stage": "chunking"}
            available_stages.extend(["ingestion", "chunking"])

        vector_path = settings.vectorstore_dir / f"{document_id}.index"
        if vector_path.exists():
            artifacts["embeddings"] = {"path": str(vector_path), "stage": "embedding"}
            available_stages.extend(["ingestion", "chunking", "embedding"])

        graph_path = graph_json_path(document_id)
        if graph_path.exists():
            artifacts["graph"] = {"path": str(graph_path), "stage": "graph_building"}
            available_stages.extend(
                ["ingestion", "chunking", "embedding", "mapping", "graph_building"]
            )
        elif settings.allow_legacy_pickle_loading:
            legacy_graph = graph_pickle_path(document_id)
            if legacy_graph.exists():
                artifacts["graph"] = {
                    "path": str(legacy_graph),
                    "stage": "graph_building",
                }
                available_stages.extend(
                    ["ingestion", "chunking", "embedding", "mapping", "graph_building"]
                )

        knowledge_map_path = (
            settings.data_dir / "knowledge_maps" / f"{document_id}.json"
        )
        if knowledge_map_path.exists():
            artifacts["knowledge_map"] = {
                "path": str(knowledge_map_path),
                "stage": "mapping",
            }
            available_stages.extend(["ingestion", "chunking", "embedding", "mapping"])

        return {
            "recoverable": len(artifacts) > 0,
            "current_stage": job.current_stage,
            "available_stages": available_stages,
            "artifacts": artifacts,
            "job_status": job.status,
            "error_message": job.error_message,
        }

    async def recover_job(
        self,
        document_id: str,
        process_func: Callable,
        skip_completed_stages: bool = True,
        **kwargs,
    ) -> Optional[str]:
        """
        Attempt to recover a failed job from partial outputs.

        Args:
            document_id: Document ID to recover
            process_func: Pipeline execution function
            skip_completed_stages: Skip stages with existing artifacts
            **kwargs: Additional kwargs for pipeline

        Returns:
            New job_id if recovery started, None if not recoverable
        """
        recovery_info = self.check_recoverable(document_id)

        if not recovery_info["recoverable"]:
            logger.warning(
                f"Document {document_id} not recoverable: {recovery_info['reason']}"
            )
            return None

        logger.info(
            f"Recovering document {document_id}, stages: {recovery_info['available_stages']}"
        )

        job = ProcessingJob(
            job_id=str(uuid.uuid4()),
            document_id=document_id,
            status=JobStatus.PENDING,
            current_stage="recovery",
            metadata={
                "recovery": True,
                "completed_stages": recovery_info["available_stages"],
            },
        )

        self._repository.create(job)

        task = asyncio.create_task(
            self._execute_job_recoverable(
                job,
                process_func,
                recovery_info["available_stages"],
                skip_completed_stages,
                **kwargs,
            )
        )
        self._active_tasks[job.job_id] = task

        return job.job_id

    async def _execute_job_recoverable(
        self,
        job: ProcessingJob,
        process_func: Callable,
        completed_stages: List[str],
        skip_completed_stages: bool,
        **kwargs,
    ) -> None:
        """Execute job with recovery support."""
        async with self._semaphore:
            try:
                job.status = JobStatus.PROCESSING
                job.started_at = datetime.now()
                self._repository.update(job)

                logger.info(
                    f"Starting recoverable job {job.job_id} for {job.document_id}"
                )

                async def progress_callback(stage: str, percent: int):
                    job.current_stage = stage
                    job.progress_percent = percent
                    self._repository.update_progress(job.job_id, stage, percent)
                    logger.debug(f"Job {job.job_id}: {stage} ({percent}%)")

                await process_func(
                    job.document_id,
                    progress_callback=progress_callback,
                    skip_existing=skip_completed_stages,
                    skip_stages=completed_stages if skip_completed_stages else [],
                    **kwargs,
                )

                job.status = JobStatus.COMPLETED
                job.progress_percent = 100
                job.completed_at = datetime.now()
                job.current_stage = "completed"

                logger.info(f"Job {job.job_id} recovered successfully")

            except Exception as e:
                logger.error(f"Job {job.job_id} recovery failed: {e}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.now()

            finally:
                self._repository.update(job)
                if job.job_id in self._active_tasks:
                    del self._active_tasks[job.job_id]

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        active_jobs = self._repository.list_active()

        return {
            "max_concurrent": self.max_concurrent,
            "current_concurrent": len(
                [j for j in active_jobs if j.status == JobStatus.PROCESSING]
            ),
            "queued_jobs": len(
                [j for j in active_jobs if j.status == JobStatus.PENDING]
            ),
            "active_jobs": len(active_jobs),
            "active_job_ids": [j.job_id for j in active_jobs],
        }

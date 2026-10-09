"""
Pipeline observability primitives: auditor and state tracker.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
import time
from typing import Any, Dict, List, Optional

from utils.logger import get_logger, log_structured
from utils.config import settings

logger = get_logger(__name__)

PIPELINE_STAGES = [
    "ingestion",
    "chunking",
    "embedding",
    "mapping",
    "graph_building",
    "evaluation",
    "hierarchy",
]


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def estimate_size(payload: Any) -> int:
    """Estimate logical size for audit metrics."""
    if payload is None:
        return 0
    if isinstance(payload, (str, bytes, bytearray)):
        return len(payload)
    if isinstance(payload, (list, tuple, set, dict)):
        return len(payload)
    if hasattr(payload, "number_of_nodes") and hasattr(payload, "number_of_edges"):
        try:
            return int(payload.number_of_nodes()) + int(payload.number_of_edges())
        except Exception:
            return 1
    if hasattr(payload, "word_count"):
        try:
            return int(payload.word_count or 0)
        except Exception:
            return 0
    return 1


class PipelineStateTracker:
    """Tracks pipeline stage progress for real-time visualization."""

    def __init__(self):
        self._state: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def start_pipeline(self, document_id: str) -> None:
        with self._lock:
            self._state[document_id] = {
                "document_id": document_id,
                "status": "running",
                "analysis_mode": "llm",
                "active_model": settings.ollama_model,
                "overall_progress": 0,
                "current_stage": "queued",
                "started_at": _utc_now(),
                "started_perf": time.perf_counter(),
                "updated_at": _utc_now(),
                "completed_at": None,
                "total_duration_sec": 0.0,
                "error": None,
                "stages": {
                    stage: {
                        "name": stage,
                        "status": "pending",
                        "progress": 0,
                        "start_time": None,
                        "start_perf": None,
                        "end_time": None,
                        "duration_sec": 0.0,
                        "input_size": 0,
                        "output_size": 0,
                        "quality_metrics": {},
                        "errors": [],
                    }
                    for stage in PIPELINE_STAGES
                },
            }

    def set_overall_progress(
        self, document_id: str, current_stage: str, progress_percent: int
    ) -> None:
        with self._lock:
            state = self._state.get(document_id)
            if not state:
                return
            state["current_stage"] = current_stage
            state["overall_progress"] = int(max(0, min(100, progress_percent)))
            state["updated_at"] = _utc_now()

    def start_stage(
        self, document_id: str, stage: str, input_size: int = 0, progress: Optional[int] = None
    ) -> None:
        with self._lock:
            stage_state = self._get_stage(document_id, stage)
            if not stage_state:
                return
            stage_state["status"] = "running"
            stage_state["start_time"] = _utc_now()
            stage_state["start_perf"] = time.perf_counter()
            stage_state["input_size"] = input_size
            stage_state["progress"] = max(int(stage_state.get("progress", 0)), 10)
            if progress is not None:
                self._state[document_id]["overall_progress"] = int(max(0, min(100, progress)))
            self._state[document_id]["current_stage"] = stage
            self._state[document_id]["updated_at"] = _utc_now()

    def update_stage_progress(
        self,
        document_id: str,
        stage: str,
        stage_progress: int,
        overall_progress: Optional[int] = None,
    ) -> None:
        with self._lock:
            stage_state = self._get_stage(document_id, stage)
            if not stage_state:
                return
            stage_state["progress"] = int(max(0, min(100, stage_progress)))
            if stage_state.get("status") == "pending":
                stage_state["status"] = "running"
            if overall_progress is not None and document_id in self._state:
                self._state[document_id]["overall_progress"] = int(max(0, min(100, overall_progress)))
                self._state[document_id]["current_stage"] = stage
            self._state[document_id]["updated_at"] = _utc_now()

    def complete_stage(
        self,
        document_id: str,
        stage: str,
        output_size: int = 0,
        quality_metrics: Optional[Dict[str, Any]] = None,
        progress: Optional[int] = None,
    ) -> None:
        with self._lock:
            stage_state = self._get_stage(document_id, stage)
            if not stage_state:
                return
            stage_state["status"] = "completed"
            stage_state["end_time"] = _utc_now()
            start_p = stage_state.get("start_perf")
            if start_p is not None:
                stage_state["duration_sec"] = round(time.perf_counter() - start_p, 4)
            stage_state["output_size"] = output_size
            stage_state["quality_metrics"] = quality_metrics or {}
            stage_state["progress"] = 100
            if progress is not None:
                self._state[document_id]["overall_progress"] = int(max(0, min(100, progress)))
            self._state[document_id]["updated_at"] = _utc_now()

    def skip_stage(self, document_id: str, stage: str, progress: Optional[int] = None) -> None:
        with self._lock:
            stage_state = self._get_stage(document_id, stage)
            if not stage_state:
                return
            now = _utc_now()
            stage_state["status"] = "skipped"
            stage_state["start_time"] = stage_state["start_time"] or now
            stage_state["end_time"] = now
            start_p = stage_state.get("start_perf")
            if start_p is not None:
                stage_state["duration_sec"] = round(time.perf_counter() - start_p, 4)
            stage_state["progress"] = 100
            if progress is not None:
                self._state[document_id]["overall_progress"] = int(max(0, min(100, progress)))
            self._state[document_id]["updated_at"] = now

    def fail_stage(
        self,
        document_id: str,
        stage: str,
        error: str,
        output_size: int = 0,
        quality_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            stage_state = self._get_stage(document_id, stage)
            if not stage_state:
                return
            stage_state["status"] = "failed"
            stage_state["end_time"] = _utc_now()
            start_p = stage_state.get("start_perf")
            if start_p is not None:
                stage_state["duration_sec"] = round(time.perf_counter() - start_p, 4)
            stage_state["output_size"] = output_size
            stage_state["quality_metrics"] = quality_metrics or {}
            stage_state["errors"].append(error)
            self._state[document_id]["status"] = "failed"
            self._state[document_id]["error"] = error
            self._state[document_id]["updated_at"] = _utc_now()

    def complete_pipeline(
        self,
        document_id: str,
        is_degraded: bool = False,
        analysis_mode: str = "llm",
        active_model: Optional[str] = None,
    ) -> None:
        with self._lock:
            state = self._state.get(document_id)
            if not state:
                return
            state["status"] = "degraded" if is_degraded else "completed"
            state["analysis_mode"] = analysis_mode
            if active_model:
                state["active_model"] = active_model
            state["overall_progress"] = 100
            state["current_stage"] = "completed"
            state["completed_at"] = _utc_now()
            start_p = state.get("started_perf")
            if start_p is not None:
                state["total_duration_sec"] = round(time.perf_counter() - start_p, 4)
            state["updated_at"] = _utc_now()

    def fail_pipeline(self, document_id: str, error: str) -> None:
        with self._lock:
            state = self._state.get(document_id)
            if not state:
                return
            state["status"] = "failed"
            state["error"] = error
            state["completed_at"] = _utc_now()
            start_p = state.get("started_perf")
            if start_p is not None:
                state["total_duration_sec"] = round(time.perf_counter() - start_p, 4)
            state["updated_at"] = _utc_now()

    def get_status(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if not document_id:
                return {"pipelines": [self._format_pipeline_status(v) for v in self._state.values()]}
            raw = self._state.get(document_id)
            if not raw:
                return {}
            return self._format_pipeline_status(raw)

    def _format_pipeline_status(self, state: Dict[str, Any]) -> Dict[str, Any]:
        out = deepcopy(state)
        stages = out.get("stages", {})
        ing = stages.get("ingestion", {})
        ch = stages.get("chunking", {})
        emb = stages.get("embedding", {})
        map_stage = stages.get("mapping", {})
        gb = stages.get("graph_building", {})

        chunk_count = int(ch.get("output_size", 0) or 0)
        emb_duration = float(emb.get("duration_sec", 0.0) or 0.0)
        throughput = round(chunk_count / emb_duration, 2) if emb_duration > 0 and chunk_count > 0 else 0.0

        is_fallback = (
            out.get("analysis_mode") == "deterministic_fallback"
            or out.get("status") == "degraded"
        )
        llm_stage_status = (
            "fallback"
            if is_fallback
            else (
                "completed"
                if map_stage.get("status") == "completed"
                else map_stage.get("status", "pending")
            )
        )

        formatted_stages = {
            "ingestion": {
                "status": ing.get("status", "pending"),
                "duration_sec": float(ing.get("duration_sec", 0.0) or 0.0),
            },
            "chunking": {
                "status": ch.get("status", "pending"),
                "duration_sec": float(ch.get("duration_sec", 0.0) or 0.0),
                "chunk_count": chunk_count,
            },
            "embedding": {
                "status": emb.get("status", "pending"),
                "duration_sec": emb_duration,
                "throughput_chunks_sec": throughput,
                "device": settings.embedding_device_actual or "cpu",
            },
            "llm_analysis": {
                "status": llm_stage_status,
                "duration_sec": float(map_stage.get("duration_sec", 0.0) or 0.0),
                "model": out.get("active_model", settings.ollama_model),
            },
            "graph_building": {
                "status": gb.get("status", "pending"),
                "duration_sec": float(gb.get("duration_sec", 0.0) or 0.0),
                "node_count": int(gb.get("quality_metrics", {}).get("node_count", 0)),
                "edge_count": int(gb.get("quality_metrics", {}).get("edge_count", 0)),
            },
        }

        # Keep original stages for backward compatibility with frontend
        for k, v in stages.items():
            if k not in formatted_stages:
                formatted_stages[k] = v

        out["stages"] = formatted_stages
        out["document_id"] = out.get("document_id", "")
        out["status"] = out.get("status", "running")
        out["analysis_mode"] = out.get("analysis_mode", "llm")
        out["active_model"] = out.get("active_model", settings.ollama_model)
        out["total_duration_sec"] = float(out.get("total_duration_sec", 0.0) or 0.0)
        return out

    def _get_stage(self, document_id: str, stage: str) -> Optional[Dict[str, Any]]:
        state = self._state.get(document_id)
        if not state:
            return None
        return state["stages"].get(stage)


class PipelineAuditor:
    """Stage-level auditor for pipeline execution quality and reliability."""

    def __init__(self, document_id: str, tracker: PipelineStateTracker):
        self.document_id = document_id
        self.tracker = tracker
        self.stages: Dict[str, Dict[str, Any]] = {
            stage: {
                "stage": stage,
                "start_time": None,
                "start_perf": None,
                "end_time": None,
                "duration_sec": None,
                "input_size": 0,
                "output_size": 0,
                "errors": [],
                "quality_metrics": {},
                "status": "pending",
                "validation": {"passed": None, "issues": []},
            }
            for stage in PIPELINE_STAGES
        }

    def start_stage(self, stage: str, input_size: int = 0, progress: Optional[int] = None) -> None:
        record = self.stages[stage]
        record["start_time"] = _utc_now()
        record["start_perf"] = time.perf_counter()
        record["status"] = "running"
        record["input_size"] = int(max(0, input_size))
        self.tracker.start_stage(self.document_id, stage, input_size=input_size, progress=progress)
        log_structured(
            logger,
            level=20,
            stage=stage,
            event="stage_start",
            message=f"Started {stage} stage",
            document_id=self.document_id,
            metadata={"input_size": input_size},
        )

    def update_stage_progress(
        self,
        stage: str,
        stage_progress: int,
        overall_progress: Optional[int] = None,
    ) -> None:
        record = self.stages.get(stage)
        if record:
            record["status"] = "running"
        self.tracker.update_stage_progress(
            self.document_id,
            stage,
            stage_progress=stage_progress,
            overall_progress=overall_progress,
        )

    def complete_stage(
        self,
        stage: str,
        output_size: int = 0,
        quality_metrics: Optional[Dict[str, Any]] = None,
        progress: Optional[int] = None,
    ) -> None:
        quality_metrics = quality_metrics or {}
        validation = self._validate(stage=stage, output_size=output_size, metrics=quality_metrics)
        record = self.stages[stage]
        record["end_time"] = _utc_now()
        start_p = record.get("start_perf")
        if start_p is not None:
            record["duration_sec"] = round(time.perf_counter() - start_p, 4)
        record["status"] = "completed"
        record["output_size"] = int(max(0, output_size))
        record["quality_metrics"] = quality_metrics
        record["validation"] = validation
        self.tracker.complete_stage(
            self.document_id,
            stage,
            output_size=output_size,
            quality_metrics=quality_metrics,
            progress=progress,
        )
        log_structured(
            logger,
            level=20,
            stage=stage,
            event="stage_complete",
            message=f"Completed {stage} stage",
            document_id=self.document_id,
            metadata={
                "output_size": output_size,
                "quality_metrics": quality_metrics,
                "validation": validation,
                "duration_sec": record.get("duration_sec"),
            },
        )

    def fail_stage(
        self,
        stage: str,
        error: str,
        output_size: int = 0,
        quality_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        quality_metrics = quality_metrics or {}
        record = self.stages[stage]
        record["end_time"] = _utc_now()
        start_p = record.get("start_perf")
        if start_p is not None:
            record["duration_sec"] = round(time.perf_counter() - start_p, 4)
        record["status"] = "failed"
        record["output_size"] = int(max(0, output_size))
        record["quality_metrics"] = quality_metrics
        record["errors"].append(error)
        self.tracker.fail_stage(
            self.document_id,
            stage,
            error=error,
            output_size=output_size,
            quality_metrics=quality_metrics,
        )
        log_structured(
            logger,
            level=40,
            stage=stage,
            event="stage_error",
            message=f"Failed {stage} stage",
            document_id=self.document_id,
            metadata={
                "error": error,
                "quality_metrics": quality_metrics,
                "duration_sec": record.get("duration_sec"),
            },
        )

    def skip_stage(self, stage: str, progress: Optional[int] = None) -> None:
        record = self.stages[stage]
        now = _utc_now()
        record["start_time"] = record["start_time"] or now
        record["end_time"] = now
        start_p = record.get("start_perf")
        if start_p is not None:
            record["duration_sec"] = round(time.perf_counter() - start_p, 4)
        record["status"] = "skipped"
        record["validation"] = {"passed": True, "issues": []}
        self.tracker.skip_stage(self.document_id, stage=stage, progress=progress)
        log_structured(
            logger,
            level=20,
            stage=stage,
            event="stage_skipped",
            message=f"Skipped {stage} stage",
            document_id=self.document_id,
            metadata={"duration_sec": record.get("duration_sec")},
        )

    def report(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "generated_at": _utc_now(),
            "stages": [deepcopy(self.stages[stage]) for stage in PIPELINE_STAGES],
        }

    def _validate(
        self, stage: str, output_size: int, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        issues: List[str] = []

        if stage in {"chunking", "embedding", "mapping", "graph_building"} and output_size <= 0:
            issues.append("Output size is zero")

        if stage == "evaluation":
            overall_score = metrics.get("overall_score")
            if overall_score is not None and overall_score < 0.3:
                issues.append("Low evaluation confidence")

        if stage == "mapping":
            if metrics.get("concept_count", 0) == 0:
                issues.append("No concepts extracted")

        if stage == "graph_building":
            if metrics.get("node_count", 0) == 0:
                issues.append("Graph has no nodes")

        return {"passed": len(issues) == 0, "issues": issues}


_tracker: Optional[PipelineStateTracker] = None


def get_pipeline_state_tracker() -> PipelineStateTracker:
    global _tracker
    if _tracker is None:
        _tracker = PipelineStateTracker()
    return _tracker

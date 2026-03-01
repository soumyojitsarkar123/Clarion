"""
Pipeline observability primitives: auditor and state tracker.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

from utils.logger import get_logger, log_structured

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
                "overall_progress": 0,
                "current_stage": "queued",
                "started_at": _utc_now(),
                "updated_at": _utc_now(),
                "completed_at": None,
                "error": None,
                "stages": {
                    stage: {
                        "name": stage,
                        "status": "pending",
                        "progress": 0,
                        "start_time": None,
                        "end_time": None,
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
            stage_state["input_size"] = input_size
            stage_state["progress"] = max(int(stage_state.get("progress", 0)), 10)
            if progress is not None:
                self._state[document_id]["overall_progress"] = int(max(0, min(100, progress)))
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
            stage_state["output_size"] = output_size
            stage_state["quality_metrics"] = quality_metrics or {}
            stage_state["errors"].append(error)
            self._state[document_id]["status"] = "failed"
            self._state[document_id]["error"] = error
            self._state[document_id]["updated_at"] = _utc_now()

    def complete_pipeline(self, document_id: str) -> None:
        with self._lock:
            state = self._state.get(document_id)
            if not state:
                return
            state["status"] = "completed"
            state["overall_progress"] = 100
            state["current_stage"] = "completed"
            state["completed_at"] = _utc_now()
            state["updated_at"] = _utc_now()

    def fail_pipeline(self, document_id: str, error: str) -> None:
        with self._lock:
            state = self._state.get(document_id)
            if not state:
                return
            state["status"] = "failed"
            state["error"] = error
            state["completed_at"] = _utc_now()
            state["updated_at"] = _utc_now()

    def get_status(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if document_id:
                return deepcopy(self._state.get(document_id, {}))
            return {"pipelines": [deepcopy(v) for v in self._state.values()]}

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
                "end_time": None,
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
            metadata={"error": error, "quality_metrics": quality_metrics},
        )

    def skip_stage(self, stage: str, progress: Optional[int] = None) -> None:
        record = self.stages[stage]
        now = _utc_now()
        record["start_time"] = record["start_time"] or now
        record["end_time"] = now
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
            metadata={},
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

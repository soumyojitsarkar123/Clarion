"""
Logs Router - API endpoints for retrieving application logs.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from utils.config import settings
from utils.log_stream import get_log_stream_broker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/logs", tags=["logs"])


class LogResponse(BaseModel):
    """Response model for log content."""

    lines: list[str]
    total_lines: int
    displayed_lines: int


@router.get("", response_model=LogResponse)
async def get_logs(
    lines: Optional[int] = Query(
        100, ge=1, le=1000, description="Number of lines to return"
    ),
    offset: Optional[int] = Query(0, ge=0, description="Line offset to start from"),
):
    """
    Get application logs.

    - **lines**: Number of lines to return (default: 100, max: 1000)
    - **offset**: Line offset to start from (default: 0)
    """
    log_file = settings.logs_dir / "clarion.log"

    if not log_file.exists():
        return LogResponse(lines=[], total_lines=0, displayed_lines=0)

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)
        start_idx = max(0, total_lines - offset - lines)
        end_idx = total_lines - offset

        displayed_lines = all_lines[start_idx:end_idx]
        clean_lines = [line.strip() for line in displayed_lines if line.strip()]

        return LogResponse(
            lines=clean_lines, total_lines=total_lines, displayed_lines=len(clean_lines)
        )
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return LogResponse(
            lines=[f"Error reading logs: {str(e)}"], total_lines=0, displayed_lines=0
        )


@router.get("/recent", response_model=LogResponse)
async def get_recent_logs(
    lines: Optional[int] = Query(
        50, ge=1, le=200, description="Number of lines to return"
    ),
):
    """
    Get the most recent application logs.
    """
    return await get_logs(lines=lines, offset=0)


@router.websocket("/stream")
async def stream_logs(websocket: WebSocket):
    """
    Stream structured logs over websocket for real-time UI observability.

    Query params:
    - document_id: Optional document ID filter
    - after_sequence: Optional sequence cursor
    """
    await websocket.accept()
    broker = get_log_stream_broker()
    document_id = websocket.query_params.get("document_id")

    after_sequence_raw = websocket.query_params.get("after_sequence", "0")
    try:
        after_sequence = int(after_sequence_raw)
    except ValueError:
        after_sequence = 0

    try:
        # Send most recent logs immediately so client has context.
        initial_events = broker.snapshot(
            after_sequence=after_sequence, document_id=document_id, limit=100
        )
        for event in initial_events:
            after_sequence = max(after_sequence, int(event.get("sequence", after_sequence)))
            await websocket.send_json(event)

        while True:
            events = broker.snapshot(
                after_sequence=after_sequence, document_id=document_id, limit=250
            )
            for event in events:
                after_sequence = max(after_sequence, int(event.get("sequence", after_sequence)))
                await websocket.send_json(event)
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
    except Exception as error:
        logger.error("Websocket log stream error: %s", error)

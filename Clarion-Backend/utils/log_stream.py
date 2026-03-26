"""
In-memory structured log stream broker for websocket clients.
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any, Deque, Dict, List, Optional


class LogStreamBroker:
    """Thread-safe broker storing recent structured logs for streaming."""

    def __init__(self, max_events: int = 5000):
        self._events: Deque[Dict[str, Any]] = deque(maxlen=max_events)
        self._lock = Lock()
        self._sequence = 0

    def publish(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Publish an event and assign a sequence number."""
        with self._lock:
            self._sequence += 1
            enriched = dict(event)
            enriched["sequence"] = self._sequence
            self._events.append(enriched)
            return enriched

    def snapshot(
        self,
        after_sequence: int = 0,
        document_id: Optional[str] = None,
        limit: int = 250,
    ) -> List[Dict[str, Any]]:
        """Return events newer than a sequence number."""
        with self._lock:
            items = [event for event in self._events if event["sequence"] > after_sequence]
            if document_id:
                items = [event for event in items if event.get("document_id") == document_id]
            if limit > 0:
                items = items[-limit:]
            return [dict(item) for item in items]

    def latest_sequence(self) -> int:
        with self._lock:
            return self._sequence


_broker: Optional[LogStreamBroker] = None


def get_log_stream_broker() -> LogStreamBroker:
    """Get singleton log broker."""
    global _broker
    if _broker is None:
        _broker = LogStreamBroker()
    return _broker

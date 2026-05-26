"""Download progress events and in-process pub/sub bus.

Event types are flat constants so the contract with the frontend is
visible in one place. Each download has its own subscriber list and
its own bounded history buffer; a client that connects late can replay
what it missed (up to the buffer cap).

The bus is in-process. Multi-instance fan-out would need Redis pub/sub
later, but that is out of scope for this module.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator

from src.domain.download import DownloadEvent

# Event vocabulary. Keep this list authoritative; do not invent ad-hoc strings.
EVENT_QUEUED = "download.queued"
EVENT_AUTH_STARTED = "download.auth.started"
EVENT_AUTH_OK = "download.auth.ok"
EVENT_AUTH_FAILED = "download.auth.failed"
EVENT_METADATA_STARTED = "download.metadata.started"
EVENT_METADATA_READY = "download.metadata.ready"
EVENT_METADATA_FAILED = "download.metadata.failed"
EVENT_FILES_STARTED = "download.files.started"
EVENT_FILES_COMPLETE = "download.files.complete"
EVENT_INDEXED = "download.indexed"
EVENT_COMPLETE = "download.complete"
EVENT_FAILED = "download.failed"


def make_event(download_id: str, event_type: str, **data) -> DownloadEvent:
    return DownloadEvent(
        download_id=download_id,
        event_type=event_type,
        timestamp=datetime.now(timezone.utc),
        data=dict(data),
    )


class DownloadEventBus:
    """In-process pub/sub keyed by download_id."""

    def __init__(self, *, max_history: int = 200) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._history: dict[str, list[DownloadEvent]] = {}
        self._lock = asyncio.Lock()
        self._max_history = max_history

    async def publish(self, event: DownloadEvent) -> None:
        async with self._lock:
            history = self._history.setdefault(event.download_id, [])
            history.append(event)
            if len(history) > self._max_history:
                del history[: len(history) - self._max_history]
            queues = list(self._subscribers.get(event.download_id, set()))
        for queue in queues:
            await queue.put(event)

    async def subscribe(
        self, download_id: str, *, replay: bool = True
    ) -> AsyncIterator[DownloadEvent]:
        queue: asyncio.Queue[DownloadEvent] = asyncio.Queue()
        async with self._lock:
            self._subscribers.setdefault(download_id, set()).add(queue)
            history = list(self._history.get(download_id, []))
        try:
            if replay:
                for event in history:
                    yield event
            while True:
                event = await queue.get()
                yield event
                if event.event_type in (EVENT_COMPLETE, EVENT_FAILED):
                    return
        finally:
            async with self._lock:
                self._subscribers.get(download_id, set()).discard(queue)

    def history(self, download_id: str) -> list[DownloadEvent]:
        return list(self._history.get(download_id, []))


_GLOBAL_BUS: DownloadEventBus | None = None


def get_bus() -> DownloadEventBus:
    """Return the process-wide bus; create on first call."""
    global _GLOBAL_BUS
    if _GLOBAL_BUS is None:
        _GLOBAL_BUS = DownloadEventBus()
    return _GLOBAL_BUS


def reset_bus_for_tests() -> None:
    """Test helper. Do not call in production code."""
    global _GLOBAL_BUS
    _GLOBAL_BUS = None

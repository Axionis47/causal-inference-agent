"""SSE transport for download progress events.

The frontend opens GET /api/v1/download/downloads/{id}/events and gets a
text/event-stream where each chunk is one DownloadEvent serialized as
JSON. The stream closes after the terminal event (complete or failed).
"""
from __future__ import annotations

from typing import AsyncIterator

from sse_starlette.sse import EventSourceResponse

from src.download import events


async def _event_iterator(
    bus: events.DownloadEventBus, download_id: str
) -> AsyncIterator[dict]:
    async for event in bus.subscribe(download_id, replay=True):
        yield {
            "event": event.event_type,
            "data": event.model_dump_json(),
        }


def stream_download_events(
    bus: events.DownloadEventBus, download_id: str
) -> EventSourceResponse:
    """Return a FastAPI SSE response that streams events for one download."""
    return EventSourceResponse(_event_iterator(bus, download_id))

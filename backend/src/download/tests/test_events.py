"""DownloadEventBus: publish/subscribe + history replay."""
from __future__ import annotations

import asyncio

import pytest

from src.download.events import (
    EVENT_COMPLETE,
    EVENT_FILES_STARTED,
    EVENT_METADATA_READY,
    DownloadEventBus,
    make_event,
)


def test_make_event_sets_type_and_data():
    event = make_event("dl-1", EVENT_METADATA_READY, owner="uciml", slug="iris")
    assert event.event_type == EVENT_METADATA_READY
    assert event.download_id == "dl-1"
    assert event.data == {"owner": "uciml", "slug": "iris"}
    assert event.timestamp is not None


@pytest.mark.asyncio
async def test_publish_to_subscriber_in_order():
    bus = DownloadEventBus()
    received: list[str] = []

    async def reader():
        async for event in bus.subscribe("dl-1", replay=False):
            received.append(event.event_type)

    task = asyncio.create_task(reader())
    await asyncio.sleep(0)  # let subscriber register

    await bus.publish(make_event("dl-1", EVENT_FILES_STARTED, total=2))
    await bus.publish(make_event("dl-1", EVENT_METADATA_READY))
    await bus.publish(make_event("dl-1", EVENT_COMPLETE))

    await asyncio.wait_for(task, timeout=1.0)
    assert received == [EVENT_FILES_STARTED, EVENT_METADATA_READY, EVENT_COMPLETE]


@pytest.mark.asyncio
async def test_late_subscriber_replays_history():
    bus = DownloadEventBus()
    await bus.publish(make_event("dl-1", EVENT_FILES_STARTED, total=1))
    await bus.publish(make_event("dl-1", EVENT_METADATA_READY))

    received: list[str] = []

    async def reader():
        async for event in bus.subscribe("dl-1", replay=True):
            received.append(event.event_type)

    task = asyncio.create_task(reader())
    await asyncio.sleep(0)
    await bus.publish(make_event("dl-1", EVENT_COMPLETE))

    await asyncio.wait_for(task, timeout=1.0)
    assert received == [EVENT_FILES_STARTED, EVENT_METADATA_READY, EVENT_COMPLETE]


@pytest.mark.asyncio
async def test_subscribe_terminates_after_complete():
    bus = DownloadEventBus()

    async def reader():
        events = []
        async for event in bus.subscribe("dl-1", replay=False):
            events.append(event)
        return events

    task = asyncio.create_task(reader())
    await asyncio.sleep(0)
    await bus.publish(make_event("dl-1", EVENT_COMPLETE))
    events = await asyncio.wait_for(task, timeout=1.0)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_multiple_subscribers_each_see_events():
    bus = DownloadEventBus()
    a: list[str] = []
    b: list[str] = []

    async def reader(out: list[str]):
        async for event in bus.subscribe("dl-1", replay=False):
            out.append(event.event_type)

    t1 = asyncio.create_task(reader(a))
    t2 = asyncio.create_task(reader(b))
    await asyncio.sleep(0)

    await bus.publish(make_event("dl-1", EVENT_METADATA_READY))
    await bus.publish(make_event("dl-1", EVENT_COMPLETE))

    await asyncio.gather(t1, t2)
    assert a == b == [EVENT_METADATA_READY, EVENT_COMPLETE]


@pytest.mark.asyncio
async def test_subscribers_are_isolated_per_download_id():
    bus = DownloadEventBus()
    a: list[str] = []

    async def reader():
        async for event in bus.subscribe("dl-1", replay=False):
            a.append(event.event_type)

    task = asyncio.create_task(reader())
    await asyncio.sleep(0)
    # event for a different download must not reach this subscriber
    await bus.publish(make_event("dl-other", EVENT_METADATA_READY))
    await bus.publish(make_event("dl-1", EVENT_COMPLETE))
    await asyncio.wait_for(task, timeout=1.0)
    assert a == [EVENT_COMPLETE]


@pytest.mark.asyncio
async def test_history_is_bounded():
    bus = DownloadEventBus(max_history=3)
    for i in range(5):
        await bus.publish(make_event("dl-1", EVENT_METADATA_READY, i=i))
    history = bus.history("dl-1")
    assert len(history) == 3
    assert [e.data["i"] for e in history] == [2, 3, 4]

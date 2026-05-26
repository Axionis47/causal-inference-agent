"""Pull orchestrator.

Composes the smaller modules into the full download flow:
    auth -> metadata -> files -> index -> persist

Public entry points:
    start_download(req, ...) -> download_id   # returns immediately, runs in background
    execute_pull(record, ...)                  # the coroutine itself; useful for tests
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.domain.download import (
    DownloadRecord,
    DownloadRequest,
    DownloadStatus,
)
from src.download import (
    auth,
    client as client_mod,
    download_store,
    events,
    profile_store,
    storage,
    url as url_mod,
)
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

_BACKGROUND_TASKS: set[asyncio.Task] = set()


async def start_download(
    req: DownloadRequest,
    *,
    root: Path,
    encryption_key: str,
    bus: events.DownloadEventBus,
) -> str:
    """Create a record, schedule the pull, return the download_id immediately."""
    download_id = _generate_id()
    parsed = url_mod.parse_kaggle_url(req.kaggle_url)
    record = DownloadRecord(
        download_id=download_id,
        profile_id=req.profile_id,
        kaggle_url=req.kaggle_url,
        dataset_ref=parsed.dataset_ref,
        status=DownloadStatus.QUEUED,
        created_at=datetime.now(timezone.utc),
    )
    download_store.save(record, root=root)
    await bus.publish(events.make_event(download_id, events.EVENT_QUEUED, dataset_ref=parsed.dataset_ref))

    task = asyncio.create_task(
        execute_pull(record, root=root, encryption_key=encryption_key, bus=bus)
    )
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return download_id


async def execute_pull(
    record: DownloadRecord,
    *,
    root: Path,
    encryption_key: str,
    bus: events.DownloadEventBus,
) -> DownloadRecord:
    """Run the full pull synchronously and return the final record."""
    try:
        creds = profile_store.get_kaggle_key(
            record.profile_id, root=root, encryption_key=encryption_key
        )
        if creds is None:
            return await _fail(
                record, root=root, bus=bus,
                phase=events.EVENT_AUTH_FAILED,
                error=f"No Kaggle credentials set for profile '{record.profile_id}'.",
            )
        username, key = creds

        parsed = url_mod.parse_kaggle_url(record.kaggle_url)

        record = await _set_status(record, DownloadStatus.AUTHENTICATING, root=root)
        await bus.publish(events.make_event(record.download_id, events.EVENT_AUTH_STARTED))

        with auth.kaggle_env_scope(username, key):
            try:
                client = client_mod.KaggleClient()
            except Exception as e:
                return await _fail(
                    record, root=root, bus=bus,
                    phase=events.EVENT_AUTH_FAILED,
                    error=f"Kaggle authentication failed: {e}",
                )
            await bus.publish(events.make_event(record.download_id, events.EVENT_AUTH_OK))

            record = await _set_status(record, DownloadStatus.FETCHING_METADATA, root=root)
            await bus.publish(events.make_event(record.download_id, events.EVENT_METADATA_STARTED))
            try:
                meta = await client.fetch_metadata(
                    parsed.owner, parsed.slug, url=record.kaggle_url
                )
            except Exception as e:
                return await _fail(
                    record, root=root, bus=bus,
                    phase=events.EVENT_METADATA_FAILED,
                    error=f"Failed to fetch Kaggle metadata: {e}",
                )
            record = record.model_copy(update={"metadata": meta})
            download_store.save(record, root=root)
            await bus.publish(events.make_event(
                record.download_id, events.EVENT_METADATA_READY,
                owner=meta.owner, slug=meta.slug, title=meta.title,
                tags=meta.tags, total_bytes=meta.total_bytes,
            ))

            effective_version = parsed.version or meta.version
            dest = storage.dataset_dir(
                root, record.profile_id, parsed.owner, parsed.slug, effective_version
            )

            record = await _set_status(record, DownloadStatus.DOWNLOADING_FILES, root=root)
            await bus.publish(events.make_event(
                record.download_id, events.EVENT_FILES_STARTED,
                dest=str(dest), version=effective_version,
            ))
            try:
                await client.download_files(
                    parsed.owner, parsed.slug, dest, version=effective_version
                )
            except Exception as e:
                return await _fail(
                    record, root=root, bus=bus,
                    phase=events.EVENT_FAILED,
                    error=_translate_download_error(e, record.dataset_ref),
                )
            await bus.publish(events.make_event(record.download_id, events.EVENT_FILES_COMPLETE))

        record = await _set_status(record, DownloadStatus.UNZIPPING, root=root)
        files = storage.index_directory(dest)
        await bus.publish(events.make_event(
            record.download_id, events.EVENT_INDEXED, file_count=len(files),
        ))

        record = record.model_copy(update={
            "storage_path": str(dest),
            "files": files,
            "status": DownloadStatus.COMPLETE,
            "completed_at": datetime.now(timezone.utc),
        })
        download_store.save(record, root=root)
        storage.write_manifest(dest, record)
        profile_store.mark_used(record.profile_id, root=root)

        await bus.publish(events.make_event(
            record.download_id, events.EVENT_COMPLETE,
            file_count=len(files),
            storage_path=str(dest),
        ))
        return record

    except Exception as e:
        logger.exception("download_pull_unexpected_error", download_id=record.download_id)
        return await _fail(
            record, root=root, bus=bus,
            phase=events.EVENT_FAILED,
            error=f"Unexpected error: {e}",
        )


async def _set_status(
    record: DownloadRecord, status: DownloadStatus, *, root: Path
) -> DownloadRecord:
    updated = record.model_copy(update={"status": status})
    download_store.save(updated, root=root)
    return updated


async def _fail(
    record: DownloadRecord,
    *,
    root: Path,
    bus: events.DownloadEventBus,
    phase: str,
    error: str,
) -> DownloadRecord:
    updated = record.model_copy(update={
        "status": DownloadStatus.FAILED,
        "error": error,
        "completed_at": datetime.now(timezone.utc),
    })
    download_store.save(updated, root=root)
    await bus.publish(events.make_event(record.download_id, phase, error=error))
    if phase != events.EVENT_FAILED:
        await bus.publish(events.make_event(record.download_id, events.EVENT_FAILED, error=error))
    return updated


def _translate_download_error(exc: Exception, dataset_ref: str) -> str:
    text = str(exc)
    if "403" in text or "Forbidden" in text:
        return f"Dataset '{dataset_ref}' is not accessible (403). It may be private or restricted."
    if "404" in text or "Not Found" in text:
        return f"Dataset '{dataset_ref}' not found (404). Verify the URL."
    if "401" in text or "Unauthorized" in text:
        return "Kaggle credentials rejected during download (401)."
    return f"Failed to download from Kaggle: {text[:200]}"


def _generate_id() -> str:
    return f"dl-{uuid.uuid4().hex[:12]}"

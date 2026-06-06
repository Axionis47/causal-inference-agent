"""DownloadRecord persistence.

A record is the index entry for one download. Listing, retrieval, and
deletion happen here; the on-disk dataset directory (with its manifest)
is owned by storage.py. The dataset manifest is a snapshot for
portability; this store is the source of truth.

Layout:
    {root}/downloads/{download_id}.json
"""
from __future__ import annotations

import os
from pathlib import Path

from src.domain.download import DownloadRecord


def _downloads_dir(root: Path) -> Path:
    return Path(root) / "downloads"


def _record_path(root: Path, download_id: str) -> Path:
    return _downloads_dir(root) / f"{download_id}.json"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save(record: DownloadRecord, *, root: Path) -> None:
    _atomic_write(
        _record_path(root, record.download_id),
        record.model_dump_json(indent=2),
    )


def load(download_id: str, *, root: Path) -> DownloadRecord | None:
    path = _record_path(root, download_id)
    if not path.exists():
        return None
    return DownloadRecord.model_validate_json(path.read_text(encoding="utf-8"))


def list_records(*, root: Path, profile_id: str | None = None) -> list[DownloadRecord]:
    """Return all records, newest first. Optionally filter by profile_id."""
    dir_path = _downloads_dir(root)
    if not dir_path.exists():
        return []

    records: list[DownloadRecord] = []
    for path in dir_path.glob("*.json"):
        try:
            record = DownloadRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            # Skip malformed records rather than crashing list endpoints.
            continue
        if profile_id and record.profile_id != profile_id:
            continue
        records.append(record)

    records.sort(key=lambda r: r.created_at, reverse=True)
    return records


def delete(download_id: str, *, root: Path) -> None:
    path = _record_path(root, download_id)
    if path.exists():
        path.unlink()

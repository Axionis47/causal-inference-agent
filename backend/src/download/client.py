"""Thin async wrapper over the Kaggle Python client.

This is the ONLY module in the download package that imports kaggle.*.
All other modules call into here. Each public method runs the blocking
Kaggle client in a thread so the caller can await it.

The caller is responsible for scoping credentials via
download.auth.kaggle_env_scope() before instantiating the client.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from src.domain.download import KaggleMetadata


class KaggleClient:
    """Authenticated wrapper. Construct inside a kaggle_env_scope."""

    def __init__(self) -> None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        self._api = KaggleApi()
        self._api.authenticate()

    async def fetch_metadata(
        self, owner: str, slug: str, *, url: str
    ) -> KaggleMetadata:
        return await asyncio.to_thread(self._fetch_metadata_sync, owner, slug, url)

    def _fetch_metadata_sync(self, owner: str, slug: str, url: str) -> KaggleMetadata:
        dataset_ref = f"{owner}/{slug}"
        view = self._api.dataset_view(dataset_ref)
        files = self._list_files_sync(owner, slug)

        return KaggleMetadata(
            owner=owner,
            slug=slug,
            title=_attr(view, "title"),
            subtitle=_attr(view, "subtitle"),
            description=_attr(view, "description") or _attr(view, "overview"),
            tags=_tags(view),
            license_name=_license(view),
            version=_version(view),
            last_updated=_last_updated(view),
            total_bytes=_sum_bytes(files),
            url=url,
        )

    async def list_files(self, owner: str, slug: str) -> list[dict]:
        return await asyncio.to_thread(self._list_files_sync, owner, slug)

    def _list_files_sync(self, owner: str, slug: str) -> list[dict]:
        dataset_ref = f"{owner}/{slug}"
        resp = self._api.dataset_list_files(dataset_ref)
        items = getattr(resp, "datasetFiles", None) or getattr(resp, "files", []) or []
        return [
            {
                "name": getattr(f, "name", None) or getattr(f, "ref", ""),
                "size_bytes": int(getattr(f, "totalBytes", 0) or getattr(f, "size", 0)),
            }
            for f in items
        ]

    async def download_files(
        self,
        owner: str,
        slug: str,
        dest_dir: Path,
        *,
        version: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._download_files_sync, owner, slug, dest_dir, version
        )

    def _download_files_sync(
        self, owner: str, slug: str, dest_dir: Path, version: str | None
    ) -> None:
        dataset_ref = f"{owner}/{slug}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict = {"path": str(dest_dir), "unzip": True}
        if version:
            kwargs["version"] = version
        self._api.dataset_download_files(dataset_ref, **kwargs)


def _attr(view: object, name: str) -> str | None:
    value = getattr(view, name, None)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _tags(view: object) -> list[str]:
    raw = getattr(view, "tags", None) or []
    out: list[str] = []
    for tag in raw:
        if isinstance(tag, str):
            out.append(tag)
        else:
            name = getattr(tag, "name", None) or getattr(tag, "ref", None)
            if name:
                out.append(str(name))
    return out


def _license(view: object) -> str | None:
    license_obj = getattr(view, "licenseName", None) or getattr(view, "license", None)
    if license_obj is None:
        return None
    if isinstance(license_obj, str):
        return license_obj
    return getattr(license_obj, "name", None) or getattr(license_obj, "nameNullable", None)


def _version(view: object) -> str | None:
    raw = (
        getattr(view, "currentVersionNumber", None)
        or getattr(view, "versionNumber", None)
        or getattr(view, "datasetVersionNumber", None)
    )
    return str(raw) if raw is not None else None


def _last_updated(view: object) -> datetime | None:
    raw = (
        getattr(view, "lastUpdated", None)
        or getattr(view, "updatedAt", None)
        or getattr(view, "lastUpdatedNullable", None)
    )
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    try:
        text = str(raw).replace("Z", "+00:00")
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _sum_bytes(files: list[dict]) -> int | None:
    if not files:
        return None
    total = sum(int(f.get("size_bytes") or 0) for f in files)
    return total or None

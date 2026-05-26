"""Thin async wrapper over the Kaggle Python client.

This is the ONLY module in the download package that imports kaggle.*.
All other modules call into here. Each public method runs the blocking
Kaggle client in a thread so the caller can await it.

The caller is responsible for scoping credentials via
download.auth.kaggle_env_scope() before instantiating the client.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.domain.download import KaggleMetadata

METADATA_FILE_NAME = "dataset-metadata.json"


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

        with tempfile.TemporaryDirectory() as tmpdir:
            self._api.dataset_metadata(dataset_ref, path=tmpdir)
            raw_path = Path(tmpdir) / METADATA_FILE_NAME
            raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}

        files = self._list_files_sync(owner, slug)

        return KaggleMetadata(
            owner=owner,
            slug=slug,
            title=_pick(raw, "title", "titleNullable"),
            subtitle=_pick(raw, "subtitle", "subtitleNullable"),
            description=_pick(raw, "description", "descriptionNullable"),
            tags=list(raw.get("keywords") or raw.get("tags") or []),
            license_name=_first_license(raw.get("licenses")),
            version=_version_from_raw(raw),
            last_updated=_parse_datetime(raw.get("lastUpdated") or raw.get("lastUpdatedNullable")),
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


def _pick(raw: dict, *keys: str) -> str | None:
    for k in keys:
        v = raw.get(k)
        if v:
            text = str(v).strip()
            if text:
                return text
    return None


def _first_license(licenses) -> str | None:
    if not licenses:
        return None
    first = licenses[0]
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        return first.get("name") or first.get("nameNullable")
    return getattr(first, "name", None) or getattr(first, "nameNullable", None)


def _version_from_raw(raw: dict) -> str | None:
    for key in ("currentVersionNumber", "versionNumber", "datasetVersionNumber"):
        v = raw.get(key)
        if v is not None:
            return str(v)
    return None


def _parse_datetime(raw) -> datetime | None:
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

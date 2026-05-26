"""On-disk dataset layout and manifest.

Layout:
    {root}/datasets/{profile_id}/{owner}__{slug}/{version}/
        <files from Kaggle, preserved as-is>
        _manifest.json   (full DownloadRecord snapshot)

This module owns the path scheme, the file index (sha256, format,
size), and the manifest. It does NOT fetch from Kaggle; the client
writes the files into the dest dir, then index_directory walks it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from src.domain.download import DownloadRecord, FileEntry

KNOWN_FORMATS = {
    "csv", "parquet", "json", "txt", "tsv",
    "xlsx", "xls", "md", "yaml", "yml", "tar", "zip",
}
MANIFEST_NAME = "_manifest.json"


def dataset_dir(
    root: Path, profile_id: str, owner: str, slug: str, version: str | None
) -> Path:
    """Canonical directory for one (profile, dataset, version)."""
    v = version or "latest"
    return Path(root) / "datasets" / profile_id / f"{owner}__{slug}" / v


def index_directory(dir_path: Path) -> list[FileEntry]:
    """Walk dir_path and return one FileEntry per file (manifest excluded).

    Files are reported in sorted relative-path order so the manifest is
    deterministic across runs.
    """
    if not dir_path.exists():
        return []

    entries: list[FileEntry] = []
    for path in sorted(dir_path.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        rel = path.relative_to(dir_path).as_posix()
        entries.append(
            FileEntry(
                name=path.name,
                relative_path=rel,
                size_bytes=path.stat().st_size,
                format=_detect_format(path),
                sha256=_sha256(path),
            )
        )
    return entries


def write_manifest(dir_path: Path, record: DownloadRecord) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / MANIFEST_NAME).write_text(
        record.model_dump_json(indent=2), encoding="utf-8"
    )


def read_manifest(dir_path: Path) -> DownloadRecord | None:
    path = dir_path / MANIFEST_NAME
    if not path.exists():
        return None
    return DownloadRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _detect_format(path: Path) -> str:
    ext = path.suffix.lstrip(".").lower()
    if not ext:
        return "other"
    if ext in KNOWN_FORMATS:
        return ext
    return "other"


def _sha256(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

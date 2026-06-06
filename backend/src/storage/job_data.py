"""Durable per-job dataset storage: the raw Kaggle bundle plus a manifest.

The pipeline persists every file Kaggle returns under
``{local_storage_path}/{job_id}/raw/`` and writes a :class:`DatasetManifest`
describing them to ``{local_storage_path}/{job_id}/manifest.json``. This is the
job-scoped record downstream reads. It is separate from the cleaned ``/tmp``
working snapshot (see ``storage.cleanup.CAUSAL_TEMP_DIR``).
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from src.config import get_settings
from src.domain import DatasetManifest, ManifestFile
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

MANIFEST_NAME = "manifest.json"
_SHA_CHUNK = 1 << 20  # 1 MiB


def job_data_dir(job_id: str) -> Path:
    """The durable per-job directory under the local storage root."""
    return Path(get_settings().local_storage_path) / job_id


def job_raw_dir(job_id: str) -> Path:
    """The directory holding a job's unzipped bundle (created if absent)."""
    raw = job_data_dir(job_id) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    return raw


def reset_job_raw_dir(job_id: str) -> Path:
    """Return a clean raw/ dir for the job, dropping any prior contents.

    Used before a fresh download so a re-run of the same job_id does not
    leave stale files from a previous bundle alongside the new one.
    """
    raw = job_data_dir(job_id) / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True, exist_ok=True)
    return raw


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_SHA_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_rowcount(path: Path) -> int | None:
    """Cheap data-row count for a csv (line count minus the header)."""
    try:
        with path.open("rb") as fh:
            n = sum(1 for _ in fh)
        return max(n - 1, 0)
    except Exception:
        return None


def _columns_and_rowcount(path: Path, fmt: str) -> tuple[list[str], int | None]:
    """Read column names and a row count without loading all rows.

    csv: header only + a line count. parquet: file metadata (num_rows) and
    schema names, so a large parquet is never materialised. Non-tabular or
    unreadable files yield ([], None).
    """
    if not path.is_file():
        return [], None
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv" or fmt == "csv":
            import pandas as pd

            columns = [str(c) for c in pd.read_csv(path, nrows=0).columns]
            return columns, _csv_rowcount(path)
        if suffix == ".parquet" or fmt == "parquet":
            import pyarrow.parquet as pq

            num_rows = pq.read_metadata(path).num_rows
            names = [str(n) for n in pq.read_schema(path).names]
            return names, int(num_rows)
    except Exception:
        return [], None
    return [], None


def _dataset_ref(url: str) -> str | None:
    parts = url.rstrip("/").split("/")
    if "datasets" in parts:
        idx = parts.index("datasets")
        if idx + 2 < len(parts):
            return f"{parts[idx + 1]}/{parts[idx + 2]}"
    return None


def build_manifest(state) -> DatasetManifest:
    """Assemble the typed manifest from the downloaded bundle.

    Pulls the file list and winner flag from ``state.dataset_info.files``, the
    full Kaggle metadata dict from ``state.raw_metadata``, and reads each
    file's columns, row count, and content hash directly from disk in the
    job's raw dir.
    """
    info = state.dataset_info
    raw = job_raw_dir(state.job_id)

    files: list[ManifestFile] = []
    winner: str | None = None
    for entry in info.files:
        path = raw / entry.name
        columns, n_rows = _columns_and_rowcount(path, entry.format)
        files.append(
            ManifestFile(
                name=entry.name,
                relative_path=f"raw/{entry.name}",
                size_bytes=entry.size_bytes,
                format=entry.format,
                sha256=_sha256(path) if path.is_file() else "",
                n_rows=n_rows,
                n_columns=len(columns) or None,
                columns=columns,
                used=entry.used,
            )
        )
        if entry.used:
            winner = entry.name

    return DatasetManifest(
        job_id=state.job_id,
        kaggle_url=info.url,
        dataset_ref=_dataset_ref(info.url),
        raw_dir=str(raw),
        files=files,
        winner=winner,
        kaggle_metadata=state.raw_metadata or {},
    )


def write_manifest(job_id: str, manifest: DatasetManifest) -> Path:
    """Write the manifest to ``{job_dir}/manifest.json``, overwriting."""
    path = job_data_dir(job_id) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))
    return path


def read_manifest(job_id: str) -> DatasetManifest | None:
    """Load the manifest for a job, or None if absent or unreadable."""
    path = job_data_dir(job_id) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return DatasetManifest.model_validate_json(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("manifest_read_failed", job_id=job_id, error=str(exc))
        return None


def _read_page(
    path: Path, fmt: str, offset: int, limit: int
) -> tuple[list[str], list[dict]]:
    import json

    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv" or fmt == "csv":
        df = pd.read_csv(
            path,
            skiprows=range(1, offset + 1) if offset else None,
            nrows=limit,
        )
    else:
        df = pd.read_parquet(path).iloc[offset : offset + limit]
    columns = [str(c) for c in df.columns]
    rows = json.loads(df.to_json(orient="records"))
    return columns, rows


def read_file_page(
    job_id: str, file_name: str, offset: int, limit: int
) -> dict | None:
    """Read one page of rows from a job's raw file, resolved via the manifest.

    Returns ``{columns, rows, total_rows, offset, limit}``, or None when the
    manifest or the named file is absent. Only files listed in the manifest
    are readable and the resolved path must sit directly inside the job's raw
    dir, so an arbitrary or traversing file_name cannot escape it.
    """
    manifest = read_manifest(job_id)
    if manifest is None:
        return None
    entry = next((f for f in manifest.files if f.name == file_name), None)
    if entry is None:
        return None

    raw = job_raw_dir(job_id).resolve()
    path = (raw / file_name).resolve()
    if path.parent != raw or not path.is_file():
        return None

    offset = max(offset, 0)
    columns, rows = _read_page(path, entry.format, offset, limit)
    return {
        "columns": columns,
        "rows": rows,
        "total_rows": entry.n_rows,
        "offset": offset,
        "limit": limit,
    }

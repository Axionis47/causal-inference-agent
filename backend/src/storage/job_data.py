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


def job_normalized_dir(job_id: str) -> Path:
    """The directory holding a job's canonical parquet files (created if absent).

    Every tabular raw file is converted to one parquet here at the gate, so the
    row preview and the analysis both read parquet regardless of source format.
    """
    norm = job_data_dir(job_id) / "normalized"
    norm.mkdir(parents=True, exist_ok=True)
    return norm


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


def _dataset_ref(url: str) -> str | None:
    parts = url.rstrip("/").split("/")
    if "datasets" in parts:
        idx = parts.index("datasets")
        if idx + 2 < len(parts):
            return f"{parts[idx + 1]}/{parts[idx + 2]}"
    return None


def build_manifest(state, normalize_results=None) -> DatasetManifest:
    """Assemble the typed manifest from the downloaded bundle.

    Pulls the file list and used flag from ``state.dataset_info.files`` and the
    full Kaggle metadata dict from ``state.raw_metadata``. Each file's columns,
    row count, and normalised parquet path come from the gate's normalisation
    pass; ``normalize_results`` carries those, and when omitted the pass is run
    here so a direct caller still gets a complete manifest. The content hash is
    read from the raw file on disk.
    """
    from src.storage.normalize import normalize_bundle

    info = state.dataset_info
    raw = job_raw_dir(state.job_id)
    results = normalize_results if normalize_results is not None else normalize_bundle(
        state.job_id
    )
    by_name = {r.name: r for r in results}

    files: list[ManifestFile] = []
    winner: str | None = None
    for entry in info.files:
        path = raw / entry.name
        r = by_name.get(entry.name)
        columns = r.columns if r else []
        normalized_path = (
            f"normalized/{r.normalized_name}" if r and r.normalized_name else None
        )
        files.append(
            ManifestFile(
                name=entry.name,
                relative_path=f"raw/{entry.name}",
                size_bytes=entry.size_bytes,
                format=entry.format,
                sha256=_sha256(path) if path.is_file() else "",
                n_rows=r.n_rows if r else None,
                n_columns=len(columns) or None,
                columns=columns,
                used=entry.used,
                normalized_path=normalized_path,
                tabular=bool(r.tabular) if r else False,
                normalize_error=r.error if r else None,
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


def read_file_page(
    job_id: str, file_name: str, offset: int, limit: int
) -> dict | None:
    """Read one page of rows from a file's normalised parquet, via the manifest.

    Returns ``{columns, rows, total_rows, offset, limit}``, or None when the
    manifest or the named file is absent. A file with no normalised parquet
    (non-tabular, or a manifest written before normalisation existed) returns an
    empty page rather than None, so the UI can show "no preview" without an
    error. Only files listed in the manifest are readable and the resolved path
    must sit directly inside the job's normalized dir, so an arbitrary or
    traversing file_name cannot escape it.
    """
    import json

    import pandas as pd

    manifest = read_manifest(job_id)
    if manifest is None:
        return None
    entry = next((f for f in manifest.files if f.name == file_name), None)
    if entry is None:
        return None

    offset = max(offset, 0)
    empty = {
        "columns": [],
        "rows": [],
        "total_rows": entry.n_rows,
        "offset": offset,
        "limit": limit,
    }
    if not entry.normalized_path:
        return empty

    norm = job_normalized_dir(job_id).resolve()
    path = (norm / Path(entry.normalized_path).name).resolve()
    if path.parent != norm or not path.is_file():
        return empty

    df = pd.read_parquet(path).iloc[offset : offset + limit]
    return {
        "columns": [str(c) for c in df.columns],
        "rows": json.loads(df.to_json(orient="records")),
        "total_rows": entry.n_rows,
        "offset": offset,
        "limit": limit,
    }

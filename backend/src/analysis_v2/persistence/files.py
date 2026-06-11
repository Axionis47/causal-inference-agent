"""Artifact file IO for one job's analysis run.

All artifact files live under {LOCAL_STORAGE_PATH}/{job_id}/analysis/,
inside the per-job data dir the storage layer already owns and cleans.
Registry entries store paths relative to that dir; this module is the only
place that resolves them, and it refuses anything escaping the dir.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.storage.job_data import job_data_dir


def analysis_dir(job_id: str) -> Path:
    path = job_data_dir(job_id) / "analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_artifact_path(job_id: str, relative_path: str) -> Path:
    """Resolve a registry-relative path, refusing traversal out of the dir."""
    base = analysis_dir(job_id).resolve()
    candidate = (base / relative_path).resolve()
    if base != candidate and base not in candidate.parents:
        raise ValueError(f"artifact path escapes the analysis dir: {relative_path!r}")
    return candidate


def write_artifact_bytes(job_id: str, relative_path: str, data: bytes) -> Path:
    path = resolve_artifact_path(job_id, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_artifact_text(job_id: str, relative_path: str, text: str) -> Path:
    return write_artifact_bytes(job_id, relative_path, text.encode("utf-8"))


def write_artifact_json(job_id: str, relative_path: str, payload: Any) -> Path:
    # default=str keeps numpy scalars, datetimes, and Paths serializable.
    return write_artifact_text(
        job_id, relative_path, json.dumps(payload, indent=2, default=str)
    )


def read_artifact_bytes(job_id: str, relative_path: str) -> bytes:
    return resolve_artifact_path(job_id, relative_path).read_bytes()

"""Dataset-view assembly for the live Data panel.

The frontend Data panel renders three independent blocks (download,
kaggle metadata, profile) each with its own status. The blocks must
distinguish three kinds of "missing" so the UI can render them
differently:

    - pending: we haven't reached this step yet
    - unavailable: the source genuinely does not provide it (Kaggle
      dataset has no description / no per-column descriptions)
    - error: fetch ran and failed

Helpers in this module build a DatasetViewResponse from either the
live in-memory AnalysisState (preferred) or the persisted results dict
(fallback for completed jobs that have been evicted from active state).
"""

from __future__ import annotations

import logging
from typing import Any

from src.analysis.agents.base import AnalysisState, JobStatus
from src.api.schemas import (
    DatasetViewResponse,
    DownloadBlock,
    FileEntryResponse,
    KaggleMetaBlock,
    KaggleMetaData,
    ProfileBlock,
)
from src.storage.job_data import read_manifest

logger = logging.getLogger(__name__)

# JobStatus values that indicate the data-fetching phase has begun.
_PROGRESSED_STATUSES = {
    s for s in JobStatus if s not in (JobStatus.PENDING, JobStatus.CANCELLED)
}


def _files_from_manifest(manifest: Any) -> list[FileEntryResponse]:
    """Per-file response entries from a DatasetManifest (model or dumped dict).

    Each entry carries the file's columns and row count so the Data panel can
    show the shape and drive the row-preview dropdown. Rows themselves are not
    here; they are fetched per page from the rows endpoint.
    """
    if manifest is None:
        return []
    raw_files = manifest.files if hasattr(manifest, "files") else manifest.get("files", [])
    entries: list[FileEntryResponse] = []
    for f in raw_files:
        d = f.model_dump() if hasattr(f, "model_dump") else dict(f)
        entries.append(
            FileEntryResponse(
                name=d.get("name", ""),
                size_bytes=d.get("size_bytes", 0),
                format=d.get("format", "other"),
                used=bool(d.get("used")),
                columns=d.get("columns") or [],
                n_rows=d.get("n_rows"),
            )
        )
    return entries


def build_from_state(state: AnalysisState) -> DatasetViewResponse:
    """Build a dataset view from a live AnalysisState."""
    info = state.dataset_info

    download_failed = (
        state.status == JobStatus.FAILED
        and (state.error_agent or "").lower() == "data_profiler"
    )

    if download_failed:
        download_status = "failed"
    elif info.files or info.local_path:
        download_status = "downloaded"
    elif state.status in _PROGRESSED_STATUSES:
        download_status = "downloading"
    else:
        download_status = "pending"

    download = DownloadBlock(
        status=download_status,
        url=info.url,
        files=_files_from_manifest(read_manifest(state.job_id)),
        error=state.error_message if download_failed else None,
    )

    has_meta = bool(
        info.kaggle_description
        or info.kaggle_column_descriptions
        or info.kaggle_tags
        or info.kaggle_domain
    )
    if has_meta:
        kaggle_meta = KaggleMetaBlock(
            status="loaded",
            data=KaggleMetaData(
                description=info.kaggle_description,
                column_descriptions=info.kaggle_column_descriptions,
                tags=info.kaggle_tags,
                domain=info.kaggle_domain,
                metadata_quality=info.metadata_quality,
            ),
        )
    elif info.metadata_quality != "unknown":
        # Fetch ran but yielded a truly sparse / empty record (Kaggle
        # didn't provide it). Distinct from "we haven't fetched yet".
        kaggle_meta = KaggleMetaBlock(status="unavailable")
    elif download_failed:
        # Download failed before metadata could even be fetched.
        kaggle_meta = KaggleMetaBlock(status="unavailable")
    else:
        kaggle_meta = KaggleMetaBlock(status="pending")

    if state.data_profile:
        profile = ProfileBlock(
            status="loaded",
            data={
                "n_samples": state.data_profile.n_samples,
                "n_features": state.data_profile.n_features,
                "feature_types": state.data_profile.feature_types,
                "missing_values": state.data_profile.missing_values,
                "treatment_candidates": state.data_profile.treatment_candidates,
                "outcome_candidates": state.data_profile.outcome_candidates,
                "potential_confounders": state.data_profile.potential_confounders,
                "potential_instruments": state.data_profile.potential_instruments,
            },
        )
    elif download_failed:
        profile = ProfileBlock(status="error", error=state.error_message)
    else:
        profile = ProfileBlock(status="pending")

    return DatasetViewResponse(
        download=download, kaggle_meta=kaggle_meta, profile=profile
    )


def build_from_persisted(
    job: dict[str, Any], results: dict[str, Any] | None
) -> DatasetViewResponse:
    """Build a dataset view from a completed job's persisted record.

    Live state has been evicted; we have less information but enough to
    render the panel for completed jobs that the user revisits later.
    """
    results = results or {}

    files_data = results.get("dataset_files", []) or []
    manifest_dict = results.get("dataset_manifest")
    job_status = (job.get("status") or "").lower()
    is_failed = job_status in (JobStatus.FAILED.value, JobStatus.CANCELLED.value)

    if files_data or manifest_dict:
        download_status = "downloaded"
    elif is_failed:
        download_status = "failed"
    else:
        download_status = "pending"

    # Prefer the manifest (carries columns + row counts); fall back to the
    # bare dataset_files list for jobs persisted before the manifest existed.
    if manifest_dict:
        files = _files_from_manifest(manifest_dict)
    else:
        files = [FileEntryResponse(**f) for f in files_data]

    download = DownloadBlock(
        status=download_status,
        url=job.get("kaggle_url"),
        files=files,
        error=job.get("error_message") if is_failed else None,
    )

    kaggle_data = results.get("kaggle_meta")
    if kaggle_data:
        kaggle_meta = KaggleMetaBlock(
            status="loaded", data=KaggleMetaData(**kaggle_data)
        )
    elif files_data or manifest_dict:
        # Download succeeded but no kaggle metadata was persisted —
        # treat as unavailable rather than pending so the UI doesn't
        # spin forever on a finished job.
        kaggle_meta = KaggleMetaBlock(status="unavailable")
    else:
        kaggle_meta = KaggleMetaBlock(status="pending")

    profile_data = results.get("data_profile")
    if profile_data:
        profile = ProfileBlock(status="loaded", data=profile_data)
    elif is_failed:
        profile = ProfileBlock(status="error", error=job.get("error_message"))
    else:
        profile = ProfileBlock(status="pending")

    return DatasetViewResponse(
        download=download, kaggle_meta=kaggle_meta, profile=profile
    )

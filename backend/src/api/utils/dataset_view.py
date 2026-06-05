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

import json
import logging
from typing import Any

from src.analysis.agents.base import AnalysisState, JobStatus
from src.api.schemas import (
    DatasetViewResponse,
    DownloadBlock,
    FileEntryResponse,
    FileSampleResponse,
    KaggleMetaBlock,
    KaggleMetaData,
    ProfileBlock,
    SampleRowsBlock,
)

logger = logging.getLogger(__name__)

# How many real rows to surface in the Data panel preview.
_SAMPLE_ROW_LIMIT = 10

# JobStatus values that indicate the data-fetching phase has begun.
_PROGRESSED_STATUSES = {
    s for s in JobStatus if s not in (JobStatus.PENDING, JobStatus.CANCELLED)
}


def _build_sample(
    file_samples: list[Any], used_names: set[str], fallback_path: str | None
) -> SampleRowsBlock:
    """Assemble the per-file raw-row preview block.

    `file_samples` is the list captured at download time (FileSample
    models or their dumped dicts). When present, one preview per tabular
    file is returned, with `used` marking the file we loaded. When absent
    (older jobs predating per-file capture), we fall back to a single
    preview read from the loaded parquet so the panel still shows rows.
    """
    if file_samples:
        files = [_to_file_sample(fs, used_names) for fs in file_samples]
        status = "loaded" if any(f.rows for f in files) else "unavailable"
        return SampleRowsBlock(status=status, files=files)
    return _sample_from_path(fallback_path)


def _to_file_sample(fs: Any, used_names: set[str]) -> FileSampleResponse:
    """Normalise a FileSample (model or dict) into the response shape."""
    d = fs.model_dump() if hasattr(fs, "model_dump") else dict(fs)
    name = d.get("name", "")
    return FileSampleResponse(
        name=name,
        used=name in used_names,
        columns=d.get("columns") or [],
        rows=d.get("rows") or [],
        total_rows=d.get("total_rows"),
        error=d.get("error"),
    )


def _sample_from_path(path: str | None) -> SampleRowsBlock:
    """Fallback single-file preview read from the loaded parquet/csv."""
    if not path:
        return SampleRowsBlock(status="pending")
    try:
        import pandas as pd

        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            return SampleRowsBlock(status="unavailable")
    except FileNotFoundError:
        return SampleRowsBlock(status="pending")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("sample-rows read failed for %s: %s", path, exc)
        return SampleRowsBlock(status="error", error=str(exc))

    head = df.head(_SAMPLE_ROW_LIMIT)
    # to_json round-trips numpy types and NaN -> null cleanly; json.loads
    # gives plain python the response model can serialize.
    rows = json.loads(head.to_json(orient="records"))
    from pathlib import Path

    return SampleRowsBlock(
        status="loaded",
        files=[
            FileSampleResponse(
                name=Path(path).name,
                used=True,
                columns=[str(c) for c in df.columns],
                rows=rows,
                total_rows=int(len(df)),
            )
        ],
    )


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
        files=[FileEntryResponse(**f.model_dump()) for f in info.files],
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

    used_names = {f.name for f in info.files if f.used}
    sample = _build_sample(
        info.file_samples, used_names, state.dataframe_path or info.local_path
    )

    return DatasetViewResponse(
        download=download, kaggle_meta=kaggle_meta, profile=profile, sample=sample
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
    job_status = (job.get("status") or "").lower()
    is_failed = job_status in (JobStatus.FAILED.value, JobStatus.CANCELLED.value)

    if files_data:
        download_status = "downloaded"
    elif is_failed:
        download_status = "failed"
    else:
        download_status = "pending"

    download = DownloadBlock(
        status=download_status,
        url=job.get("kaggle_url"),
        files=[FileEntryResponse(**f) for f in files_data],
        error=job.get("error_message") if is_failed else None,
    )

    kaggle_data = results.get("kaggle_meta")
    if kaggle_data:
        kaggle_meta = KaggleMetaBlock(
            status="loaded", data=KaggleMetaData(**kaggle_data)
        )
    elif files_data:
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

    used_names = {f.get("name") for f in files_data if f.get("used")}
    sample = _build_sample(
        results.get("dataset_file_samples") or [],
        used_names,
        results.get("dataframe_path"),
    )

    return DatasetViewResponse(
        download=download, kaggle_meta=kaggle_meta, profile=profile, sample=sample
    )

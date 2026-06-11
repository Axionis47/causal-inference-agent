"""Load the analysis frame for a confirmed job.

The PICKUP contract: the data lives at the manifest winner's normalized
parquet under the job data dir. state.dataframe_path is never set by the
input slice, so the manifest is the source of truth; raw CSV is the
fallback for pre-normalization bundles. Human-confirmed ignored_columns
are dropped here, once, so every agent sees the same frame.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis_v2.state import AnalysisState
from src.storage.job_data import job_normalized_dir, job_raw_dir, read_manifest


class DatasetUnavailable(RuntimeError):
    """The confirmed job's dataset could not be located on disk."""


def load_analysis_frame(state: AnalysisState) -> pd.DataFrame:
    job_id = state.job_id
    manifest = read_manifest(job_id)
    if manifest is None or manifest.winner is None:
        raise DatasetUnavailable(f"job {job_id} has no dataset manifest winner")

    entry = next((f for f in manifest.files if f.name == manifest.winner), None)
    if entry is None:
        raise DatasetUnavailable(
            f"manifest winner {manifest.winner!r} is not in the file list"
        )

    frame: pd.DataFrame | None = None
    if entry.normalized_path:
        parquet = job_normalized_dir(job_id) / Path(entry.normalized_path).name
        if parquet.exists():
            frame = pd.read_parquet(parquet)
    if frame is None:
        raw = job_raw_dir(job_id) / manifest.winner
        if not raw.exists():
            raise DatasetUnavailable(f"no normalized or raw file for {manifest.winner!r}")
        frame = pd.read_csv(raw)

    ignored = [c for c in state.ignored_columns if c in frame.columns]
    if ignored:
        frame = frame.drop(columns=ignored)
    return frame

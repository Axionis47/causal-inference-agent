"""Load the analysis frame for a confirmed job.

The PICKUP contract: the data lives at the manifest winner's normalized
parquet under the job data dir. state.dataframe_path is never set by the
input slice, so the manifest is the source of truth; raw CSV is the
fallback for pre-normalization bundles. Human-confirmed ignored_columns
are dropped here, once, so every agent sees the same frame. Boolean
columns are coerced to numeric here for the same reason: one seam, every
lane gets a matrix estimators can consume.
"""
from __future__ import annotations

import pandas as pd

from src.analysis_v2.spec import AssemblyPlan
from src.analysis_v2.state import AnalysisState
from src.storage.job_data import read_manifest

from .assembly import DatasetUnavailable, assemble_frame

__all__ = ["DatasetUnavailable", "coerce_bool_columns", "load_analysis_frame"]


def load_analysis_frame(
    state: AnalysisState, plan: AssemblyPlan | None = None
) -> pd.DataFrame:
    """The one seam every agent reads the dataset through. With no plan, loads
    the single manifest winner (today's behavior); with a plan, the executor
    assembles the bundle. Human-confirmed ignored_columns and bool coercion
    are applied here, once, so every lane sees the same frame."""
    job_id = state.job_id
    manifest = read_manifest(job_id)
    if manifest is None or manifest.winner is None:
        raise DatasetUnavailable(f"job {job_id} has no dataset manifest winner")

    if plan is None:
        plan = AssemblyPlan.single_file(manifest.winner)
    frame = assemble_frame(job_id, manifest, plan)

    ignored = [c for c in state.ignored_columns if c in frame.columns]
    if ignored:
        frame = frame.drop(columns=ignored)
    return coerce_bool_columns(frame)


def coerce_bool_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast bool columns to numeric; shared with the generated notebook.

    np.asarray on a frame mixing bool and float columns yields dtype=object,
    which statsmodels and sklearn reject; treatment flags often arrive as
    True/False in CSVs.
    """
    for col in frame.columns:
        if pd.api.types.is_bool_dtype(frame[col]):
            frame[col] = frame[col].astype(
                "float64" if frame[col].isna().any() else "int8"
            )
    return frame

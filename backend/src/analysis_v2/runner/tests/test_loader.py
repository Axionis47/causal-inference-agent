"""load_analysis_frame: the one seam every agent reads the dataset through."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis_v2.runner.loader import DatasetUnavailable, load_analysis_frame
from src.analysis_v2.state import AnalysisState, DatasetInfo, JobStatus
from src.domain.dataset_manifest import DatasetManifest, ManifestFile
from src.storage.job_data import job_normalized_dir, job_raw_dir, write_manifest

JOB_ID = "job-loader"


@pytest.fixture
def storage_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod
    from src.storage.local_storage import reset_local_storage_client

    settings_mod.get_settings.cache_clear()
    reset_local_storage_client()
    yield tmp_path
    settings_mod.get_settings.cache_clear()
    reset_local_storage_client()


def _state() -> AnalysisState:
    return AnalysisState(
        job_id=JOB_ID,
        dataset_info=DatasetInfo(
            url="https://www.kaggle.com/datasets/konradb/ihdp-data",
            name="ihdp",
            user_provided_context="IHDP benchmark; treatment is program assignment.",
        ),
        causal_question="Does the intervention improve cognitive scores?",
        status=JobStatus.CONFIRMED,
    )


def _stage(frame: pd.DataFrame, with_parquet: bool = True) -> None:
    raw = job_raw_dir(JOB_ID)
    frame.to_csv(raw / "data.csv", index=False)
    normalized_path = None
    if with_parquet:
        frame.to_parquet(job_normalized_dir(JOB_ID) / "data.csv.parquet", index=False)
        normalized_path = "normalized/data.csv.parquet"
    write_manifest(
        JOB_ID,
        DatasetManifest(
            job_id=JOB_ID,
            kaggle_url="https://www.kaggle.com/datasets/konradb/ihdp-data",
            raw_dir=str(raw),
            winner="data.csv",
            files=[
                ManifestFile(
                    name="data.csv",
                    relative_path="raw/data.csv",
                    size_bytes=1,
                    format="csv",
                    sha256="0" * 64,
                    used=True,
                    normalized_path=normalized_path,
                    tabular=True,
                )
            ],
        ),
    )


def test_load_analysis_frame_coerces_bool_treatment_to_numeric_for_estimators(
    storage_dir,
):
    _stage(
        pd.DataFrame(
            {"treatment": [True, False, True], "y_factual": [5.6, 6.9, 4.8]}
        )
    )

    frame = load_analysis_frame(_state())

    assert frame["treatment"].dtype.kind in "iu"
    assert frame["treatment"].tolist() == [1, 0, 1]
    assert np.asarray(frame).dtype != object


def test_load_analysis_frame_falls_back_to_raw_csv_and_still_returns_numeric_frame(
    storage_dir,
):
    _stage(
        pd.DataFrame({"treatment": [True, False], "y_factual": [5.6, 6.9]}),
        with_parquet=False,
    )

    frame = load_analysis_frame(_state())

    assert frame["treatment"].dtype.kind in "iu"
    assert np.asarray(frame).dtype != object


def test_load_analysis_frame_casts_nullable_boolean_with_missing_values_to_float(
    storage_dir,
):
    _stage(
        pd.DataFrame(
            {
                "flag": pd.array([True, None, False], dtype="boolean"),
                "y_factual": [5.6, 6.9, 4.8],
            }
        )
    )

    frame = load_analysis_frame(_state())

    assert frame["flag"].dtype == np.float64
    assert frame["flag"][0] == 1.0 and np.isnan(frame["flag"][1])


def test_load_analysis_frame_raises_when_no_manifest_exists(storage_dir):
    with pytest.raises(DatasetUnavailable):
        load_analysis_frame(_state())

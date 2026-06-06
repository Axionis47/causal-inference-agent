"""GET /jobs/{id}/dataset serves the parked state at the approval gate.

A job paused at AWAITING_APPROVAL has no active state (the task ended), but
the data the user is reviewing must still render. The endpoint loads the
parked state rather than falling through to the not-yet-saved completed-job
results (which would render an empty view).
"""
from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.analysis.agents.base import DataProfile
from src.analysis.agents.base.state import (
    AnalysisState,
    DatasetInfo,
    FileEntry,
    JobStatus,
)
from src.api.main import app
from src.domain import DatasetManifest, ManifestFile
from src.storage.job_data import write_manifest


def _parked_state(job_id: str) -> AnalysisState:
    state = AnalysisState(
        job_id=job_id,
        dataset_info=DatasetInfo(url="https://www.kaggle.com/datasets/o/n"),
    )
    state.status = JobStatus.AWAITING_APPROVAL
    state.dataset_info.files = [
        FileEntry(name="lalonde.csv", size_bytes=9400, format="csv", used=True)
    ]
    state.data_profile = DataProfile(
        n_samples=614,
        n_features=11,
        feature_names=["treat"],
        feature_types={"treat": "binary"},
        missing_values={"treat": 0},
        numeric_stats={},
        categorical_stats={},
        treatment_candidates=["treat"],
        outcome_candidates=["re78"],
    )
    return state


def test_dataset_view_serves_parked_state_at_gate(tmp_path, monkeypatch):
    job_id = "job-parked"
    fake = types.SimpleNamespace(local_storage_path=str(tmp_path / "store"))
    monkeypatch.setattr("src.storage.job_data.get_settings", lambda: fake)
    write_manifest(
        job_id,
        DatasetManifest(
            job_id=job_id,
            kaggle_url="https://www.kaggle.com/datasets/o/n",
            raw_dir=str(tmp_path),
            files=[
                ManifestFile(
                    name="lalonde.csv",
                    relative_path="raw/lalonde.csv",
                    size_bytes=9400,
                    format="csv",
                    sha256="x",
                    n_rows=614,
                    n_columns=11,
                    columns=["treat", "age"],
                    used=True,
                )
            ],
            winner="lalonde.csv",
        ),
    )

    manager = MagicMock()
    manager.get_job = AsyncMock(
        return_value={
            "status": "awaiting_approval",
            "kaggle_url": "https://www.kaggle.com/datasets/o/n",
        }
    )
    manager.get_active_state = MagicMock(return_value=None)
    manager.get_parked_state = AsyncMock(return_value=_parked_state(job_id))

    with patch("src.api.routes.jobs.get_job_manager", return_value=manager):
        client = TestClient(app)
        resp = client.get(f"/jobs/{job_id}/dataset")

    assert resp.status_code == 200
    body = resp.json()
    assert body["download"]["status"] == "downloaded"
    assert body["download"]["files"][0]["name"] == "lalonde.csv"
    assert body["download"]["files"][0]["n_rows"] == 614
    assert body["profile"]["status"] == "loaded"
    manager.get_parked_state.assert_awaited_once()

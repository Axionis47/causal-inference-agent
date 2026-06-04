"""GET + POST /jobs/{id}/approval endpoint contracts.

The UI depends on these two endpoints: GET returns the gate snapshot
(404 if the job is not parked), POST applies the human decision and
respawns the orchestrator. The handlers are thin shells around
JobManager.get_parked_snapshot / resume_from_approval; these tests pin
the HTTP layer behaviour with the manager mocked out.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_manager():
    m = MagicMock()
    m.get_job = AsyncMock(return_value=None)
    m.get_parked_snapshot = AsyncMock(return_value=None)
    m.resume_from_approval = AsyncMock(return_value={"resumed": True, "status": "discovering_causal"})
    return m


@pytest.fixture
def client(mock_manager):
    with patch("src.api.routes.jobs.get_job_manager", return_value=mock_manager):
        from src.api.main import app
        yield TestClient(app)


# --- GET /jobs/{id}/approval -----------------------------------------------


def test_get_approval_returns_404_when_not_parked(client, mock_manager):
    mock_manager.get_parked_snapshot.return_value = None
    resp = client.get("/jobs/job-1/approval")
    assert resp.status_code == 404


def test_get_approval_returns_snapshot_payload(client, mock_manager):
    mock_manager.get_parked_snapshot.return_value = {
        "treatment_variable": "t",
        "outcome_variable": "y",
        "eda_summary": {"data_quality_score": 82.0, "data_quality_issues": []},
        "proposed_dag": {"nodes": ["t", "y"], "edges": [], "adjustment_set": ["x1"]},
        "brief_flags": {"eda_agent": {"status": "done", "headline": "ok", "flags": [], "raised_issues": []}},
    }
    resp = client.get("/jobs/job-1/approval")
    assert resp.status_code == 200
    body = resp.json()
    assert body["treatment_variable"] == "t"
    assert body["proposed_dag"]["adjustment_set"] == ["x1"]
    assert "eda_agent" in body["brief_flags"]


# --- POST /jobs/{id}/approval ----------------------------------------------


def _job_row(status_value: str) -> dict:
    return {
        "id": "job-1",
        "kaggle_url": "https://www.kaggle.com/datasets/x/y",
        "status": status_value,
        "created_at": "2026-06-01T00:00:00Z",
        "updated_at": "2026-06-01T00:00:00Z",
    }


def test_post_approval_404_when_job_missing(client, mock_manager):
    mock_manager.get_job.return_value = None
    resp = client.post("/jobs/job-1/approval", json={"decision": "approved"})
    assert resp.status_code == 404


def test_post_approval_409_when_job_not_awaiting(client, mock_manager):
    mock_manager.get_job.return_value = _job_row("estimating_effects")
    resp = client.post("/jobs/job-1/approval", json={"decision": "approved"})
    assert resp.status_code == 409


def test_post_approval_approved_path_respawns(client, mock_manager):
    mock_manager.get_job.return_value = _job_row("awaiting_approval")
    mock_manager.resume_from_approval.return_value = {
        "resumed": True,
        "status": "discovering_causal",
    }
    resp = client.post(
        "/jobs/job-1/approval",
        json={
            "decision": "approved",
            "granted_by": "analyst",
            "dag_edits": {"adjustment_set": ["age", "education"]},
            "appended_context": "ignore re74",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "job-1"
    assert body["resumed"] is True
    assert body["status"] == "discovering_causal"
    # Verify the approval that reached the manager carried the edits.
    mock_manager.resume_from_approval.assert_awaited_once()
    args = mock_manager.resume_from_approval.await_args.args
    approval = args[1]
    assert approval.decision.value == "approved"
    assert approval.dag_edits.adjustment_set == ["age", "education"]
    assert approval.appended_context == "ignore re74"


def test_post_approval_rejected_requires_reason(client, mock_manager):
    mock_manager.get_job.return_value = _job_row("awaiting_approval")
    resp = client.post("/jobs/job-1/approval", json={"decision": "rejected"})
    assert resp.status_code == 422
    assert "reason" in resp.json()["detail"].lower()


def test_post_approval_rejected_path_does_not_respawn(client, mock_manager):
    mock_manager.get_job.return_value = _job_row("awaiting_approval")
    mock_manager.resume_from_approval.return_value = {
        "resumed": False,
        "status": "failed",
    }
    resp = client.post(
        "/jobs/job-1/approval",
        json={"decision": "rejected", "reason": "adjustment set is wrong"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["resumed"] is False
    assert body["status"] == "failed"

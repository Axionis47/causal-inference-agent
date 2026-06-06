"""Tests for per-request orchestrator_mode selection.

Covers the three layers the field flows through:
1. CreateJobRequest schema accepts/rejects values.
2. JobManager.create_job records the choice on AnalysisState and
   _create_orchestrator builds the matching orchestrator class.
3. POST /jobs honours the field end-to-end (mocked manager).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError


# --------------------------------------------------------------------- #
# Schema layer                                                          #
# --------------------------------------------------------------------- #

class TestCreateJobRequestOrchestratorMode:
    """CreateJobRequest validation for the new field."""

    def test_accepts_standard(self):
        from src.api.schemas import CreateJobRequest

        req = CreateJobRequest(
            kaggle_url="https://www.kaggle.com/datasets/owner/name",
            orchestrator_mode="standard",
        )
        assert req.orchestrator_mode == "standard"

    def test_accepts_react(self):
        from src.api.schemas import CreateJobRequest

        req = CreateJobRequest(
            kaggle_url="https://www.kaggle.com/datasets/owner/name",
            orchestrator_mode="react",
        )
        assert req.orchestrator_mode == "react"

    def test_defaults_to_none_when_omitted(self):
        from src.api.schemas import CreateJobRequest

        req = CreateJobRequest(
            kaggle_url="https://www.kaggle.com/datasets/owner/name",
        )
        assert req.orchestrator_mode is None

    def test_rejects_unknown_value(self):
        from src.api.schemas import CreateJobRequest

        with pytest.raises(ValidationError):
            CreateJobRequest(
                kaggle_url="https://www.kaggle.com/datasets/owner/name",
                orchestrator_mode="chaos",
            )


# --------------------------------------------------------------------- #
# JobManager layer                                                      #
# --------------------------------------------------------------------- #

@pytest.fixture()
def _mock_manager_deps():
    fake_settings = MagicMock()
    fake_settings.max_concurrent_jobs = 2
    fake_settings.agent_timeout_seconds = 300
    fake_settings.instance_id = "test1234"

    with (
        patch("src.config.get_settings", return_value=fake_settings),
        patch("src.jobs.manager.get_settings", return_value=fake_settings),
        patch("src.jobs.manager.get_storage_client") as mock_storage,
        patch("src.jobs.manager.standard_pkg.build_specialists", return_value={}),
        patch("src.jobs.manager.react_pkg.build_specialists", return_value={}),
    ):
        mock_storage.return_value = MagicMock()
        mock_storage.return_value.create_job = AsyncMock()
        mock_storage.return_value.create_job_if_capacity = AsyncMock(return_value=True)
        mock_storage.return_value.update_job = AsyncMock(return_value=True)
        mock_storage.return_value.save_results = AsyncMock()
        mock_storage.return_value.save_traces = AsyncMock()
        yield fake_settings


class TestJobManagerOrchestratorMode:
    """create_job records the choice on AnalysisState."""

    @pytest.mark.asyncio
    async def test_create_job_honors_requested_react_mode(self, _mock_manager_deps):
        from src.jobs.manager import JobManager

        manager = JobManager(orchestrator_mode="standard")
        manager._run_job = AsyncMock()  # prevent real execution

        job_id = await manager.create_job(
            kaggle_url="https://www.kaggle.com/datasets/owner/name",
            orchestrator_mode="react",
        )
        # State seeded for _run_job is captured via the AsyncMock call args.
        called_state = manager._run_job.call_args.args[0]
        assert called_state.job_id == job_id
        assert called_state.orchestrator_mode == "react"

    @pytest.mark.asyncio
    async def test_create_job_defaults_to_manager_mode(self, _mock_manager_deps):
        from src.jobs.manager import JobManager

        manager = JobManager(orchestrator_mode="standard")
        manager._run_job = AsyncMock()

        await manager.create_job(
            kaggle_url="https://www.kaggle.com/datasets/owner/name",
        )
        called_state = manager._run_job.call_args.args[0]
        assert called_state.orchestrator_mode == "standard"

    @pytest.mark.asyncio
    async def test_create_orchestrator_returns_react_class(self, _mock_manager_deps):
        from src.analysis.orchestrator import ReActOrchestrator, StandardOrchestrator
        from src.jobs.manager import JobManager

        manager = JobManager(orchestrator_mode="standard")
        assert isinstance(manager._create_orchestrator("react"), ReActOrchestrator)
        assert isinstance(manager._create_orchestrator("standard"), StandardOrchestrator)


# --------------------------------------------------------------------- #
# Endpoint layer                                                        #
# --------------------------------------------------------------------- #

class TestCreateJobEndpointOrchestratorMode:
    """POST /jobs honours and validates the field."""

    @pytest.fixture
    def mock_job_manager(self):
        m = MagicMock()
        m.create_job = AsyncMock(return_value="abc12345")
        m.get_job = AsyncMock(return_value={
            "id": "abc12345",
            "kaggle_url": "https://www.kaggle.com/datasets/owner/name",
            "status": "pending",
            "created_at": "2026-05-10T00:00:00Z",
            "updated_at": "2026-05-10T00:00:00Z",
        })
        return m

    @pytest.fixture
    def client(self, mock_job_manager):
        with patch("src.api.routes.jobs.get_job_manager", return_value=mock_job_manager):
            from src.api.main import app
            yield TestClient(app)

    def test_react_mode_forwarded_to_manager(self, client, mock_job_manager):
        with patch("src.api.routes.jobs.get_job_manager", return_value=mock_job_manager):
            response = client.post(
                "/jobs",
                json={
                    "kaggle_url": "https://www.kaggle.com/datasets/owner/name",
                    "orchestrator_mode": "react",
                },
            )
            assert response.status_code == 201
            mock_job_manager.create_job.assert_awaited_once()
            kwargs = mock_job_manager.create_job.call_args.kwargs
            assert kwargs["orchestrator_mode"] == "react"

    def test_missing_mode_forwarded_as_none(self, client, mock_job_manager):
        with patch("src.api.routes.jobs.get_job_manager", return_value=mock_job_manager):
            response = client.post(
                "/jobs",
                json={"kaggle_url": "https://www.kaggle.com/datasets/owner/name"},
            )
            assert response.status_code == 201
            kwargs = mock_job_manager.create_job.call_args.kwargs
            assert kwargs["orchestrator_mode"] is None

    def test_invalid_mode_returns_422(self, client):
        response = client.post(
            "/jobs",
            json={
                "kaggle_url": "https://www.kaggle.com/datasets/owner/name",
                "orchestrator_mode": "chaos",
            },
        )
        assert response.status_code == 422

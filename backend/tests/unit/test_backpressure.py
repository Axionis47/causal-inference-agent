"""Tests for job backpressure via asyncio.Semaphore.

Verifies that JobManager rejects new jobs when at capacity and
that the semaphore correctly limits concurrent job execution.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def _mock_deps():
    """Mock all external dependencies of JobManager."""
    fake_settings = MagicMock()
    fake_settings.max_concurrent_jobs = 2
    fake_settings.agent_timeout_seconds = 300
    fake_settings.instance_id = "test1234"

    with (
        patch("src.config.get_settings", return_value=fake_settings),
        patch("src.jobs.manager.get_settings", return_value=fake_settings),
        patch("src.jobs.manager.get_storage_client") as mock_storage,
        patch("src.jobs.manager.OrchestratorAgent"),
        patch("src.jobs.manager.create_all_agents", return_value={}),
    ):
        mock_storage.return_value = MagicMock()
        mock_storage.return_value.create_job = AsyncMock()
        mock_storage.return_value.create_job_if_capacity = AsyncMock(return_value=True)
        mock_storage.return_value.update_job = AsyncMock(return_value=True)
        mock_storage.return_value.save_results = AsyncMock()
        mock_storage.return_value.save_traces = AsyncMock()
        yield fake_settings


class TestBackpressure:
    """Semaphore-based concurrency limiting."""

    @pytest.mark.asyncio
    async def test_semaphore_created_from_settings(self, _mock_deps):
        from src.jobs.manager import JobManager

        manager = JobManager()
        assert manager._job_semaphore._value == 2

    @pytest.mark.asyncio
    async def test_rejects_when_at_capacity(self, _mock_deps):
        from src.jobs.manager import JobManager

        manager = JobManager()

        # Exhaust semaphore
        await manager._job_semaphore.acquire()
        await manager._job_semaphore.acquire()

        with pytest.raises(RuntimeError, match="Server at capacity"):
            await manager.create_job(kaggle_url="https://kaggle.com/datasets/owner/test")

    @pytest.mark.asyncio
    async def test_accepts_when_under_capacity(self, _mock_deps):
        from src.jobs.manager import JobManager

        manager = JobManager()

        # Mock _run_job to prevent actual execution
        manager._run_job = AsyncMock()

        # Should not raise
        job_id = await manager.create_job(kaggle_url="https://kaggle.com/datasets/owner/test")
        assert isinstance(job_id, str)
        assert len(job_id) == 8

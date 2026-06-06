"""Tests for local job-artifact cleanup, including the durable dataset dir."""

from __future__ import annotations

import pytest

from src.domain import DatasetManifest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point both storage roots (durable dir and /tmp snapshot) at temp dirs."""
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path / "store"))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    monkeypatch.setattr("src.storage.cleanup.CAUSAL_TEMP_DIR", tmp_path / "tmp")
    yield tmp_path
    settings_mod.get_settings.cache_clear()


def test_cleanup_removes_durable_job_dir(isolated):
    from src.storage.cleanup import cleanup_local_artifacts
    from src.storage.job_data import job_data_dir, job_raw_dir, write_manifest

    job_id = "job-clean"
    raw = job_raw_dir(job_id)
    (raw / "big.csv").write_text("a\n1\n")
    write_manifest(
        job_id, DatasetManifest(job_id=job_id, kaggle_url="u", raw_dir=str(raw))
    )
    assert job_data_dir(job_id).exists()

    cleaned = cleanup_local_artifacts(job_id)

    assert cleaned["job_data"] is True
    assert not job_data_dir(job_id).exists()


def test_cleanup_reports_false_when_no_artifacts(isolated):
    from src.storage.cleanup import cleanup_local_artifacts

    cleaned = cleanup_local_artifacts("never-existed")

    assert cleaned["job_data"] is False
    assert cleaned["dataframe"] is False
    assert cleaned["notebook"] is False

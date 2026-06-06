"""save_results embeds the dataset manifest so completed jobs can read it.

The live AnalysisState is evicted after a job finishes; the persisted results
record is what /jobs/{id}/dataset hydrates from later. The dataset manifest
(per-file records + full Kaggle metadata) must ride along in that record.
"""

from __future__ import annotations

import pytest

from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.domain import DatasetManifest, ManifestFile
from src.storage.job_data import write_manifest
from src.storage.local_storage import LocalStorageClient


@pytest.fixture
def storage(tmp_path, monkeypatch) -> LocalStorageClient:
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    client = LocalStorageClient()
    yield client
    settings_mod.get_settings.cache_clear()


@pytest.mark.asyncio
async def test_save_results_embeds_manifest_when_present(storage, tmp_path):
    job_id = "job-m"
    write_manifest(
        job_id,
        DatasetManifest(
            job_id=job_id,
            kaggle_url="https://www.kaggle.com/datasets/owner/name",
            dataset_ref="owner/name",
            raw_dir=str(tmp_path / job_id / "raw"),
            files=[
                ManifestFile(
                    name="d.csv",
                    relative_path="raw/d.csv",
                    size_bytes=1,
                    format="csv",
                    sha256="x",
                    used=True,
                )
            ],
            winner="d.csv",
            kaggle_metadata={"description": "study"},
        ),
    )

    state = AnalysisState(
        job_id=job_id,
        dataset_info=DatasetInfo(url="https://www.kaggle.com/datasets/owner/name"),
    )
    await storage.save_results(state)

    results = await storage.get_results(job_id)
    assert "dataset_manifest" in results
    assert results["dataset_manifest"]["winner"] == "d.csv"
    assert results["dataset_manifest"]["files"][0]["name"] == "d.csv"
    assert results["dataset_manifest"]["kaggle_metadata"]["description"] == "study"


@pytest.mark.asyncio
async def test_save_results_omits_manifest_when_absent(storage):
    job_id = "job-no-manifest"
    state = AnalysisState(
        job_id=job_id,
        dataset_info=DatasetInfo(url="https://www.kaggle.com/datasets/owner/name"),
    )
    await storage.save_results(state)

    results = await storage.get_results(job_id)
    assert "dataset_manifest" not in results

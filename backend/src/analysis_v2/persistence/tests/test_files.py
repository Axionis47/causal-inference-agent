"""Artifact file IO: everything stays inside the job's analysis dir."""
from __future__ import annotations

import json

import pytest

from src.analysis_v2.persistence import (
    analysis_dir,
    read_artifact_bytes,
    resolve_artifact_path,
    write_artifact_json,
    write_artifact_text,
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the storage path at a temp dir so no real ./data is touched."""
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    yield tmp_path
    settings_mod.get_settings.cache_clear()


def test_analysis_dir_is_created_under_the_job_data_dir(data_dir):
    path = analysis_dir("job-7")
    assert path == data_dir / "job-7" / "analysis"
    assert path.is_dir()


def test_text_and_json_artifacts_round_trip_with_nested_paths(data_dir):
    write_artifact_text("job-7", "intake/summary.md", "## Findings\nplain text")
    payload = {"question_type": "did", "confidence": "medium"}
    write_artifact_json("job-7", "intake/causal_spec.json", payload)

    assert read_artifact_bytes("job-7", "intake/summary.md").decode().startswith("## Findings")
    loaded = json.loads(read_artifact_bytes("job-7", "intake/causal_spec.json"))
    assert loaded == payload


def test_traversal_and_absolute_paths_are_refused(data_dir):
    with pytest.raises(ValueError, match="escapes"):
        resolve_artifact_path("job-7", "../other-job/secret.json")
    with pytest.raises(ValueError, match="escapes"):
        resolve_artifact_path("job-7", "a/../../../etc/passwd")
    # absolute paths resolve outside the analysis dir and are refused too
    with pytest.raises(ValueError, match="escapes"):
        resolve_artifact_path("job-7", "/etc/passwd")

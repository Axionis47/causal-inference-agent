"""Tests that the dataset loading module emits the dataset-lifecycle SSE events.

The frontend Data panel mounts at t=0 and progressively fills in three
blocks (download, kaggle metadata, profile) as backend events fire. The
events are the only path to live data; until the profiler completes,
no API endpoint exposes partial state. So the events MUST fire at the
right moments and carry the right payloads, or the panel will stall.

These tests pin the contract:
    - fetch_kaggle_metadata emits started + ready on success
    - fetch_kaggle_metadata emits started + failed on extractor error
    - load_from_kaggle emits download_started + download_complete on success
    - load_from_kaggle emits dataset_load_failed when the API raises
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.analysis.agents.base import AnalysisState, DatasetInfo
from src.analysis.agents.data_profiler.loading import (
    download_to_workdir,
    fetch_kaggle_metadata,
    load_dataset,
    load_from_kaggle,
    pick_default_file,
)


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    """Keep every bundle this module writes under a temp dir, never the real
    ./data storage root. Without this the kaggle-mock tests share job id
    "t-sse" on the real root and leak into one another and the wider suite
    (see CLAUDE.md 4.0). Storage isolation, not credentials: the kaggle creds
    read a different get_settings and the API itself is mocked per test."""
    import types

    fake = types.SimpleNamespace(local_storage_path=str(tmp_path / "store"))
    monkeypatch.setattr("src.storage.job_data.get_settings", lambda: fake)


def _make_state(url: str = "https://www.kaggle.com/datasets/owner/name") -> AnalysisState:
    return AnalysisState(
        job_id="t-sse",
        dataset_info=DatasetInfo(url=url, name="t"),
    )


def _event_types(state: AnalysisState) -> list[str]:
    return [e["event_type"] for e in state.sse_events]


@pytest.mark.asyncio
async def test_fetch_kaggle_metadata_emits_started_and_ready_on_success():
    state = _make_state()

    fake_metadata = {
        "extraction_success": True,
        "description": "Job-training RCT.",
        "column_descriptions": {"treat": "treatment indicator"},
        "tags": ["economics", "rct"],
        "metadata_quality": "high",
    }

    with patch(
        "src.kaggle.metadata_extractor.KaggleMetadataExtractor"
    ) as mock_extractor_cls:
        mock_extractor_cls.return_value.extract = AsyncMock(return_value=fake_metadata)
        await fetch_kaggle_metadata(state)

    types = _event_types(state)
    assert types == ["dataset_metadata_started", "dataset_metadata_ready"]

    ready = state.sse_events[1]["data"]
    assert ready["description"] == "Job-training RCT."
    assert ready["column_descriptions"] == {"treat": "treatment indicator"}
    assert ready["tags"] == ["economics", "rct"]
    assert ready["metadata_quality"] == "high"

    # State is populated as expected too.
    assert state.dataset_info.kaggle_description == "Job-training RCT."
    assert state.dataset_info.metadata_quality == "high"


@pytest.mark.asyncio
async def test_fetch_kaggle_metadata_emits_failed_on_extractor_exception():
    state = _make_state()

    with patch(
        "src.kaggle.metadata_extractor.KaggleMetadataExtractor"
    ) as mock_extractor_cls:
        mock_extractor_cls.return_value.extract = AsyncMock(
            side_effect=RuntimeError("network down")
        )
        await fetch_kaggle_metadata(state)

    types = _event_types(state)
    assert types == ["dataset_metadata_started", "dataset_metadata_failed"]
    assert "network down" in state.sse_events[1]["data"]["error"]


@pytest.mark.asyncio
async def test_fetch_kaggle_metadata_emits_failed_on_unsuccessful_extraction():
    state = _make_state()

    with patch(
        "src.kaggle.metadata_extractor.KaggleMetadataExtractor"
    ) as mock_extractor_cls:
        mock_extractor_cls.return_value.extract = AsyncMock(
            return_value={"extraction_success": False, "error": "rate limited"}
        )
        await fetch_kaggle_metadata(state)

    types = _event_types(state)
    assert types == ["dataset_metadata_started", "dataset_metadata_failed"]
    assert state.sse_events[1]["data"]["error"] == "rate limited"


@pytest.mark.asyncio
async def test_load_from_kaggle_emits_started_and_complete_on_success(tmp_path):
    state = _make_state("https://www.kaggle.com/datasets/owner/name")

    df_payload = pd.DataFrame({"treat": [0, 1, 0], "y": [1.0, 2.0, 3.0]})

    def fake_download(dataset_id: str, path: str, unzip: bool):
        Path(path, "data.csv").write_text(df_payload.to_csv(index=False))

    fake_api = MagicMock()
    fake_api.authenticate.return_value = None
    fake_api.dataset_download_files.side_effect = fake_download

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        result_df, _ = await load_from_kaggle(state, state.dataset_info.url)

    assert result_df is not None
    assert list(result_df.columns) == ["treat", "y"]

    types = _event_types(state)
    assert types == ["dataset_download_started", "dataset_download_complete"]
    started = state.sse_events[0]["data"]
    complete = state.sse_events[1]["data"]
    assert started["dataset_id"] == "owner/name"
    assert started["url"] == state.dataset_info.url
    assert complete["rows"] == 3
    assert complete["columns"] == 2
    # File list is captured into state and into the event payload.
    assert len(state.dataset_info.files) == 1
    assert state.dataset_info.files[0].name == "data.csv"
    assert state.dataset_info.files[0].used is True
    assert state.dataset_info.files[0].format == "csv"
    assert complete["files"][0]["name"] == "data.csv"


@pytest.mark.asyncio
async def test_load_from_kaggle_captures_multiple_files_with_used_flag(tmp_path):
    """When the archive contains several CSVs, the largest is loaded and the
    others appear in the file list with used=False."""
    state = _make_state("https://www.kaggle.com/datasets/owner/name")

    big = pd.DataFrame({"a": list(range(50)), "b": list(range(50))})
    small = pd.DataFrame({"a": [1], "b": [2]})

    def fake_download(dataset_id: str, path: str, unzip: bool):
        Path(path, "big.csv").write_text(big.to_csv(index=False))
        Path(path, "small.csv").write_text(small.to_csv(index=False))
        Path(path, "README.txt").write_text("notes")

    fake_api = MagicMock()
    fake_api.authenticate.return_value = None
    fake_api.dataset_download_files.side_effect = fake_download

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        result_df, _ = await load_from_kaggle(state, state.dataset_info.url)

    assert result_df is not None and len(result_df) == 50

    files_by_name = {f.name: f for f in state.dataset_info.files}
    assert set(files_by_name.keys()) == {"big.csv", "small.csv", "README.txt"}
    assert files_by_name["big.csv"].used is True
    assert files_by_name["small.csv"].used is False
    assert files_by_name["README.txt"].used is False
    assert files_by_name["README.txt"].format == "txt"
    # download_complete payload mirrors state.dataset_info.files
    complete = state.sse_events[-1]["data"]
    assert {f["name"] for f in complete["files"]} == {
        "big.csv",
        "small.csv",
        "README.txt",
    }


@pytest.mark.asyncio
async def test_load_from_kaggle_persists_raw_bundle_and_manifest(tmp_path):
    """Every downloaded file must survive on disk after the call returns (the
    bundle is no longer wiped), and a typed manifest must describe it."""
    from src.storage.job_data import job_raw_dir, read_manifest

    state = _make_state("https://www.kaggle.com/datasets/owner/name")

    big = pd.DataFrame({"a": list(range(50)), "b": list(range(50))})
    small = pd.DataFrame({"a": [1], "b": [2]})

    def fake_download(dataset_id: str, path: str, unzip: bool):
        Path(path, "big.csv").write_text(big.to_csv(index=False))
        Path(path, "small.csv").write_text(small.to_csv(index=False))

    fake_api = MagicMock()
    fake_api.authenticate.return_value = None
    fake_api.dataset_download_files.side_effect = fake_download

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        result_df, _ = await load_from_kaggle(state, state.dataset_info.url)

    assert result_df is not None

    # Both files persist after the call returns; nothing was wiped.
    raw = job_raw_dir(state.job_id)
    assert (raw / "big.csv").is_file()
    assert (raw / "small.csv").is_file()

    # The manifest is a typed record of the bundle with the winner marked.
    manifest = read_manifest(state.job_id)
    assert manifest is not None
    assert manifest.job_id == state.job_id
    assert manifest.dataset_ref == "owner/name"
    assert {f.name for f in manifest.files} == {"big.csv", "small.csv"}
    assert manifest.winner == "big.csv"
    big_rec = next(f for f in manifest.files if f.name == "big.csv")
    assert big_rec.used is True
    assert big_rec.n_rows == 50
    assert big_rec.n_columns == 2
    assert big_rec.relative_path == "raw/big.csv"
    assert len(big_rec.sha256) == 64  # sha256 hex digest

    # The bundle is normalised to parquet at the gate: the manifest records the
    # path, and the parquet really exists on disk.
    from src.storage.job_data import job_normalized_dir

    assert big_rec.tabular is True
    assert big_rec.normalized_path == "normalized/big.csv.parquet"
    assert (job_normalized_dir(state.job_id) / "big.csv.parquet").is_file()


@pytest.mark.asyncio
async def test_load_from_kaggle_emits_failed_when_download_raises():
    state = _make_state("https://www.kaggle.com/datasets/owner/name")

    fake_api = MagicMock()
    fake_api.authenticate.return_value = None
    fake_api.dataset_download_files.side_effect = RuntimeError(
        "403 Forbidden: dataset is private"
    )

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        result_df, _ = await load_from_kaggle(state, state.dataset_info.url)

    assert result_df is None
    types = _event_types(state)
    # download_started fires (we got past URL parsing) then load_failed.
    assert types[0] == "dataset_download_started"
    assert types[-1] == "dataset_load_failed"
    # Error string is the user-facing translation, not the raw exception.
    assert "not accessible" in state.sse_events[-1]["data"]["error"]


@pytest.mark.asyncio
async def test_load_from_kaggle_emits_failed_for_invalid_url():
    state = _make_state("https://example.com/not-a-kaggle-url")

    fake_api = MagicMock()
    fake_api.authenticate.return_value = None

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        result_df, _ = await load_from_kaggle(state, state.dataset_info.url)

    assert result_df is None
    types = _event_types(state)
    # URL parse fails before download_started would fire.
    assert types == ["dataset_load_failed"]
    assert "Invalid Kaggle URL" in state.sse_events[0]["data"]["error"]


# ── pick_default_file: the well-defined fallback heuristic ────────────────


def test_pick_default_file_returns_largest_csv(tmp_path):
    (tmp_path / "small.csv").write_text("a,b\n1,2\n")
    big = tmp_path / "big.csv"
    big.write_text("a,b\n" + "1,2\n" * 1000)
    assert pick_default_file(tmp_path) == big


def test_pick_default_file_falls_back_to_first_parquet_when_no_csv(tmp_path):
    parquet = tmp_path / "data.parquet"
    parquet.write_bytes(b"\x00")  # contents irrelevant; we only test selection
    assert pick_default_file(tmp_path) == parquet


def test_pick_default_file_returns_none_when_directory_has_no_data_files(tmp_path):
    (tmp_path / "README.md").write_text("hi")
    (tmp_path / "schema.json").write_text("{}")
    assert pick_default_file(tmp_path) is None


def test_pick_default_file_prefers_csv_over_parquet_when_both_present(tmp_path):
    """Regression guard: if dataset_inspector ever wraps this helper, the
    csv-over-parquet preference must survive."""
    (tmp_path / "data.csv").write_text("a\n1\n")
    (tmp_path / "data.parquet").write_bytes(b"\x00")
    assert pick_default_file(tmp_path).suffix == ".csv"


# ── load_dataset honours a pre-set dataframe_path ─────────────────────────


# ── download_to_workdir: persistent workdir for the multi-file inspector ──


@pytest.mark.asyncio
async def test_download_to_workdir_returns_files_and_workdir_on_success(
    tmp_path, monkeypatch
):
    """The inspector's per-file flow needs the archive on disk in a
    persistent location so every candidate can be loaded once and
    re-read N times across parallel inner profiles."""
    from src.analysis.agents.data_profiler import loading as loading_mod

    sandbox = tmp_path / "causal_orchestrator"
    monkeypatch.setattr(
        "src.storage.cleanup.CAUSAL_TEMP_DIR", sandbox
    )

    state = _make_state()

    def _fake_dataset_download_files(dataset_id, path, unzip):
        # Mirror what Kaggle would write into the workdir.
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "train.csv").write_text("a,b\n1,2\n")
        (Path(path) / "test.csv").write_text("a,b\n3,4\n")

    fake_api = MagicMock()
    fake_api.authenticate.return_value = None
    fake_api.dataset_download_files.side_effect = _fake_dataset_download_files

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        workdir, files, error = await download_to_workdir(
            state, state.dataset_info.url
        )

    assert error is None
    assert workdir is not None
    assert workdir.exists()
    # workdir must be job-scoped so concurrent jobs do not collide.
    assert state.job_id in workdir.name
    # capture_file_list snapshots every file (no used flag here — the
    # picker is the inspector, not this loader).
    assert {f.name for f in files} == {"train.csv", "test.csv"}
    assert all(f.used is False for f in files)


@pytest.mark.asyncio
async def test_download_to_workdir_returns_error_for_invalid_url(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.storage.cleanup.CAUSAL_TEMP_DIR", tmp_path / "causal"
    )
    state = _make_state(url="https://www.kaggle.com/not-a-dataset")

    fake_api = MagicMock()
    fake_api.authenticate.return_value = None

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        workdir, files, error = await download_to_workdir(
            state, state.dataset_info.url
        )

    assert workdir is None
    assert files == []
    assert error is not None
    assert "Invalid Kaggle URL" in error


@pytest.mark.asyncio
async def test_download_to_workdir_propagates_error_when_kaggle_raises(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "src.storage.cleanup.CAUSAL_TEMP_DIR", tmp_path / "causal"
    )
    state = _make_state()

    fake_api = MagicMock()
    fake_api.authenticate.return_value = None
    fake_api.dataset_download_files.side_effect = RuntimeError("403 Forbidden")

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        workdir, files, error = await download_to_workdir(
            state, state.dataset_info.url
        )

    assert workdir is None
    assert files == []
    assert error is not None
    assert "403" in error


@pytest.mark.asyncio
async def test_load_dataset_short_circuits_when_dataframe_path_is_set(tmp_path):
    """When dataset_inspector populates state.dataframe_path with the
    winner's parquet, the loader must read that file directly instead of
    falling through to Kaggle (which would re-download)."""
    csv = tmp_path / "chosen.csv"
    csv.write_text("a,b\n1,2\n3,4\n")

    state = _make_state()
    state.dataframe_path = str(csv)

    df, error = await load_dataset(state)
    assert error is None
    assert df is not None
    assert df.shape == (2, 2)
    # No SSE events should fire on the short-circuit path — the file was
    # already chosen upstream, so download_started must not be emitted.
    assert _event_types(state) == []


@pytest.mark.asyncio
async def test_load_from_kaggle_reads_existing_bundle_without_redownloading(tmp_path):
    """When a prior download for this job_id has left a manifest plus its
    winner file on disk (as the data-review gate does), the loader reads that
    file directly. Kaggle is never authenticated or pulled, and no
    download_started event fires."""
    from src.storage.job_data import job_raw_dir, write_manifest
    from src.domain import DatasetManifest, ManifestFile

    state = _make_state("https://www.kaggle.com/datasets/owner/name")

    raw = job_raw_dir(state.job_id)
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "winner.csv").write_text("a,b\n1,2\n3,4\n5,6\n")
    write_manifest(
        state.job_id,
        DatasetManifest(
            job_id=state.job_id,
            kaggle_url=state.dataset_info.url,
            raw_dir=str(raw),
            files=[
                ManifestFile(
                    name="winner.csv",
                    relative_path="raw/winner.csv",
                    size_bytes=18,
                    format="csv",
                    sha256="x",
                    n_rows=3,
                    n_columns=2,
                    columns=["a", "b"],
                    used=True,
                )
            ],
            winner="winner.csv",
        ),
    )

    boom = MagicMock(side_effect=AssertionError("Kaggle must not be touched"))
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", boom):
        df, error = await load_from_kaggle(state, state.dataset_info.url)

    assert error is None
    assert df is not None
    assert df.shape == (3, 2)
    assert _event_types(state) == []
    assert {f.name for f in state.dataset_info.files} == {"winner.csv"}

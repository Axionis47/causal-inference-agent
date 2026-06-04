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
    fetch_kaggle_metadata,
    load_dataset,
    load_from_kaggle,
    pick_default_file,
)


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

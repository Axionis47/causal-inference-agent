"""Tests that data_profiler emits dataset-lifecycle SSE events.

The frontend Data panel mounts at t=0 and progressively fills in three
blocks (download, kaggle metadata, profile) as backend events fire. The
events are the only path to live data — until the profiler completes,
no API endpoint exposes partial state. So the events MUST fire at the
right moments and carry the right payloads, or the panel will stall.

These tests pin the contract:
    - _fetch_kaggle_metadata emits started + ready on success
    - _fetch_kaggle_metadata emits started + failed on extractor error
    - _load_from_kaggle emits download_started + download_complete on success
    - _load_from_kaggle emits dataset_load_failed when the API raises
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.agents.base import AnalysisState, DatasetInfo
from src.agents.specialists.data_profiler import DataProfilerAgent


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
    agent = DataProfilerAgent()

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
        await agent._fetch_kaggle_metadata(state)

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
    agent = DataProfilerAgent()

    with patch(
        "src.kaggle.metadata_extractor.KaggleMetadataExtractor"
    ) as mock_extractor_cls:
        mock_extractor_cls.return_value.extract = AsyncMock(
            side_effect=RuntimeError("network down")
        )
        await agent._fetch_kaggle_metadata(state)

    types = _event_types(state)
    assert types == ["dataset_metadata_started", "dataset_metadata_failed"]
    assert "network down" in state.sse_events[1]["data"]["error"]


@pytest.mark.asyncio
async def test_fetch_kaggle_metadata_emits_failed_on_unsuccessful_extraction():
    state = _make_state()
    agent = DataProfilerAgent()

    with patch(
        "src.kaggle.metadata_extractor.KaggleMetadataExtractor"
    ) as mock_extractor_cls:
        mock_extractor_cls.return_value.extract = AsyncMock(
            return_value={"extraction_success": False, "error": "rate limited"}
        )
        await agent._fetch_kaggle_metadata(state)

    types = _event_types(state)
    assert types == ["dataset_metadata_started", "dataset_metadata_failed"]
    assert state.sse_events[1]["data"]["error"] == "rate limited"


@pytest.mark.asyncio
async def test_load_from_kaggle_emits_started_and_complete_on_success(tmp_path):
    state = _make_state("https://www.kaggle.com/datasets/owner/name")
    agent = DataProfilerAgent()

    df_payload = pd.DataFrame({"treat": [0, 1, 0], "y": [1.0, 2.0, 3.0]})

    def fake_download(dataset_id: str, path: str, unzip: bool):
        Path(path, "data.csv").write_text(df_payload.to_csv(index=False))

    fake_api = MagicMock()
    fake_api.authenticate.return_value = None
    fake_api.dataset_download_files.side_effect = fake_download

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        result_df = await agent._load_from_kaggle(state, state.dataset_info.url)

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


@pytest.mark.asyncio
async def test_load_from_kaggle_emits_failed_when_download_raises():
    state = _make_state("https://www.kaggle.com/datasets/owner/name")
    agent = DataProfilerAgent()

    fake_api = MagicMock()
    fake_api.authenticate.return_value = None
    fake_api.dataset_download_files.side_effect = RuntimeError(
        "403 Forbidden: dataset is private"
    )

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        result_df = await agent._load_from_kaggle(state, state.dataset_info.url)

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
    agent = DataProfilerAgent()

    fake_api = MagicMock()
    fake_api.authenticate.return_value = None

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        result_df = await agent._load_from_kaggle(state, state.dataset_info.url)

    assert result_df is None
    types = _event_types(state)
    # URL parse fails before download_started would fire.
    assert types == ["dataset_load_failed"]
    assert "Invalid Kaggle URL" in state.sse_events[0]["data"]["error"]

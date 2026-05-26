"""KaggleClient: thin wrapper over kaggle.api.

These tests mock the KaggleApi class and assert the wrapper produces
typed KaggleMetadata + a normalized file list, regardless of which
underlying attribute names the Kaggle library exposes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.download.client import KaggleClient
from src.domain.download import KaggleMetadata


def _make_fake_api(view: SimpleNamespace, files_resp: SimpleNamespace) -> MagicMock:
    fake = MagicMock()
    fake.authenticate.return_value = None
    fake.dataset_view.return_value = view
    fake.dataset_list_files.return_value = files_resp
    fake.dataset_download_files.return_value = None
    return fake


@pytest.fixture
def fake_view() -> SimpleNamespace:
    return SimpleNamespace(
        title="Iris",
        subtitle="Classic classification dataset",
        description="The classic Iris dataset.",
        tags=[SimpleNamespace(name="classification"), "tabular"],
        licenseName="CC0-1.0",
        currentVersionNumber=2,
        lastUpdated="2024-08-15T10:30:00Z",
    )


@pytest.fixture
def fake_files() -> SimpleNamespace:
    return SimpleNamespace(
        datasetFiles=[
            SimpleNamespace(name="Iris.csv", totalBytes=4096),
            SimpleNamespace(name="README.md", totalBytes=512),
        ]
    )


@pytest.mark.asyncio
async def test_fetch_metadata_normalizes_view(fake_view, fake_files):
    fake = _make_fake_api(fake_view, fake_files)
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake):
        client = KaggleClient()
        meta = await client.fetch_metadata(
            "uciml", "iris", url="https://www.kaggle.com/datasets/uciml/iris"
        )

    assert isinstance(meta, KaggleMetadata)
    assert meta.owner == "uciml"
    assert meta.slug == "iris"
    assert meta.title == "Iris"
    assert meta.subtitle == "Classic classification dataset"
    assert "classification" in meta.tags
    assert "tabular" in meta.tags
    assert meta.license_name == "CC0-1.0"
    assert meta.version == "2"
    assert meta.last_updated == datetime(2024, 8, 15, 10, 30, tzinfo=timezone.utc)
    assert meta.total_bytes == 4096 + 512


@pytest.mark.asyncio
async def test_fetch_metadata_falls_back_to_overview(fake_files):
    view = SimpleNamespace(
        title="X",
        subtitle=None,
        description=None,
        overview="The overview text",
        tags=[],
        licenseName=None,
        currentVersionNumber=None,
        lastUpdated=None,
    )
    fake = _make_fake_api(view, fake_files)
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake):
        client = KaggleClient()
        meta = await client.fetch_metadata("o", "s", url="https://kaggle.com/datasets/o/s")

    assert meta.description == "The overview text"


@pytest.mark.asyncio
async def test_list_files_returns_normalized_entries(fake_view, fake_files):
    fake = _make_fake_api(fake_view, fake_files)
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake):
        client = KaggleClient()
        files = await client.list_files("uciml", "iris")

    assert files == [
        {"name": "Iris.csv", "size_bytes": 4096},
        {"name": "README.md", "size_bytes": 512},
    ]


@pytest.mark.asyncio
async def test_download_files_calls_api_with_unzip(fake_view, fake_files, tmp_path):
    fake = _make_fake_api(fake_view, fake_files)
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake):
        client = KaggleClient()
        await client.download_files("uciml", "iris", tmp_path / "ds")

    fake.dataset_download_files.assert_called_once()
    call_kwargs = fake.dataset_download_files.call_args.kwargs
    assert call_kwargs["unzip"] is True
    assert call_kwargs["path"] == str(tmp_path / "ds")
    assert "version" not in call_kwargs


@pytest.mark.asyncio
async def test_download_files_passes_version_when_set(fake_view, fake_files, tmp_path):
    fake = _make_fake_api(fake_view, fake_files)
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake):
        client = KaggleClient()
        await client.download_files("uciml", "iris", tmp_path / "ds", version="3")

    assert fake.dataset_download_files.call_args.kwargs["version"] == "3"


@pytest.mark.asyncio
async def test_fetch_metadata_with_no_files_returns_none_total_bytes(fake_view):
    empty_files = SimpleNamespace(datasetFiles=[])
    fake = _make_fake_api(fake_view, empty_files)
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake):
        client = KaggleClient()
        meta = await client.fetch_metadata(
            "o", "s", url="https://www.kaggle.com/datasets/o/s"
        )

    assert meta.total_bytes is None

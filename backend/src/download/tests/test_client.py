"""KaggleClient: thin wrapper over kaggle.api.

These tests mock the KaggleApi class. dataset_metadata writes a
JSON file to disk so the mock side-effects to write the file too.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.download.client import KaggleClient
from src.domain.download import KaggleMetadata


def _metadata_writer(payload: dict):
    """Return a side_effect that mimics dataset_metadata writing the JSON file."""
    def _write(dataset, path):
        Path(path, "dataset-metadata.json").write_text(json.dumps(payload))
        return str(Path(path, "dataset-metadata.json"))
    return _write


def _make_fake_api(metadata_payload: dict, files: SimpleNamespace) -> MagicMock:
    fake = MagicMock()
    fake.authenticate.return_value = None
    fake.dataset_metadata.side_effect = _metadata_writer(metadata_payload)
    fake.dataset_list_files.return_value = files
    fake.dataset_download_files.return_value = None
    return fake


@pytest.fixture
def fake_files() -> SimpleNamespace:
    return SimpleNamespace(
        datasetFiles=[
            SimpleNamespace(name="Iris.csv", totalBytes=4096),
            SimpleNamespace(name="README.md", totalBytes=512),
        ]
    )


@pytest.mark.asyncio
async def test_fetch_metadata_normalizes_real_kaggle_shape(fake_files):
    payload = {
        "title": "Iris Species",
        "subtitle": "Classic classification dataset",
        "description": "The classic Iris dataset.",
        "keywords": ["biology", "classification"],
        "licenses": [{"name": "CC0-1.0"}],
        "currentVersionNumber": 2,
        "lastUpdated": "2024-08-15T10:30:00Z",
    }
    fake = _make_fake_api(payload, fake_files)
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake):
        client = KaggleClient()
        meta = await client.fetch_metadata(
            "uciml", "iris", url="https://www.kaggle.com/datasets/uciml/iris"
        )

    assert isinstance(meta, KaggleMetadata)
    assert meta.title == "Iris Species"
    assert meta.subtitle == "Classic classification dataset"
    assert meta.description == "The classic Iris dataset."
    assert meta.tags == ["biology", "classification"]
    assert meta.license_name == "CC0-1.0"
    assert meta.version == "2"
    assert meta.last_updated == datetime(2024, 8, 15, 10, 30, tzinfo=timezone.utc)
    assert meta.total_bytes == 4096 + 512


@pytest.mark.asyncio
async def test_fetch_metadata_falls_back_to_nullable_fields(fake_files):
    payload = {
        "titleNullable": "X",
        "subtitleNullable": None,
        "descriptionNullable": "Fallback description",
        "keywords": [],
    }
    fake = _make_fake_api(payload, fake_files)
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake):
        client = KaggleClient()
        meta = await client.fetch_metadata("o", "s", url="https://kaggle.com/datasets/o/s")

    assert meta.title == "X"
    assert meta.subtitle is None
    assert meta.description == "Fallback description"


@pytest.mark.asyncio
async def test_fetch_metadata_handles_empty_payload(fake_files):
    fake = _make_fake_api({}, fake_files)
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake):
        client = KaggleClient()
        meta = await client.fetch_metadata("o", "s", url="https://kaggle.com/datasets/o/s")

    assert meta.title is None
    assert meta.description is None
    assert meta.tags == []
    assert meta.license_name is None
    assert meta.version is None


@pytest.mark.asyncio
async def test_list_files_returns_normalized_entries(fake_files):
    fake = _make_fake_api({}, fake_files)
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake):
        client = KaggleClient()
        files = await client.list_files("uciml", "iris")

    assert files == [
        {"name": "Iris.csv", "size_bytes": 4096},
        {"name": "README.md", "size_bytes": 512},
    ]


@pytest.mark.asyncio
async def test_download_files_calls_api_with_unzip(fake_files, tmp_path):
    fake = _make_fake_api({}, fake_files)
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake):
        client = KaggleClient()
        await client.download_files("uciml", "iris", tmp_path / "ds")

    fake.dataset_download_files.assert_called_once()
    kwargs = fake.dataset_download_files.call_args.kwargs
    assert kwargs["unzip"] is True
    assert kwargs["path"] == str(tmp_path / "ds")
    assert "version" not in kwargs


@pytest.mark.asyncio
async def test_download_files_passes_version_when_set(fake_files, tmp_path):
    fake = _make_fake_api({}, fake_files)
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake):
        client = KaggleClient()
        await client.download_files("uciml", "iris", tmp_path / "ds", version="3")

    assert fake.dataset_download_files.call_args.kwargs["version"] == "3"

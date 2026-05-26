"""Auth: env scope and Kaggle key validation.

Env scope is the security boundary: a download for profile A must never
leak its key into profile B's later download in the same process. The
validation tests use a mocked KaggleApi so they run without network.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.download.auth import (
    ValidationResult,
    kaggle_env_scope,
    validate_kaggle_key,
)


def test_env_scope_sets_vars_inside_block():
    os.environ.pop("KAGGLE_USERNAME", None)
    os.environ.pop("KAGGLE_KEY", None)
    with kaggle_env_scope("alice", "secret"):
        assert os.environ["KAGGLE_USERNAME"] == "alice"
        assert os.environ["KAGGLE_KEY"] == "secret"


def test_env_scope_restores_originals_after_block():
    os.environ["KAGGLE_USERNAME"] = "original-user"
    os.environ["KAGGLE_KEY"] = "original-key"
    try:
        with kaggle_env_scope("alice", "secret"):
            assert os.environ["KAGGLE_USERNAME"] == "alice"
        assert os.environ["KAGGLE_USERNAME"] == "original-user"
        assert os.environ["KAGGLE_KEY"] == "original-key"
    finally:
        os.environ.pop("KAGGLE_USERNAME", None)
        os.environ.pop("KAGGLE_KEY", None)


def test_env_scope_clears_when_no_original():
    os.environ.pop("KAGGLE_USERNAME", None)
    os.environ.pop("KAGGLE_KEY", None)
    with kaggle_env_scope("alice", "secret"):
        pass
    assert "KAGGLE_USERNAME" not in os.environ
    assert "KAGGLE_KEY" not in os.environ


def test_env_scope_restores_even_on_exception():
    os.environ.pop("KAGGLE_USERNAME", None)
    with pytest.raises(RuntimeError):
        with kaggle_env_scope("alice", "secret"):
            raise RuntimeError("boom")
    assert "KAGGLE_USERNAME" not in os.environ


def test_env_scope_rejects_empty_credentials():
    with pytest.raises(ValueError):
        with kaggle_env_scope("", ""):
            pass


@pytest.mark.asyncio
async def test_validate_kaggle_key_success():
    fake_api = MagicMock()
    fake_api.authenticate.return_value = None
    fake_api.dataset_list.return_value = []

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        result = await validate_kaggle_key("alice", "secret")

    assert result.valid is True
    assert result.error is None
    fake_api.authenticate.assert_called_once()
    fake_api.dataset_list.assert_called_once_with(page=1, max_size=1)


@pytest.mark.asyncio
async def test_validate_kaggle_key_translates_401():
    fake_api = MagicMock()
    fake_api.authenticate.return_value = None
    fake_api.dataset_list.side_effect = RuntimeError("401 Client Error: Unauthorized")

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        result = await validate_kaggle_key("alice", "bad-key")

    assert result.valid is False
    assert "401" in result.error


@pytest.mark.asyncio
async def test_validate_kaggle_key_translates_403():
    fake_api = MagicMock()
    fake_api.authenticate.return_value = None
    fake_api.dataset_list.side_effect = RuntimeError("403 Forbidden")

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        result = await validate_kaggle_key("alice", "secret")

    assert result.valid is False
    assert "403" in result.error


@pytest.mark.asyncio
async def test_validate_kaggle_key_translates_generic_error():
    fake_api = MagicMock()
    fake_api.authenticate.side_effect = RuntimeError("network unreachable")

    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=fake_api):
        result = await validate_kaggle_key("alice", "secret")

    assert result.valid is False
    assert "network unreachable" in result.error

"""Enhanced API tests: authentication, URL validation, and request handling.

Tests:
- API key authentication (verify_api_key function)
- Kaggle URL validation (CreateJobRequest validator)
- Variable name validation
"""

import pytest


class TestAPIKeyAuth:
    """Tests for API key authentication."""

    def test_verify_api_key_no_key_configured_passes(self):
        """When no API key is configured (dev mode), all requests should pass."""
        from unittest.mock import patch, MagicMock

        # Mock settings to return api_key_value=None
        mock_settings = MagicMock()
        mock_settings.api_key_value = None

        with patch("src.api.main.get_settings", return_value=mock_settings):
            import asyncio
            from src.api.main import verify_api_key

            # Should not raise any exception
            result = asyncio.get_event_loop().run_until_complete(
                verify_api_key(x_api_key=None)
            )
            assert result is None

    def test_verify_api_key_correct_key_passes(self):
        """Request with correct API key should pass."""
        from unittest.mock import patch, MagicMock

        mock_settings = MagicMock()
        mock_settings.api_key_value = "test-secret-key-123"

        with patch("src.api.main.get_settings", return_value=mock_settings):
            import asyncio
            from src.api.main import verify_api_key

            # Should not raise any exception
            result = asyncio.get_event_loop().run_until_complete(
                verify_api_key(x_api_key="test-secret-key-123")
            )
            assert result is None

    def test_verify_api_key_wrong_key_returns_401(self):
        """Request with wrong API key should raise HTTPException 401."""
        from unittest.mock import patch, MagicMock
        from fastapi import HTTPException

        mock_settings = MagicMock()
        mock_settings.api_key_value = "test-secret-key-123"

        with patch("src.api.main.get_settings", return_value=mock_settings):
            import asyncio
            from src.api.main import verify_api_key

            with pytest.raises(HTTPException) as exc_info:
                asyncio.get_event_loop().run_until_complete(
                    verify_api_key(x_api_key="wrong-key")
                )
            assert exc_info.value.status_code == 401

    def test_verify_api_key_missing_key_returns_401(self):
        """Request without API key when one is configured should raise 401."""
        from unittest.mock import patch, MagicMock
        from fastapi import HTTPException

        mock_settings = MagicMock()
        mock_settings.api_key_value = "test-secret-key-123"

        with patch("src.api.main.get_settings", return_value=mock_settings):
            import asyncio
            from src.api.main import verify_api_key

            with pytest.raises(HTTPException) as exc_info:
                asyncio.get_event_loop().run_until_complete(
                    verify_api_key(x_api_key=None)
                )
            assert exc_info.value.status_code == 401


class TestKaggleURLValidation:
    """Tests for Kaggle URL validation in CreateJobRequest."""

    def test_valid_kaggle_url(self):
        """A valid Kaggle dataset URL should pass validation."""
        from src.api.schemas.job import CreateJobRequest

        req = CreateJobRequest(
            kaggle_url="https://www.kaggle.com/datasets/owner/dataset-name"
        )
        assert req.kaggle_url == "https://www.kaggle.com/datasets/owner/dataset-name"

    def test_valid_kaggle_url_trailing_slash(self):
        """A valid Kaggle URL with trailing slash should pass."""
        from src.api.schemas.job import CreateJobRequest

        req = CreateJobRequest(
            kaggle_url="https://www.kaggle.com/datasets/owner/dataset-name/"
        )
        assert req.kaggle_url.startswith("https://www.kaggle.com/datasets/")

    def test_valid_kaggle_url_without_www(self):
        """A valid Kaggle URL without www should pass."""
        from src.api.schemas.job import CreateJobRequest

        req = CreateJobRequest(
            kaggle_url="https://kaggle.com/datasets/owner/dataset-name"
        )
        assert "kaggle.com" in req.kaggle_url

    def test_valid_kaggle_url_http(self):
        """A valid Kaggle URL with http (not https) should pass."""
        from src.api.schemas.job import CreateJobRequest

        req = CreateJobRequest(
            kaggle_url="http://www.kaggle.com/datasets/owner/dataset-name"
        )
        assert "kaggle.com" in req.kaggle_url

    def test_invalid_url_not_kaggle(self):
        """A non-Kaggle URL should be rejected with validation error."""
        from pydantic import ValidationError
        from src.api.schemas.job import CreateJobRequest

        with pytest.raises(ValidationError) as exc_info:
            CreateJobRequest(
                kaggle_url="https://www.google.com/datasets/owner/dataset"
            )
        error_str = str(exc_info.value)
        assert "Invalid Kaggle URL" in error_str or "kaggle" in error_str.lower()

    def test_invalid_url_missing_datasets_path(self):
        """A Kaggle URL missing /datasets/ path should be rejected."""
        from pydantic import ValidationError
        from src.api.schemas.job import CreateJobRequest

        with pytest.raises(ValidationError):
            CreateJobRequest(
                kaggle_url="https://www.kaggle.com/competitions/some-comp"
            )

    def test_invalid_url_missing_owner(self):
        """A Kaggle URL missing owner/name structure should be rejected."""
        from pydantic import ValidationError
        from src.api.schemas.job import CreateJobRequest

        with pytest.raises(ValidationError):
            CreateJobRequest(
                kaggle_url="https://www.kaggle.com/datasets/"
            )

    def test_empty_url_rejected(self):
        """An empty URL should be rejected."""
        from pydantic import ValidationError
        from src.api.schemas.job import CreateJobRequest

        with pytest.raises(ValidationError):
            CreateJobRequest(kaggle_url="")

    def test_whitespace_only_url_rejected(self):
        """A whitespace-only URL should be rejected."""
        from pydantic import ValidationError
        from src.api.schemas.job import CreateJobRequest

        with pytest.raises(ValidationError):
            CreateJobRequest(kaggle_url="   ")

    def test_url_with_hyphens_and_numbers(self):
        """A Kaggle URL with hyphens and numbers should pass."""
        from src.api.schemas.job import CreateJobRequest

        req = CreateJobRequest(
            kaggle_url="https://www.kaggle.com/datasets/user-123/my-dataset-v2"
        )
        assert "my-dataset-v2" in req.kaggle_url


class TestVariableNameValidation:
    """Tests for treatment/outcome variable name validation."""

    def test_valid_variable_name(self):
        """Alphanumeric variable names with underscores should pass."""
        from src.api.schemas.job import CreateJobRequest

        req = CreateJobRequest(
            kaggle_url="https://www.kaggle.com/datasets/owner/dataset-name",
            treatment_variable="treatment_var",
            outcome_variable="outcome_var",
        )
        assert req.treatment_variable == "treatment_var"
        assert req.outcome_variable == "outcome_var"

    def test_none_variable_name(self):
        """None variable names should be accepted."""
        from src.api.schemas.job import CreateJobRequest

        req = CreateJobRequest(
            kaggle_url="https://www.kaggle.com/datasets/owner/dataset-name",
            treatment_variable=None,
            outcome_variable=None,
        )
        assert req.treatment_variable is None
        assert req.outcome_variable is None

    def test_empty_variable_name_becomes_none(self):
        """Empty or whitespace-only variable names should become None."""
        from src.api.schemas.job import CreateJobRequest

        req = CreateJobRequest(
            kaggle_url="https://www.kaggle.com/datasets/owner/dataset-name",
            treatment_variable="",
            outcome_variable="  ",
        )
        assert req.treatment_variable is None
        assert req.outcome_variable is None

    def test_invalid_variable_name_special_chars(self):
        """Variable names with special characters should be rejected."""
        from pydantic import ValidationError
        from src.api.schemas.job import CreateJobRequest

        with pytest.raises(ValidationError):
            CreateJobRequest(
                kaggle_url="https://www.kaggle.com/datasets/owner/dataset-name",
                treatment_variable="var with spaces",
            )

    def test_variable_name_with_hyphens(self):
        """Variable names with hyphens should pass."""
        from src.api.schemas.job import CreateJobRequest

        req = CreateJobRequest(
            kaggle_url="https://www.kaggle.com/datasets/owner/dataset-name",
            treatment_variable="my-var",
        )
        assert req.treatment_variable == "my-var"


class TestKaggleURLPattern:
    """Tests for the KAGGLE_URL_PATTERN regex itself."""

    def test_pattern_matches_standard_url(self):
        """The regex should match standard Kaggle dataset URLs."""
        from src.api.schemas.job import KAGGLE_URL_PATTERN

        assert KAGGLE_URL_PATTERN.match(
            "https://www.kaggle.com/datasets/myuser/my-dataset"
        )

    def test_pattern_does_not_match_arbitrary_url(self):
        """The regex should not match non-Kaggle URLs."""
        from src.api.schemas.job import KAGGLE_URL_PATTERN

        assert not KAGGLE_URL_PATTERN.match(
            "https://www.example.com/datasets/user/data"
        )

    def test_pattern_does_not_match_incomplete_path(self):
        """The regex should not match incomplete Kaggle paths."""
        from src.api.schemas.job import KAGGLE_URL_PATTERN

        assert not KAGGLE_URL_PATTERN.match(
            "https://www.kaggle.com/datasets/"
        )
        assert not KAGGLE_URL_PATTERN.match(
            "https://www.kaggle.com/datasets/user"
        )

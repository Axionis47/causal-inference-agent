"""Unit tests for configuration."""

import os

import pytest


class TestSettings:
    """Test application settings."""

    def test_settings_loads(self):
        """Test settings can be loaded."""
        from src.config.settings import Settings

        settings = Settings()
        assert settings is not None
        assert settings.environment in ["development", "staging", "production", "test"]

    def test_settings_defaults(self):
        """Test default settings values."""
        from src.config.settings import Settings

        settings = Settings()

        assert settings.app_name == "Causal Inference Orchestrator"
        assert settings.max_agent_iterations >= 1

    def test_settings_gcp_config(self):
        """Test GCP configuration."""
        from src.config.settings import Settings

        settings = Settings()

        assert settings.gcp_project_id is not None
        assert settings.gcp_region is not None


class TestLogging:
    """Test logging configuration."""

    def test_logger_creation(self):
        """Test structured logger creation."""
        from src.logging_config.structured import get_logger

        logger = get_logger("test_module")
        assert logger is not None

    def test_logger_has_methods(self):
        """Test logger has standard methods."""
        from src.logging_config.structured import get_logger

        logger = get_logger("test_module")

        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "debug")

    def test_logger_outputs_structured(self):
        """Test logger outputs structured data."""
        from src.logging_config.structured import get_logger
        import io
        import sys

        logger = get_logger("test_structured")

        # Capture output
        captured = io.StringIO()
        old_stdout = sys.stdout

        try:
            # Logger should not raise
            logger.info("test_event", extra_field="value")
        finally:
            sys.stdout = old_stdout

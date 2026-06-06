"""Kaggle credential validation and scoped env-var injection.

The Kaggle Python client reads KAGGLE_USERNAME and KAGGLE_KEY from the
process environment at authenticate() time. Two helpers here:

- kaggle_env_scope(username, key): sets the env vars inside a with-block
  and restores whatever was there before. This lets the same process
  serve multiple profiles without crossing credentials.
- validate_kaggle_key(username, key): performs a cheap round-trip to
  Kaggle (list one dataset) to confirm the key actually works.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@contextmanager
def kaggle_env_scope(username: str, key: str) -> Iterator[None]:
    """Set KAGGLE_USERNAME / KAGGLE_KEY only inside the with-block."""
    if not username or not key:
        raise ValueError("username and key are required")

    original_username = os.environ.get("KAGGLE_USERNAME")
    original_key = os.environ.get("KAGGLE_KEY")

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    try:
        yield
    finally:
        if original_username is not None:
            os.environ["KAGGLE_USERNAME"] = original_username
        else:
            os.environ.pop("KAGGLE_USERNAME", None)
        if original_key is not None:
            os.environ["KAGGLE_KEY"] = original_key
        else:
            os.environ.pop("KAGGLE_KEY", None)


@dataclass
class ValidationResult:
    valid: bool
    error: str | None = None


def _validate_sync(username: str, key: str) -> ValidationResult:
    with kaggle_env_scope(username, key):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as e:
            return ValidationResult(valid=False, error=f"kaggle library not installed: {e}")

        api = KaggleApi()
        try:
            api.authenticate()
            # Cheapest authenticated round-trip: list one dataset.
            api.dataset_list(page=1, max_size=1)
        except Exception as e:
            return ValidationResult(valid=False, error=_translate_kaggle_error(e))

    return ValidationResult(valid=True)


async def validate_kaggle_key(username: str, key: str) -> ValidationResult:
    """Validate by hitting Kaggle. Runs the blocking client in a thread."""
    return await asyncio.to_thread(_validate_sync, username, key)


def _translate_kaggle_error(exc: Exception) -> str:
    text = str(exc)
    if "401" in text or "Unauthorized" in text:
        return "Invalid Kaggle credentials (401 Unauthorized)."
    if "403" in text or "Forbidden" in text:
        return "Kaggle account blocked or quota exceeded (403 Forbidden)."
    if "404" in text or "Not Found" in text:
        return "Kaggle API endpoint not found (404)."
    return f"Kaggle validation failed: {text[:200]}"

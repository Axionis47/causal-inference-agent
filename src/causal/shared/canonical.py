"""Canonical JSON serialization and SHA-256 content hashing (SYSTEM-CONTRACT 3, D-004)."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Final

__all__ = ["CanonicalizationError", "canonical_bytes", "content_hash"]

NON_FINITE_NUMBER: Final = "non_finite_number"
UNSUPPORTED_TYPE: Final = "unsupported_type"


class CanonicalizationError(ValueError):
    """A payload cannot be canonically serialized. `code` is a stable contract value."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def _unsupported(value: object) -> CanonicalizationError:
    return CanonicalizationError(
        f"unsupported type for canonical serialization: {type(value).__name__}",
        UNSUPPORTED_TYPE,
    )


def _normalize(value: object) -> Any:
    """Reject unsupported types and NFC-normalize every string, recursively."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(f"non-finite number: {value!r}", NON_FINITE_NUMBER)
        return value
    if isinstance(value, Mapping):
        return _normalize_mapping(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise _unsupported(value)
    if isinstance(value, Sequence):
        return [_normalize(item) for item in value]
    raise _unsupported(value)


def _normalize_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise _unsupported(key)
        canonical_key = unicodedata.normalize("NFC", key)
        if canonical_key in normalized:
            # Distinct keys collapsing to one canonical key would make the output
            # depend on insertion order, defeating the point of canonicalization.
            raise CanonicalizationError(
                f"duplicate key after NFC normalization: {canonical_key!r}",
                UNSUPPORTED_TYPE,
            )
        normalized[canonical_key] = _normalize(item)
    return normalized


def canonical_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize `payload` to the single canonical UTF-8 JSON byte string."""
    if not isinstance(payload, Mapping):
        raise _unsupported(payload)
    return json.dumps(
        _normalize(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_hash(payload: Mapping[str, object]) -> str:
    """Lowercase 64-character hex SHA-256 of the canonical bytes of `payload`."""
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()

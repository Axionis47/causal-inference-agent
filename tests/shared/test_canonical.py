"""Tests for canonical serialization and hashing (T-002, EV-SYS-001 unit layer)."""

from __future__ import annotations

import datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st

from causal.shared.canonical import CanonicalizationError, canonical_bytes, content_hash


class TestDeterminism:
    def test_key_order_permutations_identical(self) -> None:
        first = {"b": 1, "a": {"y": [1, 2], "x": "v"}}
        second = {"a": {"x": "v", "y": [1, 2]}, "b": 1}
        assert canonical_bytes(first) == canonical_bytes(second)
        assert content_hash(first) == content_hash(second)

    def test_repeated_hashing_is_identical(self) -> None:
        payload = {"k": "value", "n": [1, 2.5, None, True]}
        assert content_hash(payload) == content_hash(payload)

    def test_hash_shape(self) -> None:
        digest = content_hash({"a": 1})
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=8),
            st.one_of(st.integers(), st.text(max_size=8), st.booleans(), st.none()),
            max_size=6,
        )
    )
    def test_insertion_order_invariance(self, payload: dict[str, object]) -> None:
        reordered = dict(reversed(list(payload.items())))
        assert content_hash(payload) == content_hash(reordered)


class TestUnicode:
    def test_nfc_and_nfd_values_normalize_identically(self) -> None:
        assert canonical_bytes({"k": "é"}) == canonical_bytes({"k": "é"})

    def test_nfc_and_nfd_keys_normalize_identically(self) -> None:
        assert canonical_bytes({"é": 1}) == canonical_bytes({"é": 1})

    def test_key_collision_after_nfc_raises_dedicated_code(self) -> None:
        with pytest.raises(CanonicalizationError) as excinfo:
            canonical_bytes({"é": 1, "é": 2})
        assert excinfo.value.code == "duplicate_key_after_normalization"


class TestRejections:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_numbers(self, bad: float) -> None:
        with pytest.raises(CanonicalizationError) as excinfo:
            canonical_bytes({"k": bad})
        assert excinfo.value.code == "non_finite_number"

    @pytest.mark.parametrize(
        "bad",
        [datetime.datetime(2026, 1, 1), b"bytes", bytearray(b"x"), {1, 2}, object()],  # noqa: DTZ001
    )
    def test_unsupported_types(self, bad: object) -> None:
        with pytest.raises(CanonicalizationError) as excinfo:
            canonical_bytes({"k": bad})
        assert excinfo.value.code == "unsupported_type"

    def test_non_string_key_rejected(self) -> None:
        with pytest.raises(CanonicalizationError):
            canonical_bytes({1: "v"})  # type: ignore[dict-item]


class TestShape:
    def test_none_values_survive(self) -> None:
        assert canonical_bytes({"a": None}) == b'{"a":null}'

    def test_compact_sorted_output(self) -> None:
        assert canonical_bytes({"b": 2, "a": [1, "x"]}) == b'{"a":[1,"x"],"b":2}'

    def test_non_ascii_not_escaped(self) -> None:
        assert canonical_bytes({"k": "é"}) == '{"k":"é"}'.encode()

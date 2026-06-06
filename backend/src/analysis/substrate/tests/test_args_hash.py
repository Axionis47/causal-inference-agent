"""args_hash: deterministic, key-order-independent, robust to odd types."""
from __future__ import annotations

import pytest

from src.analysis.substrate.tool_log import args_hash


def test_same_args_produce_same_hash():
    a = {"column": "fertilizer", "threshold": 0.5}
    b = {"column": "fertilizer", "threshold": 0.5}
    assert args_hash(a) == args_hash(b)


def test_key_order_does_not_matter():
    a = {"column": "fertilizer", "threshold": 0.5}
    b = {"threshold": 0.5, "column": "fertilizer"}
    assert args_hash(a) == args_hash(b)


def test_different_values_produce_different_hashes():
    assert args_hash({"x": 1}) != args_hash({"x": 2})


def test_int_vs_float_distinct_when_both_canonical():
    # 1 and 1.0 serialise differently in JSON; the hash reflects that.
    assert args_hash({"x": 1}) != args_hash({"x": 1.0})


def test_nested_dict_with_different_inner_order_hashes_same():
    a = {"outer": {"a": 1, "b": 2}}
    b = {"outer": {"b": 2, "a": 1}}
    assert args_hash(a) == args_hash(b)


def test_list_order_matters():
    # Tool args are positional in semantics; reordering changes meaning.
    assert args_hash({"cols": ["a", "b"]}) != args_hash({"cols": ["b", "a"]})


def test_set_order_normalised():
    # Sets have no semantic order; identical sets hash identically.
    assert args_hash({"tags": {"a", "b"}}) == args_hash({"tags": {"b", "a"}})


def test_hash_is_16_hex_chars():
    h = args_hash({"x": 1})
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_non_json_value_falls_back_to_repr_and_does_not_crash():
    class _Custom:
        def __repr__(self) -> str:
            return "Custom(x=1)"

    h = args_hash({"obj": _Custom()})
    assert isinstance(h, str)
    # Same repr should produce the same hash.
    assert h == args_hash({"obj": _Custom()})


def test_empty_args_have_a_stable_hash():
    assert args_hash({}) == args_hash({})


def test_none_values_distinct_from_missing_keys():
    assert args_hash({"x": None}) != args_hash({})


@pytest.mark.parametrize("val", [True, False, 0, 1, "", "x", None])
def test_primitives_round_trip(val):
    a = {"v": val}
    assert args_hash(a) == args_hash(dict(a))

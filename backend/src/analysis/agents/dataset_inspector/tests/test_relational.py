"""Tests for the relational report builder (dataset_inspector/relational.py).

The builder turns per-file DataProfiles plus full-file candidate keys into a
descriptive RelationalProfile. It assembles nothing; it only classifies the
bundle's shape from schema identity and reports cheap, checkable facts.
"""
from __future__ import annotations

import pandas as pd

from src.analysis.agents.data_profiler import DataProfile
from src.analysis.agents.dataset_inspector.relational import (
    build_relational_profile,
    candidate_keys,
)


def _profile(feature_names: list[str], feature_types: dict[str, str], n: int = 100) -> DataProfile:
    return DataProfile(
        n_samples=n,
        n_features=len(feature_names),
        feature_names=feature_names,
        feature_types=feature_types,
        missing_values={c: 0 for c in feature_names},
        numeric_stats={},
        categorical_stats={},
        potential_confounders=[],
    )


def test_candidate_keys_flags_unique_nonnull_columns():
    df = pd.DataFrame({"id": [1, 2, 3], "grp": [1, 1, 2], "name": ["a", "b", "c"]})
    assert candidate_keys(df) == ["id", "name"]


def test_candidate_keys_excludes_columns_with_nulls_even_if_distinct():
    df = pd.DataFrame({"id": [1, 2, None]})
    assert candidate_keys(df) == []


def test_candidate_keys_on_empty_frame_is_empty():
    assert candidate_keys(pd.DataFrame({"a": []})) == []


def test_candidate_keys_requires_full_file_uniqueness_not_a_sample():
    # The load-bearing invariant: a column unique in the first rows but
    # duplicated later is NOT a key. A sampled uniqueness check would wrongly
    # flag it, and a false key silently licenses a fan-out join downstream.
    df = pd.DataFrame({"id": list(range(1000)), "v": list(range(1000))})
    df.loc[999, "id"] = 0  # duplicate the first id at the very end
    assert "id" not in candidate_keys(df)
    assert candidate_keys(df) == ["v"]


def test_single_file_is_shape_single():
    profiles = {"a.csv": _profile(["x", "y"], {"x": "numeric", "y": "numeric"})}
    rp = build_relational_profile(profiles, {})
    assert rp.shape_hint == "single"
    assert rp.files[0].n_rows == 100
    assert rp.files[0].n_cols == 2
    assert rp.same_schema_groups == []


def test_identical_schema_files_are_shards():
    schema = {"x": "numeric", "y": "numeric"}
    profiles = {
        "part1.csv": _profile(["x", "y"], schema),
        "part2.csv": _profile(["x", "y"], schema),
    }
    rp = build_relational_profile(profiles, {})
    assert rp.shape_hint == "same_schema_shards"
    assert rp.same_schema_groups == [["part1.csv", "part2.csv"]]


def test_distinct_schema_files_are_unrelated():
    profiles = {
        "a.csv": _profile(["x"], {"x": "numeric"}),
        "b.csv": _profile(["y", "z"], {"y": "numeric", "z": "categorical"}),
    }
    rp = build_relational_profile(profiles, {})
    assert rp.shape_hint == "unrelated"
    assert rp.same_schema_groups == []


def test_mixed_schemas_are_multiple_files():
    schema = {"x": "numeric"}
    profiles = {
        "p1.csv": _profile(["x"], schema),
        "p2.csv": _profile(["x"], schema),
        "dim.csv": _profile(["k", "label"], {"k": "numeric", "label": "categorical"}),
    }
    rp = build_relational_profile(profiles, {})
    assert rp.shape_hint == "multiple_files"
    assert ["p1.csv", "p2.csv"] in rp.same_schema_groups


def test_key_candidates_flow_into_the_file_entry():
    profiles = {"a.csv": _profile(["id", "v"], {"id": "numeric", "v": "numeric"})}
    rp = build_relational_profile(profiles, {"a.csv": ["id"]})
    assert rp.files[0].key_candidates == ["id"]

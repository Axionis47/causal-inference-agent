"""Semantic typing and quality findings on crafted frames."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis_v2.agents.profiling.tools import (
    HIGH_MISSINGNESS,
    build_profile_summary,
    likely_time_columns,
    semantic_type,
)


def _crafted_frame() -> pd.DataFrame:
    n = 100
    return pd.DataFrame(
        {
            "Unnamed: 0": range(n),  # id_like by name + monotonic unique ints
            "treat": [0, 1] * 50,  # binary
            "dose": [1, 2, 3, 4, 5] * 20,  # ordinal (5 levels)
            "earnings": np.linspace(0, 60000, n),  # numeric
            "signup_date": pd.date_range("2020-01-01", periods=n),  # datetime
            "region": ["north", "south", "east", "west"] * 25,  # categorical
            "comment": [f"free text row {i}" for i in range(n)],  # text-ish
            "constant_col": ["same"] * n,  # constant
            "mostly_missing": [None] * 30 + list(range(70)),  # 30% missing
        }
    )


def test_semantic_types_cover_every_kind():
    df = _crafted_frame()
    n = len(df)
    expected = {
        "Unnamed: 0": "id_like",
        "treat": "binary",
        "dose": "ordinal",
        "earnings": "numeric",
        "signup_date": "datetime",
        "region": "categorical",
        "comment": "text",
        "constant_col": "binary",  # one level: nunique<=2 non-numeric
    }
    for name, want in expected.items():
        assert semantic_type(name, df[name], n) == want, name


def test_quality_findings_flag_constants_ids_missingness_and_duplicates():
    df = _crafted_frame()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # one duplicate row
    summary = build_profile_summary(df)

    assert summary.n_rows == 101
    assert summary.constant_columns == ["constant_col"]
    assert "Unnamed: 0" not in summary.constant_columns
    assert summary.id_like_columns == ["Unnamed: 0"]
    assert summary.duplicate_row_count == 1
    assert "mostly_missing" in summary.high_missingness_columns
    assert summary.column("mostly_missing").missing_fraction >= HIGH_MISSINGNESS
    assert any("duplicate" in w for w in summary.warnings)


def test_time_columns_found_by_dtype_and_by_name():
    df = _crafted_frame()
    hits = likely_time_columns(df)
    assert "signup_date" in hits  # datetime dtype and name
    df2 = pd.DataFrame({"year": [2001, 2002], "value": [1.0, 2.0]})
    assert likely_time_columns(df2) == ["year"]


def test_empty_frame_profiles_to_zero_rows_without_crashing():
    summary = build_profile_summary(pd.DataFrame({"a": []}))
    assert summary.n_rows == 0
    assert summary.n_columns == 1

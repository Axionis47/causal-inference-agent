"""Protect the assignment schedule and sample behind the fresh live datasets."""

import io

import polars as pl

from tools.model_quality import _case_csv


def test_castle_selection_preserves_outcomes_and_common_transition_gap() -> None:
    rows = [
        {"sid": state, "year": year, "post": int(start is not None and year >= start),
         "l_homicide": None if (state, year) == (1, 2008) else float(state + year)}
        for state, start in ((1, 2007), (2, None), (3, 2006), (4, 2009))
        for year in range(2000, 2011)
    ]
    source = pl.DataFrame(rows)
    actual = pl.read_csv(io.BytesIO(_case_csv(
        {"transform": "castle_2007_cohort_projection_v1"}, source.write_csv().encode())))
    assert actual.height == 20
    assert actual["state_id"].unique().sort().to_list() == [1, 2]
    assert actual["period"].unique().sort().to_list() == list(range(10))
    assert actual["log_homicide"].null_count() == 1  # No outcome-dependent selection.
    assert actual.filter((pl.col("state_id") == 1) & (pl.col("period") == 5))[
        "log_homicide"].item() == 2006.0  # Last untreated wave is calendar 2005.
    assert actual.filter((pl.col("state_id") == 1) & (pl.col("period") >= 6))[
        "treatment"].to_list() == [1, 1, 1, 1]
    assert actual.filter(pl.col("state_id") == 2)["treatment"].sum() == 0


def test_turnout_projection_keeps_randomized_clusters_and_strata() -> None:
    raw = b"rownames,treated,p,strata,n\n1,1,0.2,4,900\n2,0,0.3,4,100\n"
    actual = pl.read_csv(io.BytesIO(_case_csv(
        {"transform": "rock_the_vote_projection_v1"}, raw)))
    assert actual.shape == (2, 4)  # Do not expand cluster counts into individual rows.
    assert actual["turnout_rate"].to_list() == [0.2, 0.3]
    assert actual["randomization_stratum"].to_list() == [4, 4]

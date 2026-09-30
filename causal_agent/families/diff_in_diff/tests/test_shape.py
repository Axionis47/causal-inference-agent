"""The canonical panel: each unit's first treated period read three ways, the columns pyfixest's DID surface needs, the facts and
the guard. No model, no library fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_agent.families.diff_in_diff.lane import shape as SH
from causal_agent.families.diff_in_diff.lane.contracts import Groups, Periods

GROUPS = Groups(column="arm", treated_level="yes", reason="r", cites=["change:1.note"])
PERIODS = Periods(kind="long", time_column="time", first_post="6", reason="r", cites=["change:1.note"])


def one_shot(n_units=10, periods=8) -> pd.DataFrame:
    rows = [{"unit": f"u{u}", "time": t, "arm": "yes" if u < 4 else "no", "y": float(u + t)} for u in range(n_units) for t in range(1, periods + 1)]
    return pd.DataFrame(rows)


def staggered(cohorts=(4, 6, 8), never=3, per=2, periods=10, *, reversal=False) -> pd.DataFrame:
    """A per-row treatment indicator that switches on at each cohort's period and stays on; `never` units stay off."""
    rows = []
    u = 0
    for start in cohorts:
        for _ in range(per):
            for t in range(1, periods + 1):
                on = t >= start and not (reversal and start == cohorts[0] and t == periods)
                rows.append({"unit": f"u{u}", "time": t, "arm": "yes" if on else "no", "y": float(t)})
            u += 1
    for _ in range(never):
        for t in range(1, periods + 1):
            rows.append({"unit": f"u{u}", "time": t, "arm": "no", "y": float(t)})
        u += 1
    return pd.DataFrame(rows)


def test_a_label_and_one_change_period_read_as_one_shot_adoption():
    panel, f = SH.canonical(one_shot(), GROUPS, PERIODS, "y", [], unit_column="unit")
    assert list(panel.columns) == SH.CANON
    assert f.adoption == "one_shot" and f.first_treated_source == "label" and f.cohorts == 1 and f.units_by_cohort == {"6": 4}
    assert f.units_treated == 4 and f.units_control == 6 == f.units_never_treated and f.never_treated_exists and f.balanced and f.clusters == 10
    assert f.periods_pre == 5 and f.periods_post == 3
    treated = panel[panel["unit"] == "u0"].sort_values("time")
    assert treated["time_index"].tolist() == list(range(1, 9)) and treated["cohort"].unique().tolist() == [6]
    assert treated["rel_time"].tolist() == [-5, -4, -3, -2, -1, 0, 1, 2] and treated["treat"].tolist() == [0.0] * 5 + [1.0] * 3
    never = panel[panel["unit"] == "u9"].sort_values("time")
    assert never["cohort"].unique().tolist() == [0] and never["rel_time"].unique().tolist() == [-1] and never["post"].tolist() == [0] * 5 + [1] * 3
    assert "ladder:shape.adoption" in dict(f.lines()) and "one-shot" in dict(f.lines())["ladder:shape.adoption"]


def test_an_indicator_that_switches_on_within_units_reads_as_staggered_adoption():
    panel, f = SH.canonical(staggered(), GROUPS, PERIODS, "y", [], unit_column="unit")
    assert f.adoption == "staggered" and f.first_treated_source == "indicator" and f.cohorts == 3
    assert f.units_by_cohort == {"4": 2, "6": 2, "8": 2} and f.units_treated == 6 and f.units_never_treated == 3 and f.never_treated_exists
    assert f.periods_pre == 3 and f.periods_post == 7  # before and from the earliest cohort
    by_unit = panel.groupby("unit")["cohort"].first()
    assert by_unit["u0"] == 4 and by_unit["u2"] == 6 and by_unit["u4"] == 8 and by_unit["u6"] == 0
    late = panel[panel["unit"] == "u4"].sort_values("time")
    assert late["treat"].tolist() == [0.0] * 7 + [1.0] * 3 and late["rel_time"].tolist() == list(range(-7, 3)) and late["treated"].unique().tolist() == [1]
    assert (panel["post"] == (panel["time_index"] >= 4).astype(int)).all()  # post is from the earliest cohort, for everyone
    assert "3 first-treated periods: 4 (2 units), 6 (2 units), 8 (2 units)" in dict(f.lines())["ladder:shape.adoption"]


def test_a_unit_that_leaves_the_treatment_is_a_typed_stop():
    with pytest.raises(SH.ShapeError, match="units leave the treatment"):
        SH.canonical(staggered(reversal=True), GROUPS, PERIODS, "y", [], unit_column="unit")


def test_an_adoption_column_from_the_pack_sets_the_cohorts():
    df = one_shot(n_units=9)
    adopt = {"u0": 3, "u1": 3, "u2": 5, "u3": 5, "u4": 0, "u5": np.nan, "u6": 99, "u7": 0, "u8": 0}  # 99 lies after the span: never within it
    df["adopt"] = df["unit"].map(adopt)
    df["arm"] = "no"
    panel, f = SH.canonical(df, GROUPS, PERIODS, "y", [], unit_column="unit", cohort_column="adopt")
    assert f.first_treated_source == "cohort_column" and f.adoption == "staggered" and f.units_by_cohort == {"3": 2, "5": 2}
    assert f.units_never_treated == 5 and f.periods_pre == 2
    by_unit = panel.groupby("unit")["cohort"].first()
    assert by_unit["u0"] == 3 and by_unit["u2"] == 5 and by_unit["u4"] == 0 and by_unit["u5"] == 0 and by_unit["u6"] == 0
    assert (
        panel.loc[(panel["unit"] == "u0") & (panel["time"] == 3), "treat"].item() == 1.0
        and panel.loc[(panel["unit"] == "u0") & (panel["time"] == 2), "treat"].item() == 0.0
    )


def test_an_adoption_period_that_cannot_be_read_is_a_typed_stop():
    df = one_shot(n_units=4)
    df["adopt"] = df["unit"].map({"u0": "soon", "u1": 0, "u2": 0, "u3": 0})
    with pytest.raises(SH.ShapeError, match="could not be read"):
        SH.canonical(df, GROUPS, PERIODS, "y", [], unit_column="unit", cohort_column="adopt")


def test_everyone_treated_at_once_leaves_nothing_to_compare_and_a_wide_table_keeps_its_shape():
    df = one_shot(n_units=4)
    df["arm"] = "yes"
    with pytest.raises(SH.ShapeError, match="no unit is untreated"):
        SH.canonical(df, GROUPS, PERIODS, "y", [], unit_column="unit")
    wide = pd.DataFrame({"arm": ["yes", "yes", "no", "no", "no"], "before": [1.0, 2.0, 1.5, 1.0, 2.5], "after": [3.0, 4.0, 2.0, 1.5, 3.0]})
    panel, f = SH.canonical(wide, GROUPS, Periods(kind="wide", before_column="before", after_column="after", reason="r", cites=["change:1.note"]), "y", [])
    assert (
        f.kind == "wide" and f.units_treated == 2 and f.units_control == 3 and f.cohorts == 1 and f.adoption == "one_shot" and f.units_by_cohort == {"after": 2}
    )
    assert (
        sorted(panel["time_index"].unique()) == [1, 2]
        and set(panel.loc[panel["treated"] == 1, "cohort"]) == {2}
        and set(panel.loc[panel["treated"] == 0, "rel_time"]) == {-1}
    )

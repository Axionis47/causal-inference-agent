"""The read-only data tools: what each returns, and the one rule that no tool joins the outcome with the treatment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_agent.lane import tools as TL


def table(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    district = rng.choice(["north", "south"], n)
    school = np.where(district == "north", rng.choice(["n1", "n2"], n), rng.choice(["s1", "s2", "s3"], n))
    lunch = rng.choice(["standard", "free"], n, p=[0.6, 0.4])
    parent = rng.choice(["school", "college", "degree"], n)
    course = np.where((lunch == "standard") & (rng.random(n) < 0.7), "completed", "none")
    reading = rng.normal(60, 10, n) + (lunch == "standard") * 5
    score = reading * 0.5 + rng.normal(0, 5, n) + (course == "completed") * 4
    return pd.DataFrame(
        {
            "Math Score": score,
            "test preparation course": course,
            "lunch": lunch,
            "parental level of education": parent,
            "district": district,
            "school": school,
            "reading score": reading,
        }
    )


@pytest.fixture
def tools() -> TL.Tools:
    df = table()
    return TL.Tools(
        df,
        outcome="Math Score",
        treatment="test preparation course",
        treated=df["test preparation course"] == "completed",
        timing_of={"lunch": "before", "parental level of education": "before", "district": "before", "school": "before", "reading score": "after"},
    )


def test_describe_a_number_and_a_category(tools):
    r = tools.describe("Math Score")
    assert "a number" in r.text and "quartiles" in r.text and r.value == pytest.approx(tools.df["Math Score"].mean())
    r = tools.describe("lunch")
    assert "levels: standard" in r.text and r.value == 2.0


def test_a_key_names_a_column_and_an_unknown_name_is_refused(tools):
    assert tools.column("math_score") == "Math Score" and tools.column("parental_level_of_education") == "parental level of education"
    with pytest.raises(TL.Refused, match="not a column"):
        tools.describe("height")


def test_by_arm_and_the_outcome_rule(tools):
    r = tools.by_arm("lunch")
    assert "of the treated" in r.text and r.value is not None and r.value > 0.2
    r = tools.by_arm("reading score")
    assert "standardised difference" in r.text
    with pytest.raises(TL.Refused, match="join the outcome"):
        tools.by_arm("Math Score")
    tools.frozen = True
    assert "standardised difference" in tools.by_arm("Math Score").text


def test_association_refuses_the_pair_in_either_order_and_allows_the_rest(tools):
    assert tools.association("lunch", "reading score").value is not None
    assert tools.association("test preparation course", "lunch").value is not None  # the treatment against a candidate is allowed
    assert tools.association("Math Score", "reading score").value is not None  # the outcome against a candidate is allowed
    for a, b in (("Math Score", "test preparation course"), ("test_preparation_course", "math_score")):
        with pytest.raises(TL.Refused, match="join the outcome"):
            tools.association(a, b)
    with pytest.raises(TL.Refused, match="same column"):
        tools.association("lunch", "lunch")


def test_redundancy_reads_nesting(tools):
    assert "'school' sits inside 'district'" in tools.redundancy("school", "district").text
    assert "'school' sits inside 'district'" in tools.redundancy("district", "school").text
    assert tools.redundancy("lunch", "parental level of education").value == 0.0
    with pytest.raises(TL.Refused, match="join the outcome"):
        tools.redundancy("Math Score", "test preparation course")


def test_cells_is_the_overlap_table_and_never_splits_the_outcome(tools):
    r = tools.cells(["lunch", "parental level of education"])
    assert r.text.startswith("6 cells over lunch, parental level of education") and "lunch=free, parental level of education=college: treated" in r.text
    assert r.value == float(min(int(x) for x in __import__("re").findall(r"treated (\d+), control (\d+)", r.text) for x in x))
    with pytest.raises(TL.Refused, match="join the outcome"):
        tools.cells(["lunch", "Math Score"])
    with pytest.raises(TL.Refused, match="at most 3"):
        tools.cells(["lunch", "district", "school", "parental level of education"])
    with pytest.raises(TL.Refused, match="at least one"):
        tools.cells([])


def test_no_arms_without_a_treated_mask(tools):
    tools.treated = None
    with pytest.raises(TL.Refused, match="no arms"):
        tools.by_arm("lunch")
    with pytest.raises(TL.Refused, match="no arms"):
        tools.cells(["lunch"])


def test_timing_lists_every_other_column_by_when(tools):
    r = tools.timing()
    assert r.text == "before: lunch, parental level of education, district, school; at: none; after: reading score; unknown: none" and r.value == 0.0
    tools.timing_of = {}
    assert tools.timing().value == 5.0


def test_call_dispatches_by_name_and_refuses_the_rest(tools):
    assert tools.call("describe", {"column": "lunch"}).value == 2.0
    with pytest.raises(TL.Refused, match="no tool named"):
        tools.call("drop_table", {})
    with pytest.raises(TL.Refused, match="could not run"):
        tools.call("describe", {"col": "lunch"})


def test_the_schemas_the_model_sees(tools):
    schemas = tools.schemas()
    assert [s.name for s in schemas] == list(TL.NAMES[:6])  # a plain table offers the six; the shape-aware four need a panel or a score
    by = {s.name: s for s in schemas}
    assert set(by["association"].args) == {"a", "b"} and set(by["cells"].args) == {"columns"} and by["timing"].args == {}
    assert "join" not in by["by_arm"].description  # the rule is code, not a line the model could argue with
    with pytest.raises(TL.Refused, match="no tool named"):
        tools.call("composition", {})


# ------------------------------------------------------------------ the outcome mask, and the tools on a panel


def panel(n_units: int = 20, periods: int = 6, first_post: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for u in range(n_units):
        treated = int(u < n_units // 2)
        for t in range(1, periods + 1):
            if u == 0 and t == periods:  # one unit leaves before the last period
                continue
            y = 10 + u * 0.1 + 0.5 * t + (3.0 if treated and t >= first_post else 0.0) + rng.normal(0, 0.1)
            rows.append(
                {"unit": f"u{u}", "time": t, "treated": treated, "y": y, "spend": 2.0 + 0.2 * t + treated * 0.5, "region": "north" if u % 2 else "south"}
            )
    return pd.DataFrame(rows)


@pytest.fixture
def panel_tools() -> TL.Tools:
    df = panel()
    return TL.Tools(
        df,
        outcome="y",
        treatment="treated",
        treated=df["treated"] == 1,
        allow_outcome_rows=df["time"] < 4,
        allow_outcome_words="the periods before the change",
        aliases={"Sales": "y", "Arm": "treated"},
    )


def test_the_mask_lets_the_outcome_be_seen_before_the_change_and_nowhere_else(panel_tools):
    r = panel_tools.by_arm("Sales")  # the pack's name reaches the panel's column through the alias
    assert "the periods before the change only" in r.text and r.value is not None
    r = panel_tools.by_group_over_time("y")
    assert r.text.startswith("'y' by group and period (mean): 1: treated") and "4:" not in r.text and "the periods before the change only" in r.text
    assert abs(r.value) < 0.3  # before the change the gap barely moves
    panel_tools.allow_outcome_rows = None
    with pytest.raises(TL.Refused, match="join the outcome"):
        panel_tools.by_group_over_time("y")
    panel_tools.allow_outcome_rows = panel_tools.df["time"] > 99
    with pytest.raises(TL.Refused, match="there are none"):
        panel_tools.by_arm("y")


def test_by_group_over_time_on_a_control_reads_every_period(panel_tools):
    r = panel_tools.by_group_over_time("spend")
    assert "6: treated" in r.text and "the gap moved from +0.5 to +0.5" in r.text and r.value == pytest.approx(0.0, abs=1e-9)
    r = panel_tools.by_group_over_time("region")
    assert "(share of 'north')" in r.text or "(share of 'south')" in r.text


def test_composition_counts_units_per_period_and_the_leavers(panel_tools):
    r = panel_tools.composition()
    assert r.text.startswith(
        "units present per period: treated 10, 10, 10, 10, 10, 9; others 10, 10, 10, 10, 10, 10; 0 entered after the first period, 1 left before the last"
    )
    assert r.value == 1.0
    assert panel_tools.names() == TL.NAMES[:8]


# ------------------------------------------------------------------ the tools on a recentred score


def scored(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    x = np.concatenate([rng.uniform(-10, 0, n // 2), rng.uniform(0, 10, n // 2 + 40)])  # more rows just above the line
    age = 40 + 0.5 * x + rng.normal(0, 3, len(x))
    y = 1.0 + 0.2 * x + (x >= 0) * 2.0 + rng.normal(0, 0.5, len(x))
    return pd.DataFrame({"x": x, "y": y, "age": age, "kind": np.where(x >= 0, "a", rng.choice(["a", "b"], len(x)))})


@pytest.fixture
def score_tools() -> TL.Tools:
    df = scored()
    return TL.Tools(df, outcome="y", treatment=None, treated=df["x"] >= 0, aliases={"Income": "x", "Health": "y"})


def test_by_side_near_reads_a_band_and_refuses_the_outcome(score_tools):
    r = score_tools.by_side_near("age", 3.0)
    assert r.text.startswith("within 3 of the line (") and "on the treated side" in r.text and "standardised difference" in r.text
    with pytest.raises(TL.Refused, match="join the outcome"):
        score_tools.by_side_near("Health", 3.0)
    with pytest.raises(TL.Refused, match="positive number"):
        score_tools.by_side_near("age", 0)
    with pytest.raises(TL.Refused, match="no rows on one side"):
        score_tools.by_side_near("age", 1e-9)


def test_score_histogram_shows_both_sides_and_the_bins_at_the_line(score_tools):
    r = score_tools.score_histogram(10)
    assert r.text.startswith("rows per bin of width 2, control side then treated side, the line between them: ") and " | " in r.text
    assert r.value is not None and r.value > 1.0  # more rows just above the line than just below
    assert score_tools.names() == TL.NAMES[:6] + ("by_side_near", "score_histogram")
    with pytest.raises(TL.Refused, match="whole number"):
        score_tools.score_histogram("many")

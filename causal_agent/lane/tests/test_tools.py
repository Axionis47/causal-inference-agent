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
    assert [s.name for s in schemas] == list(TL.NAMES)
    by = {s.name: s for s in schemas}
    assert set(by["association"].args) == {"a", "b"} and set(by["cells"].args) == {"columns"} and by["timing"].args == {}
    assert "join" not in by["by_arm"].description  # the rule is code, not a line the model could argue with

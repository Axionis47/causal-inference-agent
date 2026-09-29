"""The data facts computed before a run: by arm, associations, redundancy and nesting, the outcome, the timing; and the one rule,
that no fact joins the outcome with the treatment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_agent.memory import facts as FX
from causal_agent.memory.records import Column, ColumnFacts, Memory


def _memory(df: pd.DataFrame, when: dict[str, str]) -> Memory:
    m = Memory(name="t")
    for c in df.columns:
        k = FX._key(c)
        m.columns[k] = Column(name=c, key=k, facts=ColumnFacts(distinct=int(df[c].nunique())))
        if c in when:
            m.set(f"col:{k}.when", when[c], status="confirmed", source="user:turn:1")
    m.set("claim:assignment.kind", "own_choice", status="confirmed", source="user:turn:1")
    m.set("claim:assignment.treatment_column", "course", status="confirmed", source="user:turn:1")
    m.set("claim:assignment.treated_level", "yes", status="confirmed", source="user:turn:1")
    return m


def _frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    school = rng.integers(0, 4, n)
    district = school // 2  # a school sits inside a district
    income = rng.normal(50, 10, n) + 5 * school
    lunch = np.where(income < 48, "free", "standard")
    course = np.where(rng.random(n) < 0.3 + 0.2 * (lunch == "free"), "yes", "no")
    score = 60 + 0.4 * income + 5 * (course == "yes") + rng.normal(0, 5, n)
    return pd.DataFrame({"school": school, "district": district, "income": income, "lunch": lunch, "course": course, "score": score})


def test_facts_cover_arms_associations_redundancy_outcome_and_timing_and_never_join_outcome_with_treatment():
    df = _frame()
    m = _memory(df, {"school": "before", "district": "before", "income": "before", "lunch": "before", "course": "at", "score": "after"})
    out = FX.facts(df, m, m.to_claims(), outcome="score", treatment="course", columns=list(df.columns))
    names = {p.name for p in out}
    assert {"by_arm.lunch", "by_arm.income", "with_outcome.income", "redundancy.school~district", "timing"} <= names
    by = {p.name: p for p in out}
    assert "free" in by["by_arm.lunch"].detail and "of the treated" in by["by_arm.lunch"].detail and 0 < by["by_arm.lunch"].value <= 1
    assert "standardised difference" in by["by_arm.income"].detail
    assert "sits inside" in by["redundancy.school~district"].detail and "'school' sits inside 'district'" in by["redundancy.school~district"].detail
    assert by["with_outcome.income"].value > 0.5 and "correlation 0." in by["with_outcome.income"].detail
    assert by["timing"].detail == "before: school, district, income, lunch; at: course; after: score; unknown: none"
    assert all(p.family == "data" and p.passed is None and p.address.startswith("probe:data.") for p in out)
    # the rule: nothing pairs the outcome with the treatment, and neither is a candidate
    assert not any("score" in p.name and "course" in p.name for p in out)
    assert "by_arm.score" not in names and "by_arm.course" not in names and "with_outcome.course" not in names


def test_association_picks_the_measure_by_the_kinds_and_redundancy_reads_both_ways():
    df = _frame()
    v, how = FX.association(df["income"], df["score"])
    assert how == "correlation" and v > 0.5
    v, how = FX.association(df["lunch"], df["course"])
    assert how == "Cramér's V" and 0 < v < 1
    v, how = FX.association(df["lunch"], df["income"])
    assert how == "correlation ratio" and 0.5 < v < 1
    assert FX.redundancy(df["school"], df["district"]) == "a in b" and FX.redundancy(df["district"], df["school"]) == "b in a"
    assert FX.redundancy(df["school"], df["school"].map({0: "a", 1: "b", 2: "c", 3: "d"})) == "same"
    assert FX.redundancy(df["lunch"], df["course"]) is None


def test_without_a_treated_mask_there_are_no_arm_facts_and_quiet_pairs_are_listed_once():
    df = _frame()
    m = _memory(df, {})
    m.set("claim:assignment.treatment_column", None, status="empty", source=None)
    out = FX.facts(df, m, m.to_claims(), outcome="score", treatment="course", columns=list(df.columns))
    names = [p.name for p in out]
    assert not any(n.startswith("by_arm.") for n in names) and names.count("assoc.quiet") <= 1 and "timing" in names

"""The table a lane starts from: the filter and the window by code, pack-named columns loaded, declines never stops."""

from __future__ import annotations

import pandas as pd
import pytest

from causal_agent.common.contracts import DidDesign, Scope
from causal_agent.desk.handoff import forced
from causal_agent.families.adjustment.design import AdjustmentDesign
from causal_agent.lane import intake
from causal_agent.memory import store


def students(scope: Scope | None = None):
    return forced(
        "students3",
        "q",
        "adjustment",
        "math score",
        "test preparation course",
        ["lunch", "gender"],
        scope=scope,
        memory=store.migrate("students3", write=False),
    )


def cigar(scope: Scope | None = None):
    return forced("cigar", "q", "diff_in_diff", "sales", "state", ["state", "year", "sales", "price"], scope=scope, memory=store.migrate("cigar", write=False))


# ------------------------------------------------------------------ the filter grammar


def test_filter_grammar_on_levels_numbers_and_lists():
    df = pd.DataFrame({"lunch": ["standard", "free/reduced", "standard"], "age": [17, 18, 19], "state": [5, 6, 7]})
    kept, d, facts = intake.apply_filter(df, "lunch == standard")
    assert d is None and list(kept.index) == [0, 2] and facts == {"filter": "lunch == standard", "rows_before": 3, "rows_after": 2}
    kept, d, _ = intake.apply_filter(df, "age >= 18 and state in [5, 6]")
    assert d is None and list(kept.index) == [1]
    kept, d, _ = intake.apply_filter(df, "lunch != 'standard'")
    assert d is None and list(kept.index) == [1]
    kept, d, _ = intake.apply_filter(df, "age < 19")
    assert list(kept.index) == [0, 1]


def test_a_filter_the_code_cannot_read_is_a_decline_and_keeps_every_row():
    df = pd.DataFrame({"lunch": ["standard", "free/reduced"]})
    kept, d, facts = intake.apply_filter(df, "students at the school")
    assert (
        len(kept) == 2
        and d is not None
        and d.check == "intake.filter_unparsed"
        and d.about == "scope.population_filter"
        and d.pack_value == "students at the school"
        and facts == {}
    )
    kept, d, _ = intake.apply_filter(df, "grade == 12")
    assert len(kept) == 2 and d.check == "intake.filter_unknown_column"
    df2 = pd.DataFrame({"age": [17, 18]})
    kept, d, _ = intake.apply_filter(df2, "age == old")
    assert len(kept) == 2 and d.check == "intake.filter_value_type"
    assert d.address == "decline:load.scope_population_filter"


# ------------------------------------------------------------------ the window grammar


def test_window_on_numbers_and_dates():
    df = pd.DataFrame({"year": [70, 80, 89, 92], "y": [1, 2, 3, 4]})
    for text, want in [
        ("from 80 to 89", [80, 89]),
        ("80..89", [80, 89]),
        ("80 to 92", [80, 89, 92]),
        (">= 89", [89, 92]),
        ("after 80", [89, 92]),
        ("before 89", [70, 80]),
        ("until 80", [70, 80]),
        ("since 89", [89, 92]),
    ]:
        kept, d, facts = intake.apply_window(df, text, "year")
        assert d is None and list(kept["year"]) == want, text
        assert facts["time_column"] == "year" and facts["rows_after"] == len(want)
    dd = pd.DataFrame({"day": ["2020-01-01", "2020-06-01", "2021-01-01"], "y": [1, 2, 3]})
    kept, d, _ = intake.apply_window(dd, "from 2020-03-01 to 2020-12-31", "day")
    assert d is None and list(kept["y"]) == [2]


def test_a_window_without_a_time_column_or_in_prose_is_a_decline():
    df = pd.DataFrame({"year": [70, 80], "y": [1, 2]})
    kept, d, _ = intake.apply_window(df, "from 80 to 89", None)
    assert len(kept) == 2 and d.check == "intake.window_no_time_column"
    kept, d, _ = intake.apply_window(df, "the years around the tax", "year")
    assert len(kept) == 2 and d.check == "intake.window_unparsed"
    kept, d, _ = intake.apply_window(df, "from eighty to ninety", "year")
    assert len(kept) == 2 and d.check == "intake.window_value_type"


# ------------------------------------------------------------------ load


def test_load_applies_the_scope_and_records_the_facts(tmp_path):
    h = students(Scope(population_filter="gender == female"))
    it = intake.load(h, "test")
    assert it.declines == [] and it.facts["rows_before"] == 1000 and it.facts["rows_after"] == 518
    assert set(it.table["gender"]) == {"female"} and it.table_path.exists() and it.table_path.parent == it.run_dir
    assert it.columns["test_preparation_course"] == "test preparation course"
    h = cigar(Scope(window="from 80 to 92"))
    it = intake.load(h, "test")
    assert it.declines == [] and it.facts["time_column"] == "year" and it.table["year"].min() == 80 and it.table["year"].max() == 92


def test_load_takes_every_column_the_design_block_names_and_declines_a_missing_one():
    h = students()
    assert "lunch" in intake.wanted_columns(h) and "reading_score" not in intake.wanted_columns(h)
    h.design = AdjustmentDesign(adjustment_candidates=["parental level of education"], mediator="reading score", instrument="offer_rank")
    wanted = intake.wanted_columns(h)
    assert "parental_level_of_education" in wanted and "reading_score" in wanted and "offer_rank" in wanted
    it = intake.load(h, "test")
    assert "reading_score" in it.table.columns and "parental_level_of_education" in it.table.columns and "offer_rank" not in it.table.columns
    assert [d.check for d in it.declines] == ["intake.column_missing"] and it.declines[0].about == "col:offer_rank"


def test_a_prose_filter_is_a_decline_not_a_stop_and_a_missing_outcome_is_a_stop():
    h = students(Scope(population_filter="students at the school"))
    it = intake.load(h, "test")
    assert len(it.table) == 1000 and [d.check for d in it.declines] == ["intake.filter_unparsed"]
    h = students()
    h.outcome = "final grade"
    with pytest.raises(intake.IntakeStop) as e:
        intake.load(h, "test")
    assert e.value.feasibility.stage == "load" and "final_grade" in e.value.feasibility.facts[0]


def test_time_key_comes_from_the_panel_block_then_the_change_then_the_entry():
    h = cigar()
    assert intake.time_key(h, {"time": "year"}) == "year"
    h.design = DidDesign(time="period")
    assert intake.time_key(h) == "period"
    h.design = None
    h.change = {"date_column": "when"}
    assert intake.time_key(h) == "when"
    h.change = {}
    assert intake.time_key(h, {}) is None


def test_a_scope_written_as_words_for_nothing_is_not_a_decline():
    df = pd.DataFrame({"year": [70, 80], "lunch": ["a", "b"]})
    for text in (None, "", "null", "None", "n/a", "all students", "every row", "whole", "no filter"):
        kept, d, facts = intake.apply_filter(df, text)
        assert d is None and len(kept) == 2 and facts == {}, text
    for text in ("null", "none", "all years", "no window"):
        kept, d, _ = intake.apply_window(df, text, "year")
        assert d is None and len(kept) == 2, text
    assert intake.blank("gender == female") is False and intake.blank("from 80 to 89") is False and intake.blank("all") is True

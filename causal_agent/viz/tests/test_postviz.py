"""The figures a run leaves behind, from its artifacts alone."""

from __future__ import annotations

from causal_agent.desk.contracts import RunRecord
from causal_agent.viz import postviz


def _rec(**kw) -> RunRecord:
    sr = {"status": "done", "estimates": [{"contrast": "completed_vs_none", "method": "linear_regression", "value": 5.6, "ci_low": 3.7, "ci_high": 7.5, "secondary": False, "error": None},
                                          {"contrast": "completed_vs_none", "method": "psw", "value": 5.9, "ci_low": 3.9, "ci_high": 7.9, "secondary": True, "error": None}],
          "refutations": [{"contrast": "completed_vs_none", "refuter": "placebo_treatment_refuter", "kind": "falsification", "new_effect": 0.1, "passed": True},
                          {"contrast": "completed_vs_none", "refuter": "data_subset_refuter", "kind": "falsification", "new_effect": 5.4, "passed": True},
                          {"contrast": "completed_vs_none", "refuter": "add_unobserved_common_cause", "kind": "sensitivity", "range_low": 4.1, "range_high": 6.8, "passed": None}]}
    sr.update(kw)
    return RunRecord(index=1, dataset="d", question="q", family="adjustment", specialist="dowhy", status="done", effect=5.6, specialist_result=sr)


def test_effect_against_refutations():
    f = postviz.effect_and_refutations(_rec())
    assert f.kind == "interval" and f.id == "effect_completed_vs_none" and f.address == "figure:effect_completed_vs_none"
    s = f.series[0]
    assert s.x[0] == "linear_regression" and s.y[0] == 5.6 and s.lo[0] == 3.7 and s.hi[0] == 7.5
    assert s.x[1] == "placebo_treatment_refuter ✓" and s.y[1] == 0.1 and s.lo[1] is None
    assert s.x[3] == "add_unobserved_common_cause" and s.lo[3] == 4.1 and s.hi[3] == 6.8
    assert f.marks[0].kind == "hline" and "every falsification passed" in f.note
    assert "estimate:completed_vs_none.value" in f.draws_on and "refute:completed_vs_none.placebo_treatment_refuter.new_effect" in f.draws_on
    assert "figure:effect_completed_vs_none.effect.1" in f.spec_addresses() if hasattr(f, "spec_addresses") else "figure:effect_completed_vs_none.effect.1" in f.addresses()


def test_dynamic_effects_only_when_the_lane_left_them():
    assert postviz.dynamic_effects(_rec()) is None
    rec = _rec(dynamic={"-2": [0.1, -0.5, 0.7], "-1": [0.0, -0.4, 0.4], "0": [2.1, 1.2, 3.0], "1": [2.4, 1.5, 3.3]})
    rec.specialist = "pyfixest"
    f = postviz.dynamic_effects(rec)
    assert f is not None and f.series[0].x == [-2.0, -1.0, 0.0, 1.0] and f.series[0].y[2] == 2.1 and f.series[0].lo[2] == 1.2
    assert any(m.kind == "vline" for m in f.marks)
    assert [x.id for x in postviz.figures(rec)] == ["effect_completed_vs_none", "dynamic_effects"]


def test_no_estimate_no_figure():
    rec = _rec(estimates=[], refutations=[])
    assert postviz.figures(rec) == []


def test_the_builders_over_dicts_are_what_the_record_wrappers_draw():
    from causal_agent.viz.postviz import common

    rec = _rec(dynamic={"-1": [0.0, -0.4, 0.4], "0": [2.1, 1.2, 3.0]})
    sr = rec.specialist_result
    assert common.effect_and_refutations(sr["estimates"], sr["refutations"], "refute") == postviz.effect_and_refutations(rec)
    assert common.event_study(sr["dynamic"]) == postviz.dynamic_effects(rec)
    f = common.event_study(sr["dynamic"], contrast="a_vs_b", draws_on=["check:a_vs_b.pre_trends"])
    assert f.id == "event_study_a_vs_b" and f.draws_on == ["check:a_vs_b.pre_trends"]
    assert common.effect_and_refutations([], [], "placebo") is None and common.event_study({}) is None

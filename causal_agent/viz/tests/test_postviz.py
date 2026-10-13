"""Fallback plots preserve design ownership and failure of a selected lead."""

from causal_agent.common.contracts import RunRecord
from causal_agent.viz import postviz


def test_each_design_gets_its_own_qualified_plot():
    result = {
        "designs": {"lead": {}, "rival": {}},
        "lead_design_id": "lead",
        "estimates": [
            {"design_id": "lead", "endpoint_id": "total.backdoor", "contrast": "treated", "method": "linear", "error": "fit failed"},
            {"design_id": "rival", "endpoint_id": "total.general_adjustment", "contrast": "treated", "method": "linear", "value": 2.0},
        ],
        "falsifications": [{"design_id": "rival", "contrast": "treated", "refuter": "placebo", "new_effect": 0.1, "passed": None}],
    }
    rec = RunRecord(index=1, dataset="d", question="q", lead_design_id="lead", specialist_result=result)
    figures = postviz.figures(rec)
    assert len(figures) == 1 and figures[0].id == "effect_rival_treated"
    assert "estimate:rival.treated.value" in figures[0].draws_on
    assert "falsification:rival.treated.placebo.new_effect" in figures[0].draws_on
    assert "0 falsifications passed" in figures[0].note


def test_no_successful_effect_means_no_fallback_plot():
    assert postviz.figures(RunRecord(index=1, dataset="d", question="q", specialist_result={"designs": {"d": {}}})) == []

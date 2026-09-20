"""A pack names its family block by kind; the block comes back as the family's own class without the pack naming a family."""

from __future__ import annotations

import pytest

from causal_agent.common.contracts import DESIGNS, AdjustmentDesign, Design, DidDesign, Handoff, RdDesign, parse_design


def test_every_built_family_block_is_registered_by_its_kind():
    assert DESIGNS["adjustment"] is AdjustmentDesign and DESIGNS["diff_in_diff"] is DidDesign and DESIGNS["discontinuity"] is RdDesign


def test_a_dict_becomes_the_registered_block_and_an_unknown_kind_is_refused():
    d = parse_design({"kind": "diff_in_diff", "unit": "state", "time": "year", "controls_allowed": ["price"]})
    assert isinstance(d, DidDesign) and d.columns() == ["state", "year", "price"] and d.time_column() == "year"
    assert parse_design(None) is None and parse_design(d) is d
    with pytest.raises(ValueError, match="no family block is registered for kind 'lottery'"):
        parse_design({"kind": "lottery"})


def test_the_block_survives_a_trip_through_json_with_its_own_fields():
    h = Handoff.model_validate(
        dict(
            family="discontinuity",
            specialist="rdrobust",
            supported_now=True,
            outcome="y",
            treatment=None,
            scope={},
            pack_name="p",
            relevant_columns=[],
            chosen_assumption="a",
            reasons=[],
            design={"kind": "discontinuity", "score": "margin", "cutoff": 0.0, "treated_side": "above", "covariates_allowed": ["age"]},
        )
    )
    assert isinstance(h.design, RdDesign) and h.design.columns() == ["margin", "age"]
    back = Handoff.model_validate_json(h.model_dump_json())
    assert isinstance(back.design, RdDesign) and back.design.cutoff == 0.0 and "margin" in back.render_design()
    assert issubclass(RdDesign, Design)

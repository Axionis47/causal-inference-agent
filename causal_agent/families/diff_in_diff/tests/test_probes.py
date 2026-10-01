"""The family's probes: periods before the change, how many units got it, and the treated group seen before it. No model."""

from __future__ import annotations

import pandas as pd

from causal_agent.families.diff_in_diff.probes import probes
from causal_agent.memory.claims import Claim, ClaimTable

TH = {"probe": {"min_pre_periods": 2}}


def table(**change_extra) -> ClaimTable:
    return ClaimTable(
        claims={
            "grain": Claim(
                kind="grain", key="grain", fields={"row_is": "a unit in a period", "key_columns": ["unit", "time"], "panel": True}, status="confirmed"
            ),
            "change": Claim(
                kind="change",
                key="change",
                fields={"what": "w", "to_whom": "t", "when": "6", "date_column": "time", "period_value": "6", **change_extra},
                status="confirmed",
            ),
            "assignment": Claim(
                kind="assignment",
                key="assignment",
                fields={"kind": "date_by_others", "rule": "r", "treatment_column": "arm", "treated_level": "yes"},
                status="confirmed",
            ),
        }
    )


def panel(treated_units=3, n_units=10, periods=8) -> pd.DataFrame:
    return pd.DataFrame(
        [{"unit": f"u{u}", "time": t, "arm": "yes" if u < treated_units else "no", "y": 1.0} for u in range(n_units) for t in range(1, periods + 1)]
    )


def test_the_probes_count_pre_periods_treated_units_and_the_treated_group_before_the_change():
    out = {p.name: p for p in probes(panel(), table(), TH)}
    assert set(out) == {"pre_periods", "treated_units", "treated_before"}
    assert out["pre_periods"].value == 5 and out["pre_periods"].passed
    assert out["treated_units"].value == 3 and out["treated_units"].passed and out["treated_units"].detail == "3 distinct units got the change"
    assert out["treated_before"].value == 15 and out["treated_before"].passed
    assert out["treated_units"].address == "probe:diff_in_diff.treated_units"


def test_no_treated_unit_fails_the_probe_and_rows_stand_for_units_without_a_unit_column():
    out = {p.name: p for p in probes(panel(treated_units=0), table(), TH)}
    assert out["treated_units"].value == 0 and out["treated_units"].passed is False
    t = table()
    t.claims["grain"].fields["key_columns"] = []
    out = {p.name: p for p in probes(panel(), t, TH)}
    assert out["treated_units"].value == 24 and "rows stand for units" in out["treated_units"].detail


def test_the_adoption_column_rides_onto_the_block_as_the_cohort_column():
    from causal_agent.common.contracts import Scope
    from causal_agent.families.base import BlockInputs
    from causal_agent.families.diff_in_diff.handoff import design_block

    def block(**extra):
        t = table(**extra)
        claims = {k: c.fields for k, c in t.claims.items()}
        return design_block(BlockInputs(briefs=[], claims=claims, beliefs={}, probes=[], entry={}, scope=Scope(target="on_treated"), treatment="arm"))

    d = block()
    assert d.treated_group == {"column": "arm", "level": "yes"} and d.columns() == ["unit", "time", "arm", "unit"] and d.staggered is None
    d = block(adoption_column="first_period")
    assert d.staggered is True
    assert d.treated_group == {"column": "arm", "level": "yes", "cohort_column": "first_period"} and "first_period" in d.columns()
    assert "first treated period per unit in first_period" in d.render()

"""Every falsification and sensitivity in the catalogue, on the frozen design: each passes on a panel where the assumption holds
and fails on one doctored to break what it tests. No model; pyfixest fits on a small panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_agent.common.contracts import Estimate
from causal_agent.families.diff_in_diff.lane import adapter
from causal_agent.families.diff_in_diff.lane.knowledge import estimator, load_placebos, placebo

FORMULA = "y ~ treat + x1 | unit+time"
VCOV = {"CRV1": "unit"}


def panel(*, n_units=30, treated=12, periods=10, change=6, effect=3.0, seed=4, doctor: str | None = None) -> pd.DataFrame:
    """Thirty units over ten periods; twelve get the change from period 6; a time-varying column x1 the change does not move; a
    group above the unit. `doctor` breaks one assumption: anticipation (the outcome moves one period early), one_unit (one treated
    unit carries the whole effect), outcome_moved (x1 moves with the change), pre_kink (the treated group's path bends in the pre
    period)."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        is_t = u < treated
        ue = rng.normal(0, 1)
        for t in range(1, periods + 1):
            treat = float(is_t and t >= change)
            x1 = 0.5 * t + rng.normal(0, 0.3)
            y = 10 + ue + 0.3 * t + 0.8 * x1 + rng.normal(0, 0.3)
            if doctor == "one_unit":
                y += treat * (effect * treated if u == 0 else 0.0)
            else:
                y += treat * effect
            if doctor == "anticipation" and is_t and t == change - 1:
                y += effect
            if doctor == "outcome_moved":
                x1 += treat * 2.0
            if doctor == "pre_kink" and is_t and t < change:
                y += 1.5 * (change - t)
            rows.append(
                {
                    "y": y,
                    "unit": f"u{u}",
                    "time": t,
                    "time_index": t,
                    "treated": int(is_t),
                    "post": int(t >= change),
                    "treat": treat,
                    "rel_time": t - change if is_t else -1,
                    "cohort": change if is_t else 0,
                    "x1": x1,
                    "cluster": f"g{u % 5}",
                }
            )
    return pd.DataFrame(rows)


def primary_of(p: pd.DataFrame) -> Estimate:
    m = adapter.fit(FORMULA, p, VCOV)
    lo, hi = (float(x) for x in m.confint().loc["treat"].to_numpy())
    return Estimate(contrast="yes_vs_no", method="twfe_static", value=float(m.coef()["treat"]), ci_low=lo, ci_high=hi, n=len(p))


def run(name: str, p: pd.DataFrame, **kw):
    entry = placebo(name)
    if name == "placebo_group":
        entry = entry.model_copy(update={"params": {**entry.params, "draws": 40}})
    return adapter.falsify(entry, FORMULA, p, VCOV, primary_of(p), ["x1"], "yes_vs_no", **kw)


def test_the_pass_rule_reads_each_declared_condition():
    prim = Estimate(contrast="c", method="m", value=2.0, ci_low=1.0, ci_high=3.0, n=10)
    null = Estimate(contrast="c", method="m", value=0.2, ci_low=-1.0, ci_high=1.4, n=10)
    assert adapter.passes({"interval_covers_zero": True}, prim, new=0.1, lo=-0.5, hi=0.7, p=None) is True
    assert adapter.passes({"interval_covers_zero": True}, prim, new=1.0, lo=0.5, hi=1.5, p=None) is False
    assert adapter.passes({"interval_overlaps_primary": True, "sign_stable_when_primary_excludes_zero": True}, prim, new=2.5, lo=1.5, hi=3.5, p=None) is True
    assert adapter.passes({"interval_overlaps_primary": True, "sign_stable_when_primary_excludes_zero": True}, prim, new=-0.5, lo=-1.5, hi=0.5, p=None) is False
    assert adapter.passes({"interval_overlaps_primary": True, "sign_stable_when_primary_excludes_zero": True}, null, new=-0.5, lo=-1.5, hi=0.5, p=None) is True
    assert adapter.passes({"p_value_lt": 0.05, "informative_only_when_primary_excludes_zero": True}, prim, new=0.0, lo=None, hi=None, p=0.01) is True
    assert adapter.passes({"p_value_lt": 0.05, "informative_only_when_primary_excludes_zero": True}, prim, new=0.0, lo=None, hi=None, p=0.3) is False
    assert adapter.passes({"p_value_lt": 0.05, "informative_only_when_primary_excludes_zero": True}, null, new=0.0, lo=None, hi=None, p=0.3) is None
    assert adapter.passes({"interval_covers_zero": True}, prim, new=None, lo=None, hi=None, p=None) is None


def test_every_falsification_passes_where_the_assumption_holds():
    p = panel()
    out = {
        name: run(name, p, outcome_columns=["x1"])[0] for name in ("placebo_group", "placebo_timing", "placebo_outcome", "leave_one_out", "anticipation_shift")
    }
    for name, r in out.items():
        assert r.kind == "falsification" and r.passed is True, (name, r.detail)
    assert out["placebo_group"].p_value == 0.0 and "40 reassignments" in out["placebo_group"].detail
    assert "fake change at 3" in out["placebo_timing"].detail
    assert out["placebo_outcome"].detail.startswith("x1: effect")
    assert "12 refits" in out["leave_one_out"].detail and "12 of 12 keep the sign" in out["leave_one_out"].detail
    assert "1 period(s) before the real one" in out["anticipation_shift"].detail


def test_the_refits_carry_the_designs_controls_and_rows():
    """A placebo tests the design as frozen: the formula keeps the controls (the stepwise operator flattened), and the leave-one-out
    points are one per treated unit."""
    assert adapter._stepless("y ~ treat + csw0(x1, x2) | unit+time") == "y ~ treat + x1 + x2 | unit+time"
    assert adapter._stepless("y ~ treat | unit+time") == "y ~ treat | unit+time"
    r, extra = run("leave_one_out", panel())
    assert sorted(pt["label"] for pt in extra["points"]) == sorted(f"without u{i}" for i in range(12)) and all(
        pt["lo"] < pt["value"] < pt["hi"] for pt in extra["points"]
    )
    assert r.range_low <= min(pt["value"] for pt in extra["points"]) and r.range_high >= max(pt["value"] for pt in extra["points"])


def test_an_early_move_fails_the_anticipation_shift():
    r, _ = run("anticipation_shift", panel(doctor="anticipation"))
    assert r.passed is False and "outcome moved before the change" in r.detail
    # the design that left that period out (its rows are gone from the estimate) is tested one further back, where nothing moved
    p = panel(doctor="anticipation")
    r2, _ = run("anticipation_shift", p[~((p["treated"] == 1) & (p["rel_time"] == -1))], excluded=1)
    assert r2.passed is True and "2 period(s) before" in r2.detail


def test_one_unit_carrying_the_effect_fails_the_leave_one_out():
    """Under row-robust inference one unit can drive both the point and a tight interval, and the refit without it sits outside;
    under unit-clustered inference that unit's leverage is already in the primary's interval, so the rule reads pass."""
    p = panel(doctor="one_unit")
    m = adapter.fit(FORMULA, p, "hetero")
    lo, hi = (float(x) for x in m.confint().loc["treat"].to_numpy())
    prim = Estimate(contrast="yes_vs_no", method="twfe_static", value=float(m.coef()["treat"]), ci_low=lo, ci_high=hi, n=len(p))
    r, _ = adapter.falsify(placebo("leave_one_out"), FORMULA, p, "hetero", prim, ["x1"], "yes_vs_no")
    assert r.passed is False and "one unit carries the conclusion" in r.detail and "without u0" in r.detail and "11 of 12 keep the sign" in r.detail
    clustered, _ = run("leave_one_out", p)
    assert clustered.passed is True and primary_of(p).ci_low < 0


def test_a_column_the_change_moved_fails_the_placebo_outcome():
    r, _ = run("placebo_outcome", panel(doctor="outcome_moved"), outcome_columns=["x1"])
    assert r.passed is False and "a column the change could not have moved" in r.detail
    r2, _ = run("placebo_outcome", panel(), outcome_columns=[])
    assert r2.passed is None and "no column" in r2.detail


def test_a_bend_in_the_pre_period_fails_the_placebo_timing():
    r, _ = run("placebo_timing", panel(doctor="pre_kink"))
    assert r.passed is False and "pre-period 'effect'" in r.detail


def test_the_sensitivities_report_a_range_and_no_verdict():
    p = panel()
    prim = primary_of(p)
    for name in ("unit_trends", "group_time_fe"):
        r, _ = run(name, p)
        assert r.kind == "sensitivity" and r.passed is None and r.range_low < r.new_effect < r.range_high, name
        assert abs(r.new_effect - prim.value) < 0.5, (name, r.new_effect, prim.value)
    naive, _ = adapter.falsify(placebo("twfe_naive"), "", p, VCOV, prim, ["x1"], "yes_vs_no")
    assert naive.kind == "sensitivity" and abs(naive.new_effect - prim.value) < 1e-6  # on one cohort the naive fit is the design


def test_no_yaml_key_is_dead_and_every_applies_when_key_is_a_fact():
    """Every key of every placebo entry is a field of its model; every applies_when key names a fact the freeze computes; every
    field the model declares is read somewhere."""
    from pathlib import Path

    import yaml

    from causal_agent.families.diff_in_diff.lane.knowledge import PlaceboEntry
    from causal_agent.lane.knowledge import fact_names

    here = Path(__file__).resolve().parents[1] / "lane"
    raw = yaml.safe_load((here / "knowledge" / "placebos.yaml").read_text())
    fields = set(PlaceboEntry.model_fields) - {"name"}
    code = "".join((here / f).read_text() for f in ("nodes.py", "adapter.py")) + (here / "knowledge" / "__init__.py").read_text()
    nodes = (here / "nodes.py").read_text()
    for name, entry in raw.items():
        assert not set(entry) - fields, f"{name} has keys no field reads: {sorted(set(entry) - fields)}"
        for fact in fact_names(entry["applies_when"]):
            assert f'"{fact}"' in nodes, f"{name} applies by {fact!r}, which the freeze never computes"
    for field in fields:
        assert f".{field}" in code, f"{field!r} is declared on PlaceboEntry but nothing reads it"
    assert {e.name for e in load_placebos()} == set(raw) and estimator("twfe_static").engine == "feols"


@pytest.mark.parametrize("name", [e.name for e in load_placebos()])
def test_each_entry_has_words_a_source_and_a_rule_matching_its_kind(name):
    e = placebo(name)
    assert e.in_words and e.source
    assert bool(e.pass_when) == (e.kind == "falsification")

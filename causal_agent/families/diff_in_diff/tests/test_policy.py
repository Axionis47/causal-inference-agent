"""Deterministic admission, outcome restrictions, identities and frozen targets."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from causal_agent.common.contracts import Checks, Contrast
from causal_agent.families.diff_in_diff.lane.contracts import Design, Groups, Periods, ShapeFacts
from causal_agent.families.diff_in_diff.lane.engine.diagnostics import holm, pre_trends, regular_support
from causal_agent.families.diff_in_diff.lane.engine.estimators import run
from causal_agent.families.diff_in_diff.lane.knowledge import ENDPOINTS, SPEC, load_estimators
from causal_agent.families.diff_in_diff.lane.targets import projection_weights, target_groups
from causal_agent.lane.endpoints import verify_catalogue


def panel(*, never=True, cohorts=(4, 7), units=18, periods=10):
    rng = np.random.default_rng(13)
    records = []
    for u in range(units * (len(cohorts) + int(never))):
        g = cohorts[u // units] if u // units < len(cohorts) else 0
        fixed = rng.normal()
        for t in range(1, periods + 1):
            on = int(g > 0 and t >= g)
            records.append(
                dict(
                    unit=str(u),
                    cluster=str(u),
                    time=t,
                    time_index=t,
                    cohort=g,
                    rel_time=t - g if g else -1,
                    treat=float(on),
                    treated=int(g > 0),
                    post=int(t >= min(cohorts)),
                    y=fixed + 0.3 * t + 2 * on + rng.normal(scale=0.2),
                )
            )
    p = pd.DataFrame(records)
    p["row_id"] = p.index
    return p


def design(p, engine, estimator, mode="pooled", *, endpoint="untreated_imputation_att"):
    return Design(
        id="dtest",
        endpoint_id=endpoint,
        context_id="context",
        ledger_revision=1,
        identification_certificate={"comparison": "declared test comparison", "assumptions": ["parallel trends"]},
        target=dict(aggregation=mode, horizons=[0, 1, 2], cohorts=["4", "7"], population="treated", outcome="y"),
        contrast=Contrast(treated="yes", control="no", reason="declared", cites=[]),
        groups=Groups(column="treated", treated_level="1", reason="declared", cites=[]),
        periods=Periods(kind="long", time_column="time", first_post="4", reason="declared", cites=[]),
        shape=ShapeFacts(kind="long", rows=len(p), units_treated=36, units_control=18, periods_pre=3, periods_post=7, cohorts=2),
        checks=Checks(),
        estimator=estimator,
        engine=engine,
        formula="y ~ treat | unit+time_index",
        first_stage="~ 0 | unit+time_index",
        second_stage="~ treat" if estimator == "did2s" else "~ i(rel_time, ref=-1)" if estimator == "did2s_dynamic" else "~ cohort_treat_4 + cohort_treat_7",
        inference="native_crv1",
        cluster_column="cluster",
        vcov={"CRV1": "cluster"},
        panel_path="not-read-by-engine",
        row_ids=tuple(p.index),
    )


def test_declared_endpoints_have_complete_executable_trajectories():
    assert len(ENDPOINTS.endpoints) == 6
    assert verify_catalogue(ENDPOINTS, SPEC, {e.name for e in load_estimators()}) == []
    assert all("freeze_design" in e.trajectory for e in ENDPOINTS.endpoints)


def test_lead_ranking_is_order_independent_and_effect_blind():
    from causal_agent.common.contracts import Handoff
    from causal_agent.families.diff_in_diff.design import DidDesign
    from causal_agent.families.diff_in_diff.lane.reason.selection import select_lead
    from causal_agent.lane.ledger import Entry, Evidence, Ledger

    p = panel()
    h = Handoff.model_construct(design=DidDesign(aggregation="horizon_profile", horizons=[0, 1, 2], target_weighting="treated_cell_shares"))
    ledger = Ledger(
        entries={
            "rel:parallel": Entry(
                id="rel:parallel",
                kind="parallel_trends",
                value=True,
                status="supported",
                evidence=[Evidence(source="pack", address="claim:trend_continues.believed")],
                reason="sourced conditional premise",
            )
        }
    )
    target = {
        "aggregation": "horizon_profile",
        "population": "treated",
        "horizons": [0, 1, 2],
        "cohorts": [],
        "support": "admitted_window",
        "weighting": "treated_cell_shares",
    }
    target["components"] = {label: {"included_units": sorted(set(p.loc[mask, "unit"]))} for label, mask in target_groups(p, target)}
    designs = []
    for identifier, engine, method, endpoint in [
        ("za", "did2s", "did2s_dynamic", "untreated_imputation_att"),
        ("zb", "did2s", "did2s_dynamic", "untreated_imputation_att"),
        ("aa", "lpdid", "lpdid", "clean_projection_att"),
    ]:
        raw = design(p, engine, method, "horizon_profile", endpoint=endpoint).model_dump(exclude={"integrity"})
        raw.update(id=identifier, target=target, assumptions=["rel:parallel"])
        if engine == "lpdid":
            raw["target"] = {**target, "weighting": "clean_adopter_projection"}
        designs.append(Design.model_validate(raw))
    first = select_lead(h, {d.id: d for d in designs}, ledger, p)
    assert first["chosen_id"] == "za"
    assert first["effect_blind"]
    assert next(item for item in first["ranking"] if item["design_id"] == "aa")["target_penalties"]["weighting"] == 1
    for order in permutations(designs):
        assert select_lead(h, {d.id: d for d in order}, ledger, p) == first
    changed = p.copy()
    changed.y = np.arange(len(changed)) * 1e12
    assert select_lead(h, {d.id: d for d in designs}, ledger, changed) == first
    companion_only = select_lead(h, {"aa": designs[-1]}, ledger, p)
    assert companion_only["chosen_id"] is None
    assert not companion_only["requested_target_identified"]
    clean_request = Handoff.model_construct(design=DidDesign(aggregation="horizon_profile", horizons=[0, 1, 2], target_weighting="clean_adopter_projection"))
    assert select_lead(clean_request, {d.id: d for d in designs}, ledger, p)["chosen_id"] == "aa"
    projection_request = Handoff.model_construct(
        design=DidDesign(aggregation="horizon_profile", horizons=[0, 1, 2], target_weighting="linear_response_horizon_projection")
    )
    raw = design(p, "feols", "twfe_dynamic", "horizon_profile", endpoint="common_adoption_projection_effect").model_dump(exclude={"integrity"})
    raw.update(id="projection", target={**target, "weighting": "linear_response_horizon_projection"}, assumptions=["rel:parallel"])
    projection = Design.model_validate(raw)
    assert select_lead(projection_request, {**{d.id: d for d in designs}, projection.id: projection}, ledger, p)["chosen_id"] == "projection"


def test_data_citations_cannot_be_promoted_into_study_accounts():
    from causal_agent.common.contracts import Fact, Handoff
    from causal_agent.families.diff_in_diff.lane.contracts import ComparisonOp, ComparisonReading
    from causal_agent.families.diff_in_diff.lane.reason.ledger import commit
    from causal_agent.lane.episode import EpisodeLog
    from causal_agent.lane.ledger import Evidence, Ledger

    h = Handoff.model_construct(facts={"col:c.when": Fact(value="before", status="confirmed", source="data")})
    op = ComparisonOp(
        kind="parallel_trends",
        subject="comparison",
        value=True,
        status="supported",
        reason="timing is not a causal premise",
        evidence=[Evidence(source="pack", address="rel:context.col:c.when", category="study_account")],
    )
    record = ComparisonReading(notes="test source integrity", operations=[op])
    _, errors = commit(Ledger(), record, "comparison", lambda _a: True, handoff=h, log=EpisodeLog(node="comparison"))
    assert any("causal support needs a study account" in error for error in errors)
    record.operations[0].status = "proposed"
    accepted, errors = commit(Ledger(), record, "comparison", lambda _a: True, handoff=h, log=EpisodeLog(node="comparison"))
    assert not errors
    source = accepted.entries["rel:parallel_trends.comparison"].evidence[0]
    assert source.source == "data" and source.category == "measurement_timing"


def test_covariates_absorbed_by_fixed_effects_do_not_invoke_control_judgement(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from causal_agent.common.contracts import Handoff
    from causal_agent.families.diff_in_diff.lane.reason import nodes

    h = Handoff.model_validate(json.loads((Path(__file__).parents[1] / "evals/handoffs/card_krueger_forced.json").read_text()))
    p = panel()
    p["total_emp_feb"] = p.unit.astype(int)
    path = tmp_path / "panel.csv"
    p.to_csv(path, index=False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an absorbed baseline covariate cannot require model control classification")

    monkeypatch.setattr(nodes, "_judge", forbidden)
    command = nodes.controls({"handoff": h, "panel_path": str(path)})
    assert command.goto == "probe"
    assert command.update["reasoning_history"][0]["operations"] == []


def test_pre_only_diagnostics_are_invariant_to_post_outcomes():
    p = panel(cohorts=(4,))
    first = pre_trends(p, [], "cluster")
    changed = p.copy()
    changed.loc[changed.post == 1, "y"] = np.arange((changed.post == 1).sum()) * 1e8
    assert pre_trends(changed, [], "cluster") == first
    assert set(first["coefficients"]) <= {"time_index::1:treated", "time_index::2:treated"}


def test_projection_weights_do_not_read_outcomes_and_sum_to_one():
    p = panel(cohorts=(4,))
    w = projection_weights(p, [])
    p.y = 10000
    assert projection_weights(p, []) == w
    assert sum(w.values()) == pytest.approx(1.0)


def test_period_gaps_are_not_one_period_projection_transitions():
    p = panel()
    assert regular_support(p)
    assert not regular_support(p.loc[~((p.unit == "0") & (p.time == 3))])
    p.loc[p.time > 5, "time"] += 1
    assert not regular_support(p)


def test_holm_preserves_missing_and_corrects_the_declared_family():
    assert holm([0.01, None, 0.03, 0.2]) == [0.03, None, 0.06, 0.2]


@pytest.mark.parametrize(
    ("engine", "estimator", "mode"),
    [
        ("did2s", "did2s", "pooled"),
        ("did2s", "did2s_dynamic", "horizon_profile"),
        ("did2s", "did2s_cohort", "cohort_profile"),
        ("saturated", "saturated_event_study", "cohort_profile"),
        ("lpdid", "lpdid", "pooled"),
        ("lpdid", "lpdid", "horizon_profile"),
    ],
)
def test_native_engines_honor_explicit_targets(engine, estimator, mode):
    p = panel()
    d = design(p, engine, estimator, mode)
    out = run(d, p)
    assert out and all(e.error is None for e in out), [e.error for e in out]
    assert all(e.value == pytest.approx(2.0, abs=0.3) and e.ci_low is not None and e.design_id == "dtest" for e in out)
    expected = 2 if mode == "cohort_profile" else 3 if mode == "horizon_profile" else 1
    assert len(out) == expected


def test_no_never_prefix_is_a_frozen_scope_not_an_engine_retry():
    p = panel(never=False, cohorts=(4, 7, 10), periods=11)
    prefix = p.loc[p.time_index < 10].copy()
    prefix.loc[prefix.cohort == 10, ["cohort", "treated", "treat"]] = 0
    prefix.loc[prefix.cohort == 0, "rel_time"] = -1
    d = design(prefix, "did2s", "did2s", endpoint="untreated_prefix_imputation_att")
    out = run(d, prefix)
    assert out[0].error is None and out[0].value == pytest.approx(2.0, abs=0.3)
    assert d.row_ids == tuple(prefix.index)


def test_unsupported_exact_target_is_a_recorded_failure_not_silent_renormalization():
    p = panel()
    raw = design(p, "saturated", "saturated_event_study", "cohort_horizon").model_dump(exclude={"integrity"})
    raw["target"] = {"aggregation": "cohort_horizon", "horizons": [100], "cohorts": ["4"]}
    d = Design.model_validate(raw)
    assert run(d, p)[0].error is not None
    assert not target_groups(p, d.target)[0][1].any()


def test_saturated_exact_cohort_horizon_uses_only_the_requested_cell():
    p = panel()
    raw = design(p, "saturated", "saturated_event_study", "cohort_horizon").model_dump(exclude={"integrity"})
    raw["target"] = {"aggregation": "cohort_horizon", "horizons": [1], "cohorts": ["4"], "population": "treated"}
    out = run(Design.model_validate(raw), p)
    assert len(out) == 1 and out[0].error is None
    assert out[0].level == "4:horizon:1"
    assert out[0].value == pytest.approx(2.0, abs=0.3)


def test_execution_accepts_frozen_csv_row_identity_without_library_index_collision():
    p = panel().set_index("row_id", drop=False)
    d = design(p, "did2s", "did2s_dynamic", "horizon_profile")
    assert all(e.error is None for e in run(d, p))


def test_nested_mutation_is_rejected_before_any_fit():
    p = panel()
    d = design(p, "did2s", "did2s")
    d.target["population"] = "changed population"
    with pytest.raises(ValueError, match="frozen specification changed"):
        run(d, p)


def test_dynamic_projection_freezes_exact_linear_response_weights():
    from causal_agent.families.diff_in_diff.lane.engine.estimators import fit
    from causal_agent.families.diff_in_diff.lane.targets import horizon_projection_weights

    p = panel(cohorts=(4,))
    weights = horizon_projection_weights(p, [], 2)
    m = fit("y ~ i(rel_time, treated, ref=-1) | unit+time_index", p, {"CRV1": "cluster"})
    name = next(n for n in m.coef().index if n.startswith("rel_time::2:") or n == "rel_time::2")
    reconstructed = sum(p.loc[int(rowid), "y"] * w for rowid, w in weights.items())
    assert reconstructed == pytest.approx(float(m.coef()[name]), abs=1e-8)
    p.y = 100000
    assert horizon_projection_weights(p, [], 2) == weights


@pytest.mark.parametrize("representation", ["label", "indicator", "cohort_column"])
@pytest.mark.parametrize("never", [True, False])
def test_desk_support_fact_matches_lane_adoption_mapping_without_any_outcome(representation, never):
    from causal_agent.families.diff_in_diff.probes import context_facts
    from causal_agent.memory.claims import Claim, ClaimTable

    p = panel(never=never, cohorts=(4, 7) if representation != "label" else (4,)).drop(columns="y")
    receipt = "treated" if representation == "label" else "treat"
    claims = ClaimTable(
        claims={
            "grain": Claim(kind="grain", key="grain", fields={"key_columns": ["unit", "time"]}),
            "assignment": Claim(
                kind="assignment", key="assignment", fields={"treatment_column": receipt, "treated_level": "1" if representation == "label" else "1.0"}
            ),
            "change": Claim(
                kind="change",
                key="change",
                fields={"date_column": "time", "period_value": "4", **({"adoption_column": "cohort"} if representation == "cohort_column" else {})},
            ),
        }
    )
    fact = context_facts(p, claims, {})["claim:assignment.never_treated_exists"]
    assert fact is never
    if representation == "cohort_column":
        p.loc[p.index[0], "cohort"] = np.nan
        assert context_facts(p, claims, {}) == {"claim:assignment.never_treated_exists": None}


def test_linear_target_inference_matches_native_pyfixest_degrees_of_freedom():
    from causal_agent.families.diff_in_diff.lane.engine.estimators import fit, linear_combination

    p = panel(cohorts=(4,), units=4)
    m = fit("y ~ treat | unit+time_index", p, {"CRV3": "cluster"})
    effect, low, high, pvalue = linear_combination(m, {"treat": 1.0})
    native = m.tidy().loc["treat"]
    assert effect == pytest.approx(native["Estimate"])
    assert low == pytest.approx(native["2.5%"])
    assert high == pytest.approx(native["97.5%"])
    assert pvalue == pytest.approx(native["Pr(>|t|)"], abs=1e-12)

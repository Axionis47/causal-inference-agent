# The observational AIPW adapter (T-027 §2; PRD-004 §10, §17 row two): score parity against
# hand-computed numbers, the deterministic fold deal, the §10.2 leakage contract under wall 6, the
# §10.5 overlap guard, the §10.6 branch loop, and the rule that no prediction, weight, or influence
# value ever reaches a committed payload.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from causal.analysis.common import legacy_diagnostics as diagnostic_policy
from causal.analysis.common import legacy_engine as engine
from causal.analysis.integration import RESOURCE_ROOT
from causal.analysis.integration import contracts as ec
from causal.analysis.integration import walls as ew
from causal.analysis.integration.packs import load_estimation_packs
from causal.analysis.methods.aipw import estimation as aipw
from causal.analysis.tests.support import builders as te
from causal.shared.canonical import content_hash

REGISTRIES = Path(__file__).resolve().parents[6] / "registries"
PACKS = load_estimation_packs(RESOURCE_ROOT / "method-pack-estimation.v1.json",
                              REGISTRIES / "method-packs.v1.json")
PACK = PACKS.get("aipw", "aipw-pack.v1")
RULES = ew.load_validation_rules(RESOURCE_ROOT / "estimation-validation-rules.v1.json")
CONTRAST = "treated_vs_control"
ROLES = {"unit_identifier": "unit_id", "treatment": "arm", "outcome": "y",
         "adjustment_covariate": "x1", "effect_modifier": "x2",
         "missingness_indicator": "x1_missing"}
# One hand-checked six-row case: the outcome, the treatment, and the out-of-fold nuisance
# predictions are all fixed here, so the numbers below are arithmetic and not a re-run.
CASE = {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "a": [True, False, True, False, True, False],
        "p": [0.5, 0.5, 0.25, 0.75, 0.4, 0.6], "mu0": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "mu1": [2.0, 2.0, 3.0, 3.0, 4.0, 4.0]}
# ATE and ATT of that case, with their influence-function standard errors.
ATE, ATE_SE = -1.8333333333333333, 1.548595540529595
ATT, ATT_SE = -2.8333333333333335, 3.388433484883757


def table(units: int = 120, *, effect: float = 2.0, strength: float = 0.6,
          seed: int = 11) -> pl.DataFrame:
    """One frozen observational table: confounded treatment, one outcome, two covariates.

    Every column is a deterministic function of the fixture seed, so a fold deal or an estimate
    that moves between runs moved because the estimator changed, not because the data did.
    """
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=units), rng.normal(size=units)
    treated = rng.random(units) < 1.0 / (1.0 + np.exp(-(strength * x1 + 0.3 * x2)))
    outcome = (1.0 + 0.5 * x1 + 0.25 * x2 + effect * treated.astype(float)
               + rng.normal(scale=0.5, size=units))
    return pl.DataFrame({
        "unit_id": [f"u{index:03d}" for index in range(units)],
        "arm": np.where(treated, "treated", "control"), "y": outcome, "x1": x1, "x2": x2,
        "x1_missing": np.zeros(units)})


def make_plan(**over: Any) -> ec.EstimationPlanV1:
    # The frozen §6.1 plan for the AIPW pack: its own defaults, diagnostics, and branch set.
    fields: dict[str, Any] = {
        "method_id": PACK.method_id, "method_pack_version": PACK.pack_version,
        "estimator_id": PACK.estimator_id, "estimator_version": PACK.estimator_version,
        "uncertainty_method": PACK.uncertainty_method, "estimand_id": "ate",
        "finite_sample_correction": PACK.finite_sample_correction, "role_columns": dict(ROLES),
        "primary_mask_rule_id": "cross_fit_predicted", "contrast_ids": (CONTRAST,),
        "estimator_parameters": dict(PACK.parameter_defaults), "fold_count": 5,
        "nuisance_profile_id": "regularized_glm", "required_diagnostics": PACK.severities(),
        "fold_assignment_rule_id": "stratified_by_treatment_and_cluster",
        "preprocessing_recipe_ids": ("estimator_scoped_recipe",),
        "required_sensitivity_ids": tuple(row.branch_id for row in PACK.sensitivity_branches),
        "figure_builder_ids": PACK.figure_builder_ids,
        "numerical_failure_rule_ids": PACK.not_estimable_rule_ids}
    return te.make_plan(**(fields | over))


def case_arrays() -> tuple[aipw.Data, np.ndarray, np.ndarray, np.ndarray]:
    treated = np.array(CASE["a"], dtype=np.bool_)
    return (aipw.Data(np.zeros((6, 1)), np.array(CASE["y"]), treated, False),
            np.array(CASE["p"]), np.array(CASE["mu0"]), np.array(CASE["mu1"]))


def fitted(view: pl.DataFrame, plan: ec.EstimationPlanV1, **params: Any) -> aipw.CrossFitRun:
    return aipw.cross_fit(view, plan, PACK, dict(plan.estimator_parameters) | params)


def peeking(ledger: aipw.Ledger, fold: int, profile: Any, recipe_ids: Any) -> aipw.Held:
    # The trap: a fold fit that trains on every frozen row, held-out outcomes included. It reaches
    # its rows through the same ledger the honest fit uses, so the receipt records what it took.
    seen, hyper = ledger.read(fold, "all"), dict(profile.hyperparameters)
    out = ledger.held(fold)
    model = aipw._learner(profile.propensity_learner, hyper, True).fit(
        seen.features, seen.treated.astype(np.int64))
    predicted = [aipw._learner(profile.outcome_learner, hyper, seen.binary).fit(
        seen.features[seen.treated == state], seen.outcome[seen.treated == state]).predict(out)
        for state in (False, True)]
    return (model.predict_proba(out)[:, 1], predicted[0], predicted[1], True)


def wall_six(assignment: ec.CrossFitAssignmentV1, run: aipw.CrossFitRun) -> Any:
    return ew.wall(6, ew.WallContext(rules=RULES, assignments=(assignment,),
                                     fold_fit_counts=run.receipts))


def assignment_of(plan: ec.EstimationPlanV1, run: aipw.CrossFitRun) -> ec.CrossFitAssignmentV1:
    payload = aipw.mapping_object_payload(run.folds, seed=run.seed, count=run.fold_count)
    found = content_hash(payload)
    return run.assignment(
        plan, ec.ObjectRefV1(object_locator=f"objects/{found}", content_hash=found),
        plan_ref=te.ref("plan"), parents=(te.ref("plan"),))


@pytest.mark.parametrize(("estimand", "expected", "error"),
                         [("ate", ATE, ATE_SE), ("att", ATT, ATT_SE)])
def test_the_score_matches_the_hand_computed_estimand(estimand: str, expected: float,
                                                      error: float) -> None:
    data, propensity, under_control, under_treated = case_arrays()
    found = aipw.score(data, propensity, under_control, under_treated, estimand=estimand,
                       bound=0.01, level=0.95)
    assert found.estimate == pytest.approx(expected, abs=1e-12)
    assert found.standard_error == pytest.approx(error, abs=1e-12)
    assert found.lower < found.estimate < found.upper
    assert found.influence.size == 6 and float(found.influence.mean()) == pytest.approx(0.0)
    assert found.bounded_rows == 0


def test_the_numerical_bound_guards_division_and_is_reported_apart() -> None:
    data, _, under_control, under_treated = case_arrays()
    edge = np.array([0.0, 0.5, 0.25, 0.75, 0.4, 1.0])
    found = aipw.score(data, edge, under_control, under_treated, estimand="ate", bound=0.01,
                       level=0.95)
    assert found.bounded_rows == 2 and np.isfinite(found.estimate)


def test_one_seed_deals_one_assignment_and_another_seed_deals_another() -> None:
    view, plan = table(effect=0.0), make_plan()
    run, again = fitted(view, plan), fitted(view, plan)
    moved = fitted(view, plan, seed_offset=1)
    assert np.array_equal(run.folds, again.folds)
    assert content_hash(aipw.mapping_object_payload(
        run.folds, seed=run.seed, count=run.fold_count)) == content_hash(
        aipw.mapping_object_payload(again.folds, seed=again.seed, count=again.fold_count))
    assert not np.array_equal(run.folds, moved.folds)
    # A null-effect fixture: a different deal is a different fold set, not a different answer.
    assert moved.score.estimate == pytest.approx(run.score.estimate, abs=0.25)
    assert run.score.lower <= 0.0 <= run.score.upper


def test_the_deal_is_treatment_stratified_and_the_assignment_records_it() -> None:
    view, plan = table(), make_plan()
    run = fitted(view, plan)
    found = assignment_of(plan, run)
    counts = found.counts_by_fold
    assert found.fold_count == 5 and len(counts) == 5
    assert found.assignment_algorithm_id == aipw.ASSIGNMENT_ALGORITHM
    assert found.nuisance_profile_id == "regularized_glm"
    assert found.stratification_rule_ids == ("stratified_by_treatment_and_cluster",)
    assert sum(row["validation"] for row in counts.values()) == view.height
    # Each treatment state is dealt round-robin on its own, so a fold's share of either state
    # differs from another's by at most one row.
    held = [row["validation"] for row in counts.values()]
    treated = [row["treated"] for row in counts.values()]
    assert max(held) - min(held) <= 2 and max(treated) - min(treated) <= 1


def test_the_honest_fold_loop_never_trains_on_a_held_out_row() -> None:
    view, plan = table(), make_plan()
    run = fitted(view, plan)
    assert run.leaked_rows == 0
    assert all(row["train"] == run.counts_by_fold[name]["train"]
               for name, row in run.receipts.items())
    assert wall_six(assignment_of(plan, run), run).passed
    # A fold's held-out rows arrive as FEATURES and are receipted to nobody: an outcome under
    # validation is not reachable from inside a fold fit at all.
    ledger = aipw.Ledger(aipw.frame_data(view, dict(plan.role_columns), CONTRAST), run.folds)
    assert ledger.held(0).shape == (run.counts_by_fold["fold_0"]["validation"], 3)
    assert ledger.receipts == {}


def test_a_fold_fit_that_peeks_at_validation_outcomes_is_caught_by_wall_six() -> None:
    view, plan = table(), make_plan()
    honest = fitted(view, plan)
    poisoned = aipw.cross_fit(view, plan, PACK, dict(plan.estimator_parameters), peeking)
    assert poisoned.leaked_rows == view.height
    assert not np.allclose(poisoned.predictions[0], honest.predictions[0])
    report = wall_six(assignment_of(plan, poisoned), poisoned)
    assert not report.passed
    assert [issue.code for issue in report.issues] == ["fold_leakage_detected"]
    assert all(name.endswith(":validation_rows_fitted")
               for name in report.issues[0].artifact_ids)


def test_a_fold_without_both_treatment_states_is_not_estimable() -> None:
    view = table(units=12).with_columns(pl.when(pl.col("unit_id") == "u000")
                                        .then(pl.lit("control")).otherwise(pl.lit("treated"))
                                        .alias("arm"))
    with pytest.raises(ec.EstimationError) as found:
        fitted(view, make_plan(fold_count=2), fold_count=2)
    assert found.value.code == aipw.FOLD_WITHOUT_BOTH_STATES


def test_mass_at_the_edges_of_the_unit_interval_invalidates_the_overlap_guard() -> None:
    plan = make_plan()
    adapter = aipw.ObservationalAipwAdapter(te.ref("mask"))
    honest = adapter.fit(table(), plan, PACK)
    edge = adapter.fit(table(strength=9.0), plan, PACK)
    results = {row.diagnostic_id: row for row in diagnostic_policy.run_diagnostics(
        PACK, edge.harvest, te.base())}
    guard = results["propensity_common_support"]
    assert guard.policy_result == "invalidating" and guard.execution_status == "computed"
    assert diagnostic_policy.severity_result(
        "invalidation_guard", honest.harvest["propensity_common_support"],
        {"min_propensity": 0.01, "max_propensity": 0.99,
         "max_out_of_support_share": 0.05}) == "acceptable"


def test_every_required_diagnostic_and_branch_reaches_a_terminal_result() -> None:
    view, plan = table(), make_plan()
    adapter = aipw.ObservationalAipwAdapter(te.ref("mask"))
    found = adapter.fit(view, plan, PACK)
    assert set(PACK.severities()) <= set(found.harvest)
    rows = diagnostic_policy.run_diagnostics(PACK, found.harvest, te.base())
    assert [row.execution_status for row in rows] == ["computed"] * len(rows)
    item = found.items[0]
    assert item.contrast_id == CONTRAST and item.adapter_version == aipw.ADAPTER_VERSION
    assert item.method_quantities["fold_count"] == 5
    assert item.method_quantities["propensity_bound_rule_id"] == aipw.BOUND_RULE
    assert item.estimate == pytest.approx(2.0, abs=0.4)
    branches = engine.run_sensitivities(plan, PACK, adapter, view, item, te.base())
    assert {row.branch_id for row in branches} == set(plan.required_sensitivity_ids)
    assert [row.execution_status for row in branches] == ["computed"] * len(branches)
    # Every §10.6 branch carries its own estimator identity, not the primary run's.
    named = {row.branch_id: row.result.method_quantities for row in branches if row.result}
    assert named["alternative_nuisance_profile"][
        "nuisance_profile_id"] == "histogram_gradient_boosting"
    assert named["alternative_cross_fit_seed"]["fold_count"] == 10
    assert named["ate_versus_att"]["estimand"] == "att"
    assert named["propensity_bound_sensitivity"]["propensity_bound"] == 0.02


def test_scientific_overlap_counts_account_for_every_contributing_row() -> None:
    view, plan = table(), make_plan()
    adapter = aipw.ObservationalAipwAdapter(te.ref("mask"))
    found = adapter.fit(view, plan, PACK)
    overlap = found.harvest["overlap_bins"]
    bins = {name: value for name, value in overlap.items() if "_bin_" in name}
    assert len(bins) == 2 * aipw.OVERLAP_BINS
    assert sum(bins.values()) == view.height
    assert {name.rpartition("_bin_")[0] for name in bins} == {"control", "treated"}
    assert all(isinstance(value, int) and value >= 0 for value in bins.values())
    item = found.items[0]
    assert item.contributing_counts["row"] == view.height
    assert item.interval_lower <= item.estimate <= item.interval_upper


def test_no_prediction_weight_or_influence_reaches_a_committed_payload() -> None:
    view, plan = table(), make_plan()
    adapter = aipw.ObservationalAipwAdapter(te.ref("mask"))
    found = adapter.fit(view, plan, PACK)
    run = found.fit
    assert isinstance(run, aipw.CrossFitRun)
    text = json.dumps({"item": found.items[0].model_dump(mode="json"),
                       "harvest": found.harvest}, sort_keys=True)
    assert "[" not in text and "propensity_hex" not in text
    # Declared aggregate common-support extrema necessarily equal observed propensity values;
    # their two scalars are allowed, while individual prediction vectors remain restricted.
    limits = {found.harvest["overlap_bins"][key]
              for key in ("support_limit_lower", "support_limit_upper")}
    for array in (run.predictions[0], run.score.weights, run.score.influence):
        assert all(f"{float(value):.9f}"[:12] not in text
                   for value in array[:5] if float(value) not in limits)
    payload = run.prediction_payload()
    assert payload["row_count"] == view.height and payload["dtype"] == "float64"
    assert all(isinstance(value, str | int) for value in payload.values())
    assert len(str(payload["propensity_hex"])) == view.height * 16


@pytest.mark.parametrize("estimand", ["ate", "att"])
def test_public_ate_and_att_select_the_same_prespecified_numerical_score(estimand: str) -> None:
    from causal.analysis import interface as api
    from causal.analysis.tests.support.interface import approve, design

    source = table()
    fixed = design("aipw", {"adjustment_set_pre_treatment": True})
    configuration = {
        "method": "aipw", "estimand": estimand, "treatment_column": "arm",
        "unit_column": "unit_id", "treated_value": "treated", "comparator_value": "control",
        "covariate_columns": ("x1",)}
    actual = api.execute(approve(source, fixed, configuration), source)
    roles = {"outcome": "y", "treatment": "arm", "unit_identifier": "unit_id",
             "adjustment_covariate": "x1"}
    plan = make_plan(seed=17, estimand_id=estimand, role_columns=roles,
                     estimator_parameters=dict(PACK.parameter_defaults) | {"estimand": estimand})
    expected = aipw.ObservationalAipwAdapter(te.ref("mask")).fit(source, plan, PACK).items[0]
    assert actual.primary.status == "computed", actual.primary.explanation
    item = actual.primary.estimates[0]
    assert item.estimand == estimand
    assert item.estimate == pytest.approx(expected.estimate, abs=1e-12)
    assert item.standard_error == pytest.approx(expected.standard_error, abs=1e-12)


def test_public_sensitivities_honor_fold_assignment_and_numeric_bound_without_dropping_rows() -> None:
    from causal.analysis import interface as api
    from causal.analysis.tests.support.interface import approve, design

    source = table(units=160, strength=6.0)
    fixed = design("aipw", {"adjustment_set_pre_treatment": True})
    configuration = {"method": "aipw", "estimand": "ate", "treatment_column": "arm",
                     "unit_column": "unit_id", "treated_value": "treated", "comparator_value": "control",
                     "covariate_columns": ("x1",)}
    deltas = {"alternative_cross_fit_seed": {"seed_offset": 1, "fold_count": 10},
              "propensity_bound_sensitivity": {"propensity_bound": 0.02}}
    actual = api.execute(approve(source, fixed, configuration, sensitivities=tuple(deltas)), source)
    plan = make_plan(seed=17, role_columns={"outcome": "y", "treatment": "arm",
        "unit_identifier": "unit_id", "adjustment_covariate": "x1"})
    assert [branch.computation_id for branch in actual.sensitivities] == list(deltas)
    for branch in actual.sensitivities:
        expected = aipw.ObservationalAipwAdapter(te.ref("mask")).fit(
            source, plan, PACK, deltas[branch.computation_id]).items[0]
        assert branch.status == "computed", branch.explanation
        assert branch.estimates[0].estimate == pytest.approx(expected.estimate, abs=1e-12)
        assert branch.estimates[0].standard_error == pytest.approx(expected.standard_error, abs=1e-12)
        assert {m.name: m.value for m in branch.estimates[0].population}["row"] == source.height
        assert branch.estimates[0].estimate != pytest.approx(actual.primary.estimates[0].estimate, abs=1e-8)


def test_selected_ten_fold_sensitivity_requires_ten_units_in_each_arm_before_execution() -> None:
    from causal.analysis import interface as api
    from causal.analysis.tests.support.interface import assess, design

    source = table(units=12).with_columns(pl.Series("arm", ["control", "treated"] * 6))
    fixed = design("aipw", {"adjustment_set_pre_treatment": True})
    assessment = assess(fixed, {
        "dataset": api.identify_data(source, "small-study"),
        "configuration": {"method": "aipw", "estimand": "ate", "treatment_column": "arm",
                          "unit_column": "unit_id", "treated_value": "treated", "comparator_value": "control",
                          "covariate_columns": ("x1",)},
        "sensitivities": ["alternative_cross_fit_seed"]})
    assert assessment.specification is not None
    result = api.preflight(assessment.specification, source)
    assert not result.ready
    assert any(issue.category == "incompatible_data" for issue in result.issues)


@pytest.mark.parametrize(("kind", "binary", "units"), [
    ("continuous", False, "score points"), ("binary", True, "risk_difference")])
def test_public_outcome_kind_selects_the_declared_nuisance_model(
        kind: str, binary: bool, units: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from causal.analysis import interface as api
    from causal.analysis.contracts import Outcome
    from causal.analysis.tests.support.interface import approve, design

    source = table().with_columns((
        pl.col("unit_id").str.strip_prefix("u").cast(pl.Int64) % 3 == 0).cast(pl.Float64).alias("y"))
    fixed = design("aipw", {"adjustment_set_pre_treatment": True}).model_copy(update={
        "outcome": Outcome(column="y", kind=kind, units="score points")})
    configuration = {"method": "aipw", "estimand": "ate", "treatment_column": "arm",
                     "unit_column": "unit_id", "treated_value": "treated", "comparator_value": "control",
                     "covariate_columns": ("x1",)}
    learner = aipw._learner
    choices = []

    def observed(learner_id: str, hyper: ec.ValueMap, is_binary: bool) -> Any:
        if learner_id == "sklearn_ridge_regression_l2":
            choices.append(is_binary)
        return learner(learner_id, hyper, is_binary)

    monkeypatch.setattr(aipw, "_learner", observed)
    actual = api.execute(approve(source, fixed, configuration), source)
    assert actual.primary.status == "computed", actual.primary.explanation
    assert choices and set(choices) == {binary}
    assert actual.primary.estimates[0].units == units


def test_reported_fold_leakage_invalidates_integrity_without_erasing_primary_estimates(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from causal.analysis import interface as api
    from causal.analysis.methods.aipw import diagnostics
    from causal.analysis.tests.support.interface import approve, design

    source = table()
    fixed = design("aipw", {"adjustment_set_pre_treatment": True})
    configuration = {"method": "aipw", "estimand": "ate", "treatment_column": "arm",
                     "unit_column": "unit_id", "treated_value": "treated", "comparator_value": "control",
                     "covariate_columns": ("x1",)}
    approved = approve(source, fixed, configuration)
    original = diagnostics.harvest

    def leaked(*args: Any, **kwargs: Any) -> Any:
        rows = original(*args, **kwargs)
        rows["cross_fit_score_integrity"]["validation_rows_fitted"] = 1
        return rows

    monkeypatch.setattr(diagnostics, "harvest", leaked)
    actual = api.execute(approved, source)
    check = next(row for row in actual.diagnostics if row.computation_id == "cross_fit_score_integrity")
    assert check.status == "computed" and check.policy == "invalidating"
    assert actual.status == "incomplete"
    assert actual.primary.status == "computed" and actual.primary.estimates


def test_disjoint_support_is_explicit_without_an_invented_overlap_interval() -> None:
    from causal.analysis.methods.aipw import diagnostics

    values = diagnostics._overlap_values(np.array([0.1, 0.2, 0.8, 0.9]),
                                        np.array([False, False, True, True]))
    assert values["common_support_exists"] is False
    assert values["support_limit_lower"] == 0.8
    assert values["support_limit_upper"] == 0.2
    assert sum(value for key, value in values.items() if "_bin_" in key) == 4

"""Standalone analysis contracts: honest readiness, immutable approval, and execution evidence."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from causal.analysis import interface as api
from causal.analysis.contracts import ApprovedPlan, BoundaryError, StudyFact
from causal.analysis.tests.support.interface import approve, assess, candidate, design

CONFIGURATION = {
    "method": "randomized", "treatment_column": "arm", "unit_column": "id",
    "treated_value": "treated", "comparator_value": "control",
    "estimator": "difference_in_means"}
FACTS = {"assignment_mechanism": "individual_randomized"}


@pytest.fixture()
def data() -> pl.DataFrame:
    return pl.DataFrame({
        "id": list(range(24)), "arm": ["control", "treated"] * 12,
        "y": [2.0 + 3.0 * (i % 2) + ((i * 7) % 11) / 10 for i in range(24)]})


def draft(data: pl.DataFrame, **configuration: Any) -> dict[str, Any]:
    return {"dataset": api.identify_data(data, "frozen-study").model_dump(mode="json"),
            "configuration": CONFIGURATION | configuration, "seed": 17}


def test_discovery_and_partial_guidance_import_no_execution_services() -> None:
    code = '''
import json, sys
from causal.analysis import interface
methods = interface.list_methods()
for method in methods:
    interface.retrieve_guidance(method.method, "overview")
    interface.retrieve_guidance(method.method, "configuration", draft={})
forbidden = {"polars", "numpy", "pandas", "scipy", "sklearn", "pyfixest", "rdrobust",
             "rddensity", "psycopg", "boto3", "langgraph", "langsmith", "google.genai"}
loaded = sorted(name for name in sys.modules
                if any(name == forbidden_name or name.startswith(forbidden_name + ".")
                       for forbidden_name in forbidden))
print(json.dumps({"methods": sorted(m.method for m in methods), "loaded": loaded}))
'''
    root = Path(__file__).resolve().parents[4]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True,
                            capture_output=True, check=True)
    observed = json.loads(result.stdout)
    assert observed == {"methods": ["aipw", "did", "randomized", "rdd"], "loaded": []}


def test_partial_information_returns_actionable_context_issues(data: pl.DataFrame) -> None:
    fixed = design("randomized", {})
    result = api.evaluate_candidate(candidate(fixed, {
        "dataset": api.identify_data(data, "frozen-study").model_dump(mode="json"),
        "configuration": {"method": "randomized"}}))
    assert result.status == "needs_information"
    assert result.issues and {issue.category for issue in result.issues} == {"missing_context"}
    assert any("assignment_mechanism" in issue.field for issue in result.issues)
    assert all(issue.finding and issue.requirement and issue.explanation for issue in result.issues)


def test_a_fact_without_evidence_does_not_satisfy_the_design(data: pl.DataFrame) -> None:
    fixed = design("randomized", {}).model_copy(update={"facts": (
        StudyFact(name="assignment_mechanism", value="individual_randomized",
                  original_answer="I think the participants were randomized."),)})
    result = api.evaluate_candidate(candidate(fixed, draft(data)))
    assert result.status == "needs_information"
    assert any(issue.category == "missing_context" and "assignment_mechanism" in issue.field
               for issue in result.issues)


@pytest.mark.parametrize("configuration", [
    {"estimator": "choose_lowest_p_value"}, {"confidence_level": 0.90},
    {"cluster_covariance": "cr2"}])
def test_unsupported_settings_are_feedback_not_automatic_repairs(
        data: pl.DataFrame, configuration: dict[str, Any]) -> None:
    result = api.evaluate_candidate(candidate(design("randomized", FACTS), draft(data, **configuration)))
    assert result.status == "rejected"
    assert any(issue.category == "unsupported_capability" for issue in result.issues)


def test_a_contradictory_assignment_unit_is_rejected(data: pl.DataFrame) -> None:
    result = api.evaluate_candidate(candidate(design("randomized", FACTS),
                                      draft(data, cluster_column="school")))
    assert result.status == "rejected"
    assert any(issue.category == "contradictory_configuration" for issue in result.issues)


def test_data_preflight_rejects_repeated_units_without_fitting(data: pl.DataFrame) -> None:
    repeated = data.with_columns(pl.lit(1).alias("id"))
    assessment = assess(design("randomized", FACTS), draft(repeated))
    assert assessment.specification is not None
    result = api.preflight(assessment.specification, repeated)
    assert not result.ready
    assert any(issue.category == "incompatible_data" for issue in result.issues)
    assert any(target.kind == "data_preparation" and target.field == "dataset.id"
               for target in result.resolution_targets)
    with pytest.raises(BoundaryError):
        api.compile_plan(assessment.specification, result)


def test_restoring_a_missing_column_exposes_deeper_runnability_failure_without_redesign(
        data: pl.DataFrame) -> None:
    repeated = data.with_columns(pl.lit(1).alias("id"))
    missing = repeated.drop("arm")
    initial = assess(design("randomized", FACTS), draft(missing))
    assert initial.specification is not None
    first = api.preflight(initial.specification, missing)
    assert not first.ready and any(issue.category == "missing_data" for issue in first.issues)
    fixed = initial.specification.design
    rebound = api.assess_specification(fixed, {
        "dataset": api.identify_data(repeated, "restored-representation")})
    assert rebound.specification is not None and rebound.specification.design == fixed
    deeper = api.preflight(rebound.specification, repeated)
    assert not deeper.ready and any(issue.field == "dataset.id" for issue in deeper.issues)
    assert deeper.resolution_targets
    assert all(target.kind == "data_preparation" for target in deeper.resolution_targets)


def test_preflight_cannot_be_reused_for_a_changed_specification(data: pl.DataFrame) -> None:
    assessment = assess(design("randomized", FACTS), draft(data))
    specification = assessment.specification
    assert specification is not None
    checks = api.preflight(specification, data)
    with pytest.raises(BoundaryError):
        api.compile_plan(specification.model_copy(update={"seed": 19}), checks)


def test_approved_plan_is_immutable_and_hash_pinned(data: pl.DataFrame) -> None:
    approved = approve(data, design("randomized", FACTS), CONFIGURATION)
    with pytest.raises(ValidationError, match="frozen"):
        approved.plan.specification.seed = 1
    with pytest.raises(BoundaryError):
        api.execute(approved.model_copy(update={"approved_hash": "f" * 64}), data)


def test_data_or_capability_version_drift_requires_a_new_plan(data: pl.DataFrame) -> None:
    approved = approve(data, design("randomized", FACTS), CONFIGURATION)
    changed = data.with_columns(pl.lit("new").alias("unused"))
    with pytest.raises(BoundaryError):
        api.execute(approved, changed)
    stale = approved.plan.model_copy(update={"capability_version": "retired-version"})
    stale_approval = ApprovedPlan(plan=stale, approved_hash=stale.plan_hash,
                                 approved_by="reviewer", approval_reference="approval:18")
    with pytest.raises(BoundaryError):
        api.execute(stale_approval, data)


def test_unadjusted_randomized_journey_matches_arithmetic_and_replays(data: pl.DataFrame) -> None:
    approved = approve(data, design("randomized", FACTS), CONFIGURATION)
    first = api.execute(approved, data)
    second = api.execute(approved, data)
    expected = data.filter(pl.col("arm") == "treated")["y"].mean() - data.filter(
        pl.col("arm") == "control")["y"].mean()
    assert first.primary.status == "computed"
    assert first.primary.estimates[0].estimate == pytest.approx(expected, abs=1e-12)
    assert first.primary.estimates[0].units == "points"
    assert first.status in {"completed", "completed_with_limitations"}
    assert first.diagnostics and first.provenance.plan_hash == approved.plan.plan_hash
    assert first.provenance.seed == 17
    assert first.model_dump() == second.model_dump()


def test_diagnostic_failure_preserves_the_completed_primary(
        data: pl.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    from causal.analysis.methods.randomized import diagnostics

    approved = approve(data, design("randomized", FACTS), CONFIGURATION)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("diagnostic calculation failed")

    monkeypatch.setattr(diagnostics, "harvest", fail)
    result = api.execute(approved, data)
    assert result.primary.status == "computed" and result.primary.estimates
    assert any(row.status == "failed" for row in result.diagnostics)
    assert result.status == "incomplete"


def test_sensitivity_cannot_introduce_an_undeclared_covariate(data: pl.DataFrame) -> None:
    proposal = draft(data, estimator="ancova") | {"sensitivities": ["unadjusted_itt"]}
    result = api.evaluate_candidate(candidate(design("randomized", FACTS), proposal))
    assert result.status == "needs_information"
    assert any(issue.field == "sensitivities.unadjusted_itt" for issue in result.issues)


def test_reapproving_a_tampered_plan_cannot_remove_required_diagnostics(data: pl.DataFrame) -> None:
    approved = approve(data, design("randomized", FACTS), CONFIGURATION)
    omitted = approved.plan.model_copy(update={"diagnostics": ()})
    changed = ApprovedPlan(plan=omitted, approved_hash=omitted.plan_hash,
                           approved_by="reviewer", approval_reference="approval:changed")
    with pytest.raises(BoundaryError):
        api.execute(changed, data)


def test_planned_covariance_sensitivity_keeps_the_primary_and_reports_its_own_result(
        data: pl.DataFrame) -> None:
    approved = approve(data, design("randomized", FACTS), CONFIGURATION,
                        sensitivities=("sensitivity_covariance_profile",))
    result = api.execute(approved, data)
    assert result.primary.status == "computed"
    assert [(row.computation_id, row.status) for row in result.sensitivities] == [
        ("sensitivity_covariance_profile", "computed")]
    primary, sensitivity = result.primary.estimates[0], result.sensitivities[0].estimates[0]
    assert sensitivity.estimate == pytest.approx(primary.estimate, abs=1e-12)
    assert sensitivity.standard_error != pytest.approx(primary.standard_error, rel=1e-4)


def test_diagnostic_guidance_is_retrievable_directly_and_resolves_its_prerequisite() -> None:
    absent = api.retrieve_guidance("randomized", "diagnostic_details",
                                   diagnostic_id="baseline_balance")
    assert len(absent.diagnostics) == 1
    assert absent.diagnostics[0].obligation == "optional"
    assert absent.diagnostics[0].applicability == "inapplicable"
    assert absent.exclusions and absent.unresolved_prerequisites
    partial = api.retrieve_guidance("randomized", "diagnostic_details",
        draft={"configuration": CONFIGURATION | {"precision_covariate_column": "baseline"}},
        diagnostic_id="baseline_balance")
    assert partial.diagnostics[0].applicability == "unresolved"
    supplied = api.retrieve_guidance("randomized", "diagnostic_details",
        draft=candidate(design("randomized", FACTS), {
            "configuration": CONFIGURATION | {"precision_covariate_column": "baseline"}}),
        diagnostic_id="baseline_balance")
    assert supplied.diagnostics[0].applicability == "applicable"
    assert supplied.capability_version == absent.capability_version
    assert supplied.input_schema and supplied.explanation


def test_optional_diagnostic_selection_changes_checks_without_changing_the_primary(
        data: pl.DataFrame) -> None:
    source = data.with_columns((pl.col("id") % 7).cast(pl.Float64).alias("baseline"))
    fixed = design("randomized", FACTS)
    configuration = CONFIGURATION | {"estimator": "ancova", "precision_covariate_column": "baseline"}
    unselected = api.execute(approve(source, fixed, configuration), source)
    selected = api.execute(approve(source, fixed, configuration, diagnostics=("baseline_balance",)), source)
    before = next(row for row in unselected.diagnostics if row.computation_id == "baseline_balance")
    after = next(row for row in selected.diagnostics if row.computation_id == "baseline_balance")
    assert before.status == "not_selected" and after.status == "computed"
    assert selected.primary.estimates == unselected.primary.estimates
    assert selected.provenance.plan_hash != unselected.provenance.plan_hash


def test_selecting_an_inapplicable_optional_diagnostic_is_rejected(data: pl.DataFrame) -> None:
    result = api.evaluate_candidate(candidate(design("randomized", FACTS),
                                     draft(data) | {"diagnostics": ["baseline_balance"]}))
    assert result.status == "rejected"
    assert any(issue.field == "diagnostics.baseline_balance" and
               issue.category == "contradictory_configuration" for issue in result.issues)


def test_selected_optional_diagnostic_with_unresolved_grouping_needs_context(
        data: pl.DataFrame) -> None:
    result = api.evaluate_candidate(candidate(design("randomized", FACTS),
        draft(data, estimator="ancova", precision_covariate_column="baseline", treatment_column=None) |
        {"diagnostics": ["baseline_balance"]}))
    assert result.status == "needs_information"
    assert any(issue.field == "diagnostics.baseline_balance" and
               issue.category == "missing_context" for issue in result.issues)


def test_an_unselected_optional_check_is_not_computed(
        data: pl.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    from causal.analysis.methods.randomized import diagnostics

    source = data.with_columns((pl.col("id") % 7).cast(pl.Float64).alias("baseline"))
    configuration = CONFIGURATION | {"estimator": "ancova", "precision_covariate_column": "baseline"}
    approved = approve(source, design("randomized", FACTS), configuration)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unselected baseline balance was executed")

    monkeypatch.setattr(diagnostics.engine, "balance", forbidden)
    result = api.execute(approved, source)
    assert result.primary.status == "computed"
    assert not any(row.status == "failed" for row in result.diagnostics)
    assert next(row.status for row in result.diagnostics if
                row.computation_id == "baseline_balance") == "not_selected"


def test_changed_implementation_identity_invalidates_preflight(data: pl.DataFrame) -> None:
    assessment = assess(design("randomized", FACTS), draft(data))
    assert assessment.specification is not None
    readiness = api.preflight(assessment.specification, data)
    stale = readiness.model_copy(update={"implementation_hash": "f" * 64})
    with pytest.raises(BoundaryError):
        api.compile_plan(assessment.specification, stale)


def test_nonconverged_primary_is_retained_as_failure_without_retry(
        data: pl.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    from causal.analysis.methods.randomized import estimation

    approved = approve(data, design("randomized", FACTS), CONFIGURATION,
                        sensitivities=("sensitivity_covariance_profile",))
    original = estimation.RandomizedExperimentAdapter.fit
    calls = []

    def unconverged(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        result = original(self, *args, **kwargs)
        return result._replace(items=tuple(item.model_copy(update={"convergence": "not_converged"})
                                           for item in result.items))

    monkeypatch.setattr(estimation.RandomizedExperimentAdapter, "fit", unconverged)
    result = api.execute(approved, data)
    assert result.status == "failed" and result.primary.status == "failed"
    assert result.primary.estimates and result.primary.estimates[0].convergence == "not_converged"
    assert result.sensitivities[0].status == "blocked"
    assert calls == [1]


@pytest.mark.parametrize("method", ["randomized", "aipw"])
def test_outcome_cannot_be_used_as_its_own_adjustment_covariate(
        data: pl.DataFrame, method: str) -> None:
    configuration = (CONFIGURATION | {"estimator": "ancova", "precision_covariate_column": "y"}
                     if method == "randomized" else {
                         "method": "aipw", "estimand": "ate", "treatment_column": "arm",
                         "unit_column": "id", "treated_value": "treated", "comparator_value": "control",
                         "covariate_columns": ("y",)})
    fixed = design(method, FACTS if method == "randomized" else {"adjustment_set_pre_treatment": True})
    result = api.evaluate_candidate(candidate(fixed, draft(data) | {"configuration": configuration}))
    assert result.status == "rejected"
    assert any(issue.category == "contradictory_configuration" and
               "outcome" in issue.finding.lower() for issue in result.issues)


@pytest.mark.parametrize("method", ["randomized", "rdd"])
def test_configuration_guidance_exposes_selected_column_types_and_units(method: str) -> None:
    if method == "randomized":
        fixed, configuration = design("randomized", FACTS), CONFIGURATION
        expected = {("arm", "treatment", "categorical"), ("id", "unit_identifier", "identifier"),
                    ("y", "outcome", "numeric")}
        units = {"y": "points"}
    else:
        fixed = design("rdd", {"sharp_assignment": True, "cutoff": 5.0,
                               "assignment_direction": "above"}, outcome="child score")
        configuration = {"method": "rdd", "treatment_column": "assigned", "unit_column": "child id",
                         "treated_value": "1", "comparator_value": "0", "running_column": "eligibility score",
                         "cutoff": 5.0, "running_units": "index points"}
        expected = {("assigned", "treatment", "categorical"), ("child id", "unit_identifier", "identifier"),
                    ("eligibility score", "running_variable", "numeric"), ("child score", "outcome", "numeric")}
        units = {"child score": "points", "eligibility score": "index points"}
    found = api.retrieve_guidance(method, "configuration", draft={
        "design": fixed.model_dump(mode="json"), "configuration": configuration})
    assert {(row.name, row.role, row.kind) for row in found.required_columns} == expected
    assert {row.name: row.value for row in found.measurement_units} == units
    payload = json.loads(found.model_dump_json())
    assert {(row["name"], row["role"], row["kind"]) for row in payload["required_columns"]} == expected
    assert type(found).model_validate_json(found.model_dump_json()) == found

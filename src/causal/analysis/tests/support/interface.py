"""Small public-boundary inputs shared by method-owned execution tests."""
from __future__ import annotations

from typing import Any

from causal.analysis import interface as api
from causal.analysis.contracts import (
    ApprovedPlan,
    Assessment,
    FixedCandidate,
    FixedDesign,
    Method,
    Outcome,
    StudyFact,
)


def design(method: Method, facts: dict[str, Any], *, outcome: str = "y",
           units: str = "points") -> FixedDesign:
    return FixedDesign(
        reference="study-protocol:approved", content_hash="a" * 64, method=method,
        population="Prespecified eligible participants",
        outcome=Outcome(column=outcome, kind="continuous", units=units),
        facts=tuple(StudyFact(name=name, value=value, evidence=("protocol:assignment",))
                    for name, value in facts.items()))


def candidate(fixed: FixedDesign, proposal: dict[str, Any]) -> dict[str, Any]:
    """Declare the synthetic fixture's scientific context before its approval boundary.

    These references and timing assertions describe controlled test fixtures. They
    are not production defaults or a conversion of historical approvals.
    """
    configuration = dict(proposal.get("configuration", {}))
    roles = {
        "treatment_column": "treatment", "unit_column": "unit_identifier",
        "running_column": "running_variable", "covariate_column": "predetermined_covariate",
        "precision_covariate_column": "precision_covariate", "stratum_column": "stratum",
        "cluster_column": "cluster", "time_column": "time", "adoption_column": "adoption_time",
        "group_column": "group",
    }
    bindings = [{"role": role, "column": configuration[field],
                 "source_references": ("fixture:column-dictionary",),
                 "meaning": f"The fixture's prespecified {role}."}
                for field, role in roles.items() if configuration.get(field)]
    bindings += [{"role": "adjustment_covariates", "column": column,
                  "source_references": ("fixture:column-dictionary",),
                  "meaning": "A fixture baseline adjustment covariate."}
                 for column in configuration.get("covariate_columns", ())]
    bindings.append({"role": "outcome", "column": fixed.outcome.column,
                     "source_references": ("fixture:column-dictionary",),
                     "meaning": "The fixture's prespecified measured outcome."})
    scoped_facts = {
        "adjustment_set_pre_treatment": tuple(configuration.get("covariate_columns", ())),
        "precision_covariate_pre_treatment": tuple(filter(None, (
            configuration.get("precision_covariate_column"),))),
        "covariate_pre_treatment": tuple(filter(None, (configuration.get("covariate_column"),))),
    }
    facts = [fact.model_dump(mode="json") | {
        "support": "assumption" if fact.name.endswith("_assumption") else "fact",
        "scope": scoped_facts.get(fact.name, ())} for fact in fixed.facts]
    for name in ("precision_covariate_pre_treatment", "covariate_pre_treatment"):
        if scoped_facts[name] and name not in {fact.name for fact in fixed.facts}:
            facts.append({"name": name, "value": True, "evidence": ("fixture:baseline-timing",),
                          "support": "fact", "scope": scoped_facts[name]})
    estimands = {"randomized": "itt", "rdd": "late_at_cutoff", "did": "att_group_time_aggregate"}
    return {
        "method": fixed.method, "population": fixed.population,
        "outcome": fixed.outcome.model_dump(mode="json") | {
            "meaning": "The fixture's prespecified measured outcome."},
        "estimand": configuration.get("estimand", estimands.get(fixed.method)),
        "unit_grain": "One row per unit and period" if fixed.method == "did" else "One row per unit",
        "population_policy": "Preserve every fixture participant in the declared population",
        "missingness_policy": "Require complete observed analysis inputs",
        "bindings": bindings, "facts": facts, "configuration": configuration,
        "diagnostics": proposal.get("diagnostics", ()),
        "sensitivities": proposal.get("sensitivities", ()), "seed": proposal.get("seed", 0),
        "candidate_reference": "fixture-candidate:revision-1",
        "context_reference": "fixture:protocol", "source_dataset_reference": "fixture:source",
    }


def assess(fixed: FixedDesign, proposal: dict[str, Any]) -> Assessment:
    """Accept a complete fixture candidate, then bind its exact prepared frame."""
    accepted = FixedCandidate.from_candidate(candidate(fixed, proposal), reference="fixture:accepted")
    return api.assess_specification(accepted, {"dataset": proposal["dataset"]})


def approve(data: object, fixed: FixedDesign, configuration: dict[str, Any], *,
            sensitivities: tuple[str, ...] = (), diagnostics: tuple[str, ...] = ()) -> ApprovedPlan:
    assessment = assess(fixed, {
        "dataset": api.identify_data(data, "frozen-study").model_dump(mode="json"),
        "configuration": configuration, "sensitivities": sensitivities, "diagnostics": diagnostics, "seed": 17})
    assert assessment.status == "ready", assessment.issues
    specification = assessment.specification
    assert specification is not None
    checks = api.preflight(specification, data)
    assert checks.ready, checks.issues
    plan = api.compile_plan(specification, checks)
    return ApprovedPlan(plan=plan, approved_hash=plan.plan_hash, approved_by="reviewer",
                        approval_reference="approval:17")

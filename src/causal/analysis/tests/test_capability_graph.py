"""The capability graph guides the same scientific boundary that it enforces."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from causal.analysis import interface as api
from causal.analysis.common.candidate import CandidateDraft, CapabilityRequestError
from causal.analysis.tests.support.interface import candidate, design


def rdd(**configuration: Any) -> dict[str, Any]:
    return candidate(design("rdd", {
        "sharp_assignment": True, "cutoff": 0.0, "assignment_direction": "above"}), {
            "configuration": {
                "method": "rdd", "treatment_column": "assigned", "unit_column": "person",
                "treated_value": "1", "comparator_value": "0", "running_column": "score",
                "cutoff": 0.0, "running_units": "index points", **configuration}, "seed": 17})


def option(result: Any, suffix: str) -> Any:
    return next(row for row in result.options if row.node_id.endswith(suffix))


def requirement(result: Any, suffix: str) -> Any:
    return next(row for row in result.requirements if row.requirement_id.endswith(suffix))


def fact(proposal: dict[str, Any], name: str, value: Any) -> dict[str, Any]:
    changed = copy.deepcopy(proposal)
    next(row for row in changed["facts"] if row["name"] == name)["value"] = value
    return changed


def neighborhood(draft: Any, at: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    cursor = None
    while True:
        page = api.explore_capabilities(draft, at=at, limit=7, cursor=cursor)
        rows[page.at.node_id] = page.at
        rows.update((row.node_id, row) for row in page.nodes)
        cursor = page.coverage.next_cursor
        if cursor is None:
            return rows


def test_root_discovers_every_method_without_nominating_or_binding_anything() -> None:
    draft = CandidateDraft()
    before = draft.model_dump(mode="json")
    evaluation = api.evaluate_candidate(draft)
    root = api.explore_capabilities(draft)
    assert root.at.node_id == "analysis"
    assert root.status == evaluation.status == "needs_information"
    assert {row.node_id for row in root.nodes if row.type == "method"} == {
        f"method:{row.method}" for row in api.list_methods()}
    assert draft.model_dump(mode="json") == before
    assert draft.method is None and draft.population is None and draft.outcome is None
    assert any(target.field == "method" for row in evaluation.requirements
               for target in row.resolution_targets)


@pytest.mark.parametrize("method", ["randomized", "aipw", "did", "rdd"])
def test_every_installed_method_is_reachable_and_supplies_real_definition_context(method: str) -> None:
    draft = CandidateDraft(method=method)
    entry = api.explore_capabilities(draft, at=f"method:{method}")
    assert entry.at.type == "method" and entry.at.capability_version
    evaluation = api.evaluate_candidate(draft)
    assert evaluation.capability_version == entry.at.capability_version
    assert evaluation.status == "needs_information" and evaluation.role_slots
    assert any(row.role == "unit_identifier" and not row.bound_columns
               for row in evaluation.role_slots)
    for node_id in {row.requirement_id for row in evaluation.requirements} | {
            row.node_id for row in evaluation.options}:
        found = api.explore_capabilities(draft, at=node_id)
        assert found.at.node_id == node_id and found.at.description
        assert found.status == evaluation.status
    # Traverse declared links, including links back to shared requirements.
    seen: set[str] = set()
    pending = [f"method:{method}"]
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        for adjacent in neighborhood(draft, node_id):
            if adjacent.startswith((f"{method}:", "analysis:requirement:")) and adjacent not in seen:
                pending.append(adjacent)
    assert {row.requirement_id for row in evaluation.requirements} <= seen
    assert {row.node_id for row in evaluation.options} <= seen


def test_inspecting_a_different_family_does_not_change_nomination_or_global_evaluation() -> None:
    proposal = rdd()
    before = copy.deepcopy(proposal)
    expected = api.evaluate_candidate(proposal)
    other = api.explore_capabilities(proposal, at="method:aipw")
    assert other.at.node_id == "method:aipw"
    assert other.candidate_fingerprint == expected.candidate_fingerprint
    assert other.status == expected.status
    assert proposal == before and proposal["method"] == "rdd"


@pytest.mark.parametrize("value, expected", [(None, "unresolved"), (False, "violated"), (True, "satisfied")])
def test_rdd_false_unknown_and_true_are_distinct(value: Any, expected: str) -> None:
    proposal = fact(rdd(), "sharp_assignment", value)
    evaluation = api.evaluate_candidate(proposal)
    assert requirement(evaluation, ":sharp_assignment").status == expected
    branch = option(evaluation, ":sensitivity:half_bandwidth")
    assert branch.state == {"unresolved": "conditional", "violated": "blocked",
                            "satisfied": "available"}[expected]
    if value is not True:
        assert evaluation.status != "design_ready"
    if value is False:
        assert option(evaluation, ":option:kernel=triangular").state == "blocked"


def test_rdd_unknown_facts_cannot_be_replaced_with_an_assumption() -> None:
    proposal = rdd()
    assertion = next(row for row in proposal["facts"] if row["name"] == "sharp_assignment")
    assertion["support"] = "assumption"
    evaluation = api.evaluate_candidate(proposal)
    assert evaluation.status != "design_ready"
    assert requirement(evaluation, ":sharp_assignment").status != "satisfied"
    assert any(target.kind == "study_fact" for row in evaluation.requirements
               for target in row.resolution_targets if "sharp_assignment" in target.field)


def test_focused_branch_keeps_every_global_blocker_visible() -> None:
    proposal = fact(rdd(), "sharp_assignment", False)
    evaluation = api.evaluate_candidate(proposal)
    focused = api.explore_capabilities(proposal, at="rdd:sensitivity:half_bandwidth", limit=1)
    blockers = {row.requirement_id for row in evaluation.requirements
                if row.boundary == "design" and row.status in {"violated", "unresolved"}}
    assert focused.status == evaluation.status == "rejected"
    assert set(focused.blocker_references) == blockers
    assert focused.blocker_count == len(blockers)
    assert focused.candidate_fingerprint == evaluation.candidate_fingerprint


def test_revising_contradictory_assignment_reopens_branches_without_mutating_history() -> None:
    rejected = fact(rdd(), "sharp_assignment", False)
    before = copy.deepcopy(rejected)
    blocked = api.evaluate_candidate(rejected)
    corrected = fact(rejected, "sharp_assignment", True)
    reopened = api.evaluate_candidate(corrected)
    assert blocked.status == "rejected" and reopened.status == "design_ready"
    assert option(blocked, ":sensitivity:half_bandwidth").state == "blocked"
    assert option(reopened, ":sensitivity:half_bandwidth").state == "available"
    assert blocked.candidate_fingerprint != reopened.candidate_fingerprint
    assert rejected == before


def test_rdd_covariate_requires_scoped_timing_and_updates_every_dependent_branch() -> None:
    complete = rdd(covariate_column="baseline")
    absent = copy.deepcopy(complete)
    absent["facts"] = [row for row in absent["facts"] if row["name"] != "covariate_pre_treatment"]
    unknown = api.evaluate_candidate(absent)
    assert unknown.status == "needs_information"
    assert option(unknown, ":sensitivity:covariate_adjusted").state == "conditional"
    wrong_scope = copy.deepcopy(complete)
    next(row for row in wrong_scope["facts"] if row["name"] == "covariate_pre_treatment")["scope"] = ["other"]
    assert api.evaluate_candidate(wrong_scope).status != "design_ready"
    resolved = api.evaluate_candidate(complete)
    assert resolved.status == "design_ready"
    assert option(resolved, ":sensitivity:covariate_adjusted").state == "available"
    dependents = [row for row in resolved.options
                  if "rdd:requirement:covariate_timing" in row.dependencies]
    assert len(dependents) >= 2
    assert all(row.state == "available" for row in dependents)


def test_removing_an_adjustment_or_changing_cutoff_invalidates_selected_comparisons() -> None:
    proposal = rdd(covariate_column="baseline")
    proposal["sensitivities"] = ["covariate_adjusted"]
    assert api.evaluate_candidate(proposal).status == "design_ready"
    removed = copy.deepcopy(proposal)
    removed["configuration"]["covariate_column"] = None
    removed["bindings"] = [row for row in removed["bindings"] if row["role"] != "predetermined_covariate"]
    evaluation = api.evaluate_candidate(removed)
    selected = option(evaluation, ":sensitivity:covariate_adjusted")
    assert selected.selected and selected.invalidated and selected.state == "blocked"
    assert evaluation.status == "rejected"
    changed = copy.deepcopy(proposal)
    changed["configuration"]["cutoff"] = 1.0
    revised = api.evaluate_candidate(changed)
    assert revised.status == "rejected"
    assert option(revised, ":sensitivity:covariate_adjusted").state == "blocked"


def test_randomized_ancova_and_baseline_checks_resolve_from_the_same_timing_requirement() -> None:
    base = candidate(design("randomized", {"assignment_mechanism": "individual_randomized"}), {
        "configuration": {"method": "randomized", "estimator": "difference_in_means",
                          "treatment_column": "arm", "unit_column": "person",
                          "treated_value": "T", "comparator_value": "C"}})
    unadjusted = api.evaluate_candidate(base)
    assert unadjusted.status == "design_ready"
    assert option(unadjusted, ":option:estimator=ancova").state == "conditional"
    ancova = copy.deepcopy(base)
    ancova["configuration"]["estimator"] = "ancova"
    assert api.evaluate_candidate(ancova).status == "needs_information"
    ancova["configuration"]["precision_covariate_column"] = "baseline"
    conditional = api.evaluate_candidate(ancova)
    assert option(conditional, ":diagnostic:baseline_balance").state == "conditional"
    ancova["facts"].append({"name": "precision_covariate_pre_treatment", "value": True,
                            "evidence": ["fixture:baseline"], "scope": ["baseline"]})
    resolved = api.evaluate_candidate(ancova)
    assert resolved.status == "design_ready"
    assert option(resolved, ":diagnostic:baseline_balance").state == "available"


@pytest.mark.parametrize("count", [1, 17, 101])
def test_aipw_variable_cardinality_binds_columns_to_one_stable_role(count: int) -> None:
    columns = tuple(f"measurement {index} Ω" for index in range(count))
    proposal = candidate(design("aipw", {"adjustment_set_pre_treatment": True}), {
        "configuration": {"method": "aipw", "estimand": "att", "treatment_column": "arm",
                          "unit_column": "person", "treated_value": "T", "comparator_value": "C",
                          "covariate_columns": columns}})
    evaluated = api.evaluate_candidate(proposal)
    assert evaluated.status == "design_ready"
    slot = next(row for row in evaluated.role_slots if row.role == "adjustment_covariates")
    assert slot.bound_columns == columns and slot.maximum is None
    view = api.explore_capabilities(proposal, at=slot.node_id)
    assert view.at.node_id == "aipw:role:adjustment_covariates"
    assert not any(column in view.at.node_id for column in columns)
    proposal["configuration"]["estimand"] = None
    proposal["estimand"] = None
    assert api.evaluate_candidate(proposal).status == "needs_information"


def test_did_adoption_profiles_require_one_supported_schedule_and_allow_declared_assumptions() -> None:
    base = candidate(design("did", {"parallel_trends_assumption": True,
                                   "no_anticipation_assumption": True}), {
        "configuration": {"method": "did", "profile": "simultaneous",
                          "treatment_column": "treated", "unit_column": "person",
                          "time_column": "period", "adoption_time": 3.0}})
    simultaneous = api.evaluate_candidate(base)
    assert simultaneous.status == "design_ready"
    assert option(simultaneous, ":option:profile=staggered").state == "conditional"
    staggered = copy.deepcopy(base)
    staggered["configuration"]["profile"] = "staggered"
    assert api.evaluate_candidate(staggered).status != "design_ready"
    staggered["configuration"].pop("adoption_time")
    staggered["configuration"]["adoption_column"] = "first_treated"
    assert api.evaluate_candidate(staggered).status == "design_ready"
    staggered["configuration"]["adoption_time"] = 3.0
    assert api.evaluate_candidate(staggered).status == "rejected"


@pytest.mark.parametrize("setting", [{"kernel": "uniform"}, {"confidence_level": 0.90},
                                    {"cutoff": False}, {"arbitrary_option": 1}])
def test_explicit_unsupported_values_stay_rejected_and_are_never_silently_normalized(setting: Any) -> None:
    proposal = rdd(**setting)
    before = copy.deepcopy(proposal)
    evaluated = api.evaluate_candidate(proposal)
    assert evaluated.status == "rejected" and evaluated.issues
    assert proposal == before
    assert evaluated.configuration is None


def test_fixed_policies_and_mechanical_defaults_have_distinct_origins() -> None:
    evaluated = api.evaluate_candidate({"method": "rdd"})
    selections = {row.field: row for row in evaluated.selections}
    assert selections["configuration.kernel"].origin == "fixed_policy"
    assert selections["seed"].origin == "mechanical_default"
    assert "configuration.cutoff" not in selections or selections["configuration.cutoff"].value is None
    assert not any(row.field.startswith("facts.") and row.origin != "explicit"
                   for row in evaluated.selections)


@pytest.mark.parametrize("method", ["randomized", "aipw", "did", "rdd"])
def test_graph_fixed_policy_values_equal_the_method_owned_execution_policies(method: str) -> None:
    from causal.analysis.common.catalog import method_module

    definition = method_module(method).DEFINITION
    evaluated = api.evaluate_candidate({"method": method})
    selected = {row.node_id: row for row in evaluated.selections}
    assert definition.fixed_policies
    for policy in definition.fixed_policies:
        node_id = f"{method}:policy:{policy.id}"
        view = api.explore_capabilities({"method": method}, at=node_id)
        assert view.at.type == "policy" and view.at.permitted_values == (policy.value,)
        assert view.at.input_schema == {"const": policy.value}
        assert policy.description in view.at.description
        assert selected[node_id].origin == "fixed_policy" and selected[node_id].value == policy.value


def test_repeated_evaluation_is_deterministic_and_revision_references_change_identity() -> None:
    proposal = rdd()
    first = api.evaluate_candidate(proposal)
    assert first == api.evaluate_candidate(CandidateDraft.model_validate(proposal))
    reordered = dict(reversed(list(proposal.items())))
    reordered["configuration"] = dict(reversed(list(proposal["configuration"].items())))
    assert first == api.evaluate_candidate(reordered)
    revised = proposal | {"candidate_reference": "fixture-candidate:revision-2"}
    second = api.evaluate_candidate(revised)
    assert first.candidate_fingerprint != second.candidate_fingerprint
    assert first.requirements == second.requirements and first.options == second.options


def test_independent_assertions_can_be_resolved_in_either_order() -> None:
    proposal = rdd()
    earlier = copy.deepcopy(proposal)
    earlier["facts"] = list(reversed(earlier["facts"]))
    first, second = api.evaluate_candidate(proposal), api.evaluate_candidate(earlier)
    assert first.status == second.status == "design_ready"
    assert first.requirements == second.requirements and first.options == second.options
    # Input identities remain exact even when the two resolutions are equivalent.
    assert first.candidate_fingerprint != second.candidate_fingerprint


def test_bounded_navigation_reports_coverage_and_rejects_a_stale_candidate_cursor() -> None:
    proposal = rdd()
    first = api.explore_capabilities(proposal, at="method:rdd", limit=2)
    assert len(first.nodes) == first.coverage.returned_neighbors == 2
    assert first.coverage.truncated and first.coverage.next_cursor
    all_nodes = neighborhood(proposal, "method:rdd")
    assert len(all_nodes) - 1 == first.coverage.total_neighbors
    second = api.explore_capabilities(proposal, at="method:rdd", limit=2,
                                      cursor=first.coverage.next_cursor)
    assert second.coverage.offset == 2
    assert not {row.node_id for row in first.nodes} & {row.node_id for row in second.nodes}
    with pytest.raises(CapabilityRequestError):
        api.explore_capabilities(proposal | {"context_reference": "new-evidence"},
                                 at="method:rdd", limit=2, cursor=first.coverage.next_cursor)


@pytest.mark.parametrize("query", [{"at": "rdd:unknown-node"}, {"relations": ["invented"]},
                                    {"limit": 0}, {"cursor": "invalid"}])
def test_invalid_navigation_requests_have_typed_resolution_targets(query: Any) -> None:
    with pytest.raises(CapabilityRequestError) as raised:
        api.explore_capabilities({}, **query)
    assert raised.value.target.kind == "request" and raised.value.target.field


def test_discovery_exploration_and_evaluation_import_no_estimation_services() -> None:
    code = '''
import json, sys
from causal.analysis import interface as api
api.explore_capabilities()
for method in api.list_methods():
    api.evaluate_candidate({"method": method.method})
    api.explore_capabilities({"method": method.method}, at="method:" + method.method)
forbidden = {"polars", "numpy", "pandas", "scipy", "sklearn", "pyfixest", "rdrobust",
             "rddensity", "psycopg", "boto3", "langgraph", "langsmith", "google.genai"}
print(json.dumps(sorted(name for name in sys.modules if any(
    name == prefix or name.startswith(prefix + ".") for prefix in forbidden))))
'''
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run([sys.executable, "-c", code],
                            env=dict(os.environ, PYTHONPATH=str(root / "src")),
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == []


def test_design_checks_declare_execution_obligations_without_measuring_them() -> None:
    evaluated = api.evaluate_candidate(rdd())
    assert evaluated.status == "design_ready"
    assert any(row.boundary == "data_preflight" for row in evaluated.obligations)
    assert any(row.boundary == "execution" for row in evaluated.obligations)
    diagnostic = api.explore_capabilities(rdd(), at="rdd:diagnostic:density_manipulation_test")
    assert diagnostic.at.applicability_boundary == "design"
    assert diagnostic.at.measurement_boundary == "execution"


def test_graph_definitions_reject_dangling_dependencies_and_prerequisite_cycles() -> None:
    from causal.analysis.common.graph import build_graph, validate_definition
    from causal.analysis.methods.rdd.specification import DEFINITION

    first, second, *remaining = DEFINITION.requirements
    dangling = replace(DEFINITION, requirements=(replace(first, dependencies=("missing-rule",)),
                                                second, *remaining))
    with pytest.raises(ValueError, match="Dangling"):
        validate_definition(dangling)
    cycle = replace(DEFINITION, requirements=(replace(first, dependencies=(second.id,)),
                                             replace(second, dependencies=(first.id,)), *remaining))
    with pytest.raises(ValueError, match="cycles"):
        validate_definition(cycle)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_definition(replace(DEFINITION, requirements=DEFINITION.requirements + (first,)))
    graph = build_graph()
    identifiers = {row.node_id for row in graph.nodes}
    assert len(identifiers) == len(graph.nodes)
    assert all(edge.source in identifiers and edge.target in identifiers for edge in graph.edges)


def test_graph_version_change_invalidates_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    from causal.analysis.common import graph

    proposal = rdd()
    first = api.explore_capabilities(proposal, at="method:rdd", limit=1)
    original = graph.method_module

    def changed(method: str) -> Any:
        module = original(method)
        return (SimpleNamespace(DEFINITION=replace(module.DEFINITION, version="rdd.test-new-version"))
                if method == "rdd" else module)

    monkeypatch.setattr(graph, "method_module", changed)
    with pytest.raises(CapabilityRequestError):
        api.explore_capabilities(proposal, at="method:rdd", limit=1,
                                 cursor=first.coverage.next_cursor)


def test_new_method_definition_uses_the_common_navigation_and_evaluation_contract(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from causal.analysis.common import catalog, evaluation, graph
    from causal.analysis.common.definitions import (
        MethodDefinition,
        RequirementDefinition,
        fact_contradiction,
    )
    from causal.analysis.common.models import Model

    class SyntheticSpecification(Model):
        method: Literal["synthetic"] = "synthetic"
        estimand: Literal["fixture_contrast"] = "fixture_contrast"

    definition = MethodDefinition(
        "synthetic", "synthetic.test.v1", "Test capability", "Test shared extensibility.",
        SyntheticSpecification, ("supported_assignment",), (), (), (),
        requirements=(RequirementDefinition(
            "assignment", ("facts.supported_assignment",), "fact", "True",
            "The synthetic study assignment must be supported.",
            lambda config, facts: fact_contradiction(facts, "supported_assignment", True),
            fact_name="supported_assignment"),))
    original_method, original_diagnostics = catalog.method_module, catalog.diagnostics_module
    fake_module = SimpleNamespace(DEFINITION=definition)

    def methods(method: str) -> Any:
        return fake_module if method == "synthetic" else original_method(method)

    def diagnostics(method: str) -> Any:
        return SimpleNamespace() if method == "synthetic" else original_diagnostics(method)

    for module in (catalog, evaluation, graph, api):
        if hasattr(module, "METHODS"):
            monkeypatch.setattr(module, "METHODS", (*catalog.METHODS, "synthetic")
                                if "synthetic" not in catalog.METHODS else catalog.METHODS)
        if hasattr(module, "method_module"):
            monkeypatch.setattr(module, "method_module", methods)
        if hasattr(module, "diagnostics_module"):
            monkeypatch.setattr(module, "diagnostics_module", diagnostics)
    root = api.explore_capabilities()
    assert "method:synthetic" in {row.node_id for row in root.nodes}
    partial = api.evaluate_candidate({"method": "synthetic"})
    assert partial.status == "needs_information"
    proposal = {
        "method": "synthetic", "population": "Test units", "unit_grain": "One row per unit",
        "outcome": {"column": "response", "kind": "continuous", "units": "points"},
        "facts": [{"name": "supported_assignment", "value": True, "evidence": ["test:protocol"]}],
    }
    complete = api.evaluate_candidate(proposal)
    assert complete.status == "design_ready" and complete.capability_version == definition.version
    focused = api.explore_capabilities(proposal, at="synthetic:requirement:assignment")
    assert focused.at.description and focused.status == complete.status

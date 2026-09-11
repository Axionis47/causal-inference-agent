"""Invalid explicit inputs cannot become accepted through schema coercion."""
from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from causal.analysis import interface as api
from causal.analysis.common.candidate import CandidateDraft
from causal.analysis.common.evaluation import digest
from causal.analysis.common.graph import build_graph
from causal.analysis.contracts import fingerprint
from causal.analysis.tests.support.interface import candidate, design


def rdd_candidate() -> dict[str, Any]:
    return candidate(design("rdd", {
        "sharp_assignment": True, "cutoff": 0.0, "assignment_direction": "above"}), {
            "configuration": {
                "method": "rdd", "treatment_column": "t", "unit_column": "id",
                "treated_value": "1", "comparator_value": "0", "running_column": "score",
                "cutoff": 0.0, "running_units": "points"}, "seed": 17})


@pytest.mark.parametrize(("field", "value"), [
    ("polynomial_order", True), ("polynomial_order", 1.0),
    ("confidence_level", "0.95"), ("cutoff", False), ("cutoff", "0.0"),
])
def test_explicit_scalar_coercions_are_rejected_and_retained(field: str, value: Any) -> None:
    proposal = rdd_candidate()
    assert api.evaluate_candidate(proposal).status == "design_ready"
    proposal["configuration"][field] = value
    before = copy.deepcopy(proposal)
    evaluation = api.evaluate_candidate(proposal)
    assert evaluation.status == "rejected" and evaluation.configuration is None
    assert any(row.requirement_id == f"rdd:requirement:schema.{field}" and row.status == "violated"
               for row in evaluation.requirements)
    selected = next(row for row in evaluation.selections if row.field == f"configuration.{field}")
    assert selected.value == value and type(selected.value) is type(value)
    assert selected.origin == "explicit" and proposal == before
    if field == "polynomial_order":
        supported = next(row for row in evaluation.options if row.node_id == "rdd:option:polynomial_order=1")
        assert not supported.selected


@pytest.mark.parametrize("seed", [True, False, "17", 17.0, float("nan"), float("inf")])
def test_non_integer_seeds_return_rejected_feedback(seed: Any) -> None:
    proposal = rdd_candidate() | {"seed": seed}
    evaluation = api.evaluate_candidate(proposal)
    assert evaluation.status == "rejected"
    assert any(issue.field == "seed" for issue in evaluation.issues)
    assert evaluation.candidate_fingerprint == api.evaluate_candidate(proposal).candidate_fingerprint


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_configuration_has_stable_distinct_rejected_identity(value: float) -> None:
    proposal = rdd_candidate()
    proposal["configuration"]["cutoff"] = value
    evaluation = api.evaluate_candidate(proposal)
    assert evaluation.status == "rejected" and evaluation.configuration is None
    assert any(issue.field == "configuration.cutoff" for issue in evaluation.issues)
    assert evaluation.candidate_fingerprint == api.evaluate_candidate(proposal).candidate_fingerprint
    missing = copy.deepcopy(proposal)
    missing["configuration"]["cutoff"] = None
    assert evaluation.candidate_fingerprint != api.evaluate_candidate(missing).candidate_fingerprint
    exploration = api.explore_capabilities(proposal, at="rdd:decision:cutoff")
    assert exploration.status == "rejected"
    assert exploration.candidate_fingerprint == evaluation.candidate_fingerprint


def test_nonfinite_values_are_not_fingerprinted_as_null_or_each_other() -> None:
    payloads = [CandidateDraft(method="rdd", configuration={"cutoff": value})
                for value in (None, float("nan"), float("inf"), float("-inf"))]
    assert len({digest(payload) for payload in payloads}) == len(payloads)
    malformed = CandidateDraft(method="rdd").model_copy(update={"seed": math.nan})
    assert api.evaluate_candidate(malformed).status == "rejected"
    assert api.evaluate_candidate(malformed).candidate_fingerprint != digest(CandidateDraft(method="rdd"))


def test_invalid_assertion_value_returns_feedback_instead_of_fingerprint_failure() -> None:
    proposal = rdd_candidate()
    proposal["facts"][0]["value"] = math.nan
    evaluation = api.evaluate_candidate(proposal)
    assert evaluation.status == "rejected"
    assert any(issue.field.startswith("facts.0.value") for issue in evaluation.issues)
    assert evaluation.candidate_fingerprint == api.evaluate_candidate(proposal).candidate_fingerprint


def test_valid_candidate_fingerprint_remains_the_exact_public_contract_hash() -> None:
    proposal = CandidateDraft.model_validate(rdd_candidate())
    assert digest(proposal) == fingerprint(proposal)
    assert api.evaluate_candidate(proposal).candidate_fingerprint == fingerprint(proposal)
    assert api.evaluate_candidate(proposal.model_dump(mode="json")).candidate_fingerprint == fingerprint(proposal)


@pytest.mark.parametrize("value", [True, "0.0", float("nan"), float("inf")])
def test_guidance_reports_the_same_rejection_for_invalid_submitted_values(value: Any) -> None:
    proposal = rdd_candidate()
    proposal["configuration"]["cutoff"] = value
    evaluation = api.evaluate_candidate(proposal)
    guidance = api.retrieve_guidance("rdd", "configuration", proposal)
    assert guidance.evaluation.status == evaluation.status == "rejected"
    assert guidance.evaluation.candidate_fingerprint == evaluation.candidate_fingerprint


def test_compatibility_projection_does_not_discard_unknown_submitted_fields() -> None:
    proposal = {"design": {"candidate": rdd_candidate()}, "unimplemented_setting": True}
    evaluation = api.evaluate_candidate(proposal)
    assert evaluation.status == "rejected"
    assert any(issue.field == "unimplemented_setting" for issue in evaluation.issues)


def test_compatibility_projection_retains_scientific_revisions() -> None:
    proposal = {"design": {"candidate": rdd_candidate()}, "outcome": {"kind": "unsupported"}}
    evaluation = api.evaluate_candidate(proposal)
    assert evaluation.status == "rejected"
    assert any(issue.field == "outcome.kind" for issue in evaluation.issues)


@pytest.mark.parametrize("method", [None, "randomized", "aipw", "did", "rdd"])
def test_absent_inputs_remain_needs_information(method: str | None) -> None:
    evaluation = api.evaluate_candidate(CandidateDraft(method=method))
    assert evaluation.status == "needs_information"
    assert not any(row.status == "violated" for row in evaluation.requirements)


@pytest.mark.parametrize("method", ["randomized", "aipw", "did", "rdd"])
def test_every_option_dependency_is_a_retrievable_graph_requirement(method: str) -> None:
    graph = build_graph()
    requirements = {(edge.source, edge.target) for edge in graph.edges if edge.relation == "requires"}
    evaluation = api.evaluate_candidate(CandidateDraft(method=method))
    for option in evaluation.options:
        for dependency in option.dependencies:
            assert (option.node_id, dependency) in requirements

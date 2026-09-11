"""An invalid citation's quote can be repaired without reopening the underlying decision."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from causal.design import contracts, packs, validators
from causal.shared import agenttask
from causal.shared.envelope import TaskStatus
from causal.shared.validation import parse_strict
from tests.shared.test_agenttask import Harness, missing_requirement

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures/rct_intent_citation_repair.json"


def captured() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def setup(replies: list[dict[str, Any]], *, many: bool = False) -> Harness:
    case = captured()
    harness = Harness(replies[0], (TaskStatus.COMPLETE, TaskStatus.NEEDS_CONTEXT, TaskStatus.CONFLICT))
    harness.spec = replace(harness.spec, wall=3, correction_budget=len(replies) - 1)
    harness.gateway.replies = replies
    context = validators.ValidationContext(
        manifest=parse_strict(contracts.DesignContextManifestV1, case["manifest"]),
        rules=validators.load_validation_rules(ROOT / "registries/design-validation-rules.v1.json"),
        evidence_ids=frozenset(case["evidence"]), evidence_text=case["evidence"],
        templates=packs.load_requirement_templates(ROOT / "registries/context-requirements.v1.json"))
    harness.runner = replace(harness.runner, tasks={harness.spec.task_kind: harness.spec},
        context=lambda _: context, evidence=lambda _: case["evidence"],
        validate=validators.validate_result)
    if many:
        for reply in replies:
            reply["payload"] = {"items": [reply["payload"]]}
    return harness


def run(harness: Harness, *, many: bool = False) -> Any:
    return harness.runner.run(harness.state, harness.spec.task_kind, contracts.DesignIntentV1,
        scope_kind="design", scope_ids=(), parent_kinds=(),
        payload={"request": "saved intent citation repair"}, many=many)


def test_exact_saved_sequence_reproduces_failure_without_the_scoped_excerpt_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The old guard removed typed references from comparison but froze every source quote.
    monkeypatch.setattr(agenttask, "_citation_excerpt_paths", lambda *args, **kwargs: frozenset())
    harness = setup(captured()["decisions"])
    assert run(harness) is None
    assert len(harness.gateway.calls) == 3
    assert not harness.commits and not harness.upserts
    assert harness.state["corrections"]["bounded_draft:reference_repair_changed_semantics"] == 2
    assert harness.state["corrections"]["bounded_draft:unresolved_evidence"] == 3


@pytest.mark.parametrize("many", (False, True))
def test_exact_saved_second_response_repairs_citation_and_commits_once(many: bool) -> None:
    case = captured()
    first, second, third = case["decisions"]
    assert first == third
    before = first["payload"]["source_interpretations"][0]
    after = second["payload"]["source_interpretations"][0]
    assert not any(before["verbatim_excerpt"] in text for text in case["evidence"].values())
    assert after["verbatim_excerpt"] in case["evidence"][after["evidence_id"]]
    harness = setup(case["decisions"], many=many)
    assert run(harness, many=many) is not None
    assert len(harness.gateway.calls) == 2
    assert len(harness.commits) == len(harness.upserts) == 1
    expected = second["payload"]["items"][0] if many else second["payload"]
    assert harness.commits[0] == expected
    correction = harness.gateway.calls[1].payload["correction"]
    prefix = "/payload/items/0" if many else "/payload"
    assert correction["baseline_decision"] == first
    assert correction["editable_reference_paths"] == [f"{prefix}/source_interpretations/0/evidence_id"]
    assert correction["editable_excerpt_paths"] == [f"{prefix}/source_interpretations/0/verbatim_excerpt"]
    assert "Preserve fact_key, value and relation" in correction["instruction"]
    assert not any("reference_repair_changed" in key for key in harness.state["corrections"])


@pytest.mark.parametrize("change", (
    "fact_key", "value", "relation", "frame", "status", "requirements", "warnings", "conflicts",
    "valid_citation", "valid_quote", "row_deletion", "row_reorder",
))
def test_repair_does_not_reopen_semantics_or_valid_citations(change: str) -> None:
    case = captured()
    first, repaired = case["decisions"][:2]
    valid = deepcopy(repaired["payload"]["source_interpretations"][0])
    for decision in (first, repaired):
        decision["payload"]["source_interpretations"].append(deepcopy(valid))
    row = repaired["payload"]["source_interpretations"][0]
    match change:
        case "fact_key": row["fact_key"] = "comparator"
        case "value": row["value"] = "one_row_per_unit_period"
        case "relation": row["relation"] = "conflicting"
        case "frame": repaired["payload"]["population"]["name"] = "Different population"
        case "status":
            repaired["status"] = "conflict"
            repaired["conflicts"] = ["New conflict"]
        case "requirements":
            requirement = missing_requirement("design.unit_identity").model_dump(mode="json")
            requirement["registry_version"] = "context-requirements.v1"
            requirement["expected_answer_schema"] = "free-text.v1"
            repaired["missing_requirements"] = [requirement]
        case "warnings": repaired["warnings"] = ["Changed warning"]
        case "conflicts": repaired["conflicts"] = ["Changed conflict"]
        case "valid_citation":
            repaired["payload"]["source_interpretations"][1]["evidence_id"] = "ev:kaggle/dataset/description"
        case "valid_quote":
            repaired["payload"]["source_interpretations"][1]["verbatim_excerpt"] = "each row is one complete cable-system aggregate"
        case "row_deletion": repaired["payload"]["source_interpretations"].pop(0)
        case "row_reorder": repaired["payload"]["source_interpretations"].reverse()
    # The rows would be equal after citation repair; make reordering a real semantic change.
    if change == "row_reorder":
        first["payload"]["source_interpretations"][1]["relation"] = "corroborating"
        repaired["payload"]["source_interpretations"][0]["relation"] = "corroborating"
    harness = setup([first, repaired])
    assert run(harness) is None
    assert not harness.commits and not harness.upserts
    assert any("reference_repair_changed" in key for key in harness.state["corrections"])


def test_wrong_quote_keeps_original_freeze_until_citation_is_valid() -> None:
    case = captured()
    first, valid = case["decisions"][:2]
    bad_quote = deepcopy(valid)
    bad_quote["payload"]["source_interpretations"][0]["verbatim_excerpt"] = "unsupported quotation"
    harness = setup([first, bad_quote, valid])
    assert run(harness) is not None
    assert len(harness.commits) == len(harness.upserts) == 1
    assert harness.commits[0] == valid["payload"]
    third_feedback = harness.gateway.calls[2].payload["correction"]
    assert third_feedback["baseline_decision"] == first
    assert {issue["code"] for issue in third_feedback["issues"]} == {"evidence_quote_mismatch"}
    # The same quote failure must not permit a later, otherwise valid change in population.
    changed = deepcopy(valid)
    changed["payload"]["population"]["name"] = "Different population"
    blocked = setup([first, bad_quote, changed])
    assert run(blocked) is None
    assert not blocked.commits and not blocked.upserts


@pytest.mark.parametrize("field", ("evidence_id", "verbatim_excerpt"))
def test_invalid_replacement_still_must_pass_reference_and_quote_walls(field: str) -> None:
    first, repaired = captured()["decisions"][:2]
    repaired["payload"]["source_interpretations"][0][field] = "invented replacement"
    harness = setup([first, repaired])
    assert run(harness) is None
    assert not harness.commits and not harness.upserts
    expected = "unresolved_evidence" if field == "evidence_id" else "evidence_quote_mismatch"
    assert f"bounded_draft:{expected}" in harness.state["corrections"]

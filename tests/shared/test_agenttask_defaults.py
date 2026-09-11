"""Replay the exact RCT reference repair that omitted a valid schema default."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from causal.design.semantics import RoleEvidenceV1
from causal.shared.envelope import TaskStatus
from causal.shared.validation import parse_strict, shape_report
from tests.shared.test_agenttask import Harness


def captured() -> list[dict[str, Any]]:
    path = Path(__file__).parent / "fixtures/rct_role_evidence_reference_repair.json"
    return [row["decision"] for row in json.loads(path.read_text())["attempts"]]


def replay(decisions: list[dict[str, Any]], *, many: bool = False) -> tuple[Harness, Any]:
    original = deepcopy(decisions)
    if many:
        for decision in original:
            decision["payload"] = {"items": [decision["payload"]]}
    harness = Harness(original[0], (TaskStatus.COMPLETE,), second=original[1],
                      evidence=frozenset({"ua:context/text", "ev:doc/README.md"}))
    harness.gateway.replies.extend(original[2:])
    spec = replace(harness.spec, correction_budget=2)
    concepts = {value for row in decisions[0]["payload"]["edge_hypotheses"]
                for key, value in row.items() if key in {"source_concept_id", "target_concept_id"}}
    context = SimpleNamespace(
        concept_ids=concepts,
        manifest=SimpleNamespace(structural_inventory=(SimpleNamespace(column_name="turnout_rate"),)),
        packs=SimpleNamespace(all=lambda: (SimpleNamespace(method_id="randomized_experiment"),)))
    runner = replace(harness.runner, tasks={spec.task_kind: spec}, context=lambda _: context,
                     validate=lambda wall, kind, model, result, ctx: shape_report(model, result))
    completed = runner.run(harness.state, spec.task_kind, RoleEvidenceV1,
                           scope_kind="concept", scope_ids=decisions[0]["payload"]["assigned_scope"],
                           parent_kinds=(), payload={"request": "replay exact public RCT decisions"},
                           many=many)
    return harness, completed


@pytest.mark.parametrize("many", (False, True))
def test_exact_rct_reference_repair_with_omitted_schema_default_commits_second_attempt(
    many: bool,
) -> None:
    decisions = captured()
    assert decisions[2] == decisions[0]  # The old loop induced this regression to invalid IDs.
    assert "schema_version" not in decisions[1]["payload"]
    harness, completed = replay(decisions, many=many)
    assert completed is not None
    assert len(harness.gateway.calls) == 2
    assert len(harness.commits) == 1
    assert completed[0][0] == parse_strict(RoleEvidenceV1, decisions[1]["payload"])
    assert harness.commits[0] == decisions[1]["payload"]  # No harness rewrite of the draft.
    assert not any("reference_repair_changed" in key for key in harness.state["corrections"])


@pytest.mark.parametrize("mutation,code", (
    ("mechanism", "reference_repair_changed_semantics"),
    ("valid_reference", "reference_repair_changed_unrelated_reference"),
    ("wrong_schema", "shape_invalid"),
    ("missing_required", "shape_invalid"),
))
def test_omitted_default_does_not_relax_semantic_reference_or_shape_guards(
    mutation: str, code: str,
) -> None:
    decisions = captured()
    payload = decisions[1]["payload"]
    if mutation == "mechanism":
        payload["edge_hypotheses"][0]["mechanism_summary"] = "A different scientific mechanism"
    elif mutation == "valid_reference":
        payload["edge_hypotheses"][0]["supporting_evidence_ids"].reverse()
    elif mutation == "wrong_schema":
        payload["schema_version"] = "role-evidence.v99"
    else:
        del payload["competing_mechanisms"]
    harness, completed = replay(decisions)
    assert completed is None
    assert not harness.commits
    assert harness.state["corrections"][f"bounded_draft:{code}"] == 2

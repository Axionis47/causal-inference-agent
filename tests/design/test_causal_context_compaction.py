"""Lossless prompt compaction replayed against the exact truncated public RCT request."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causal.design.compile import load_task_table, render_prompt
from causal.shared.agenttask import evidence_availability, evidence_block

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/shared/fixtures/rct_causal_context_truncated_request.json"


def captured() -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    request = json.loads(FIXTURE.read_text())
    sections = dict(part.split("\n", 1) for part in request["prompt"].split("\n\n## ")[1:])
    evidence: dict[str, str] = {}
    key = ""
    for line in sections["allowed_evidence"].splitlines():
        if line.startswith("  "):
            evidence[key] += ("\n" if evidence[key] else "") + line[2:]
        else:
            key = line.removesuffix(":")
            evidence[key] = ""
    return request, sections, evidence


def compact_replay() -> tuple[str, dict[str, object]]:
    request, sections, evidence = captured()
    names = ("design_intent", "hypotheses", "measurement_map", "prerequisite_context")
    payload = {name: json.loads(sections[name]) for name in names}
    spec = load_task_table(ROOT / "registries/design-tasks.v1.json")["causal_context"]
    rendered = render_prompt(spec, ROOT, payload)
    for name in ("reference_catalog", "allowed_evidence", "parent_artifacts", "registered_requirement_ids"):
        value = evidence_block(evidence, group_identical=True) if name == "allowed_evidence" else sections[name]
        rendered += f"\n\n## {name}\n{value}"
    return rendered, {"before_characters": len(request["prompt"]),
                      "after_characters": len(rendered),
                      "evidence_ids": len(evidence), "payload": payload}


def test_exact_rct_prompt_groups_every_evidence_id_without_changing_any_supplied_text() -> None:
    request, sections, evidence = captured()
    assert evidence_block(evidence) == sections["allowed_evidence"]
    grouped = json.loads(evidence_block(evidence, group_identical=True))
    restored = {key: row["text"] for row in grouped for key in row["evidence_ids"]}
    assert restored == evidence
    assert sum(len(row["evidence_ids"]) for row in grouped) == len(evidence)
    assert {key: row["availability"] for row in grouped for key in row["evidence_ids"]} == (
        evidence_availability(evidence))
    assert len(evidence_block(evidence, group_identical=True)) < len(sections["allowed_evidence"]) * .65
    assert request["task_kind"] == "causal_context"


def test_exact_rct_context_preserves_every_hypothesis_fact_and_reference_with_reduced_prompt() -> None:
    before, sections, _ = captured()
    after, metrics = compact_replay()
    revised = dict(part.split("\n", 1) for part in after.split("\n\n## ")[1:])
    for name in ("design_intent", "hypotheses", "measurement_map", "prerequisite_context"):
        assert json.loads(revised[name]) == json.loads(sections[name])
        assert "\n" not in revised[name]
    for name in ("reference_catalog", "parent_artifacts", "registered_requirement_ids"):
        assert revised[name] == sections[name]
    assert len(after) < len(before["prompt"]) * .8
    assert metrics["after_characters"] < metrics["before_characters"]
    assert "one short sentence" in after
    assert "rejected relationships" in after


def test_compaction_preserves_empty_and_withheld_evidence_statuses() -> None:
    evidence = {f"ev:{index}": "x" * 3000 for index in range(10)} | {"ev:empty": ""}
    grouped = json.loads(evidence_block(evidence, group_identical=True))
    observed = {key: row["availability"] for row in grouped for key in row["evidence_ids"]}
    assert observed == evidence_availability(evidence)
    assert "withheld" in observed.values()
    assert "empty" in observed.values()

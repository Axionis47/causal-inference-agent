"""Reject conflicting edge identity before committing, using the real correction loop."""

from __future__ import annotations

from dataclasses import replace

from causal.design.semantics import CausalContextV1, GraphAlternativeV1
from causal.design.validators import validate_result
from causal.shared.envelope import EpistemicStatus, TaskStatus
from tests.design import test_validators as fx
from tests.shared.test_agenttask import Harness, result


def test_conflicting_alternative_identity_is_corrected_before_graph_commit() -> None:
    forward = fx.edge("e:stratum_assignment", "c-x", "c-treat", EpistemicStatus.EVIDENCED)
    reverse = fx.edge("e:stratum_assignment", "c-treat", "c-x", EpistemicStatus.DISPUTED)
    alternative = GraphAlternativeV1(
        alternative_id="alt:reversed", label="reversed stratum relationship",
        edges=(forward, reverse))
    rejected = fx.graph(forward, alternatives=(alternative,))
    corrected = fx.graph(forward, alternatives=(alternative.model_copy(update={"edges": (forward,)}),))
    harness = Harness(result(payload=rejected.canonical_payload()), (TaskStatus.COMPLETE,),
                      second=result(payload=corrected.canonical_payload()))
    spec = replace(harness.spec, task_kind="causal_context", output_artifact_type="CausalContext",
                    output_schema_version="causal-context.v1", wall=5)
    runner = replace(harness.runner, tasks={spec.task_kind: spec}, tools={spec.task_kind: ()},
                      evals={spec.task_kind: ("EV-TEST-001",)}, validate=validate_result,
                      context=lambda _: fx.context())
    completed = runner.run(harness.state, spec.task_kind, CausalContextV1, scope_kind="design",
                           scope_ids=(), parent_kinds=(), payload={"request": "causal graph"})
    assert completed is not None
    assert harness.commits == [corrected.canonical_payload()]
    assert len(harness.gateway.calls) == 2
    correction = harness.gateway.calls[1].payload["correction"]
    assert isinstance(correction, dict)
    assert correction["failing_payload"] == rejected.canonical_payload()
    issues = correction["issues"]
    assert isinstance(issues, list)
    assert {issue["code"] for issue in issues} == {
        "duplicate_graph_edge_id", "conflicting_graph_edge_identity"}
    assert all(issue["json_path"] == "/alternatives/0/edges/1/edge_id" for issue in issues)

"""Regression checks for releasing scientific evidence without upstream reporting."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from causal.analysis.integration import contracts as ec
from causal.analysis.integration import harness as eh
from causal.analysis.integration import nodes, walls
from causal.analysis.integration.tests.support.plans import PACK, manifest, plan, ref
from causal.analysis.integration.tests.support.walls import RULES, context, diagnostic, mask, result
from causal.shared.canonical import content_hash


class MemoryNodes(nodes.EstimationNodes):
    def __init__(self) -> None:
        super().__init__(SimpleNamespace(  # type: ignore[arg-type]
            rules=RULES, packs=SimpleNamespace(get=lambda *_: PACK),
            products=SimpleNamespace(load_envelope=lambda key: key)))
        self.saved: dict[str, Any] = {}

    def emit(self, *args: Any, **kwargs: Any) -> None:
        pass

    def commit(self, state: Any, kind: str, payload: Any, parents: Any) -> Any:
        self.saved[kind] = payload
        digest = content_hash(payload)
        state["artifacts"][kind] = kind
        state["hashes"][kind] = digest
        return SimpleNamespace(artifact_id=kind, content_hash=digest)

    def payload(self, artifact_id: str) -> dict[str, Any]:
        return self.saved[artifact_id]

    def parents(self, state: Any, *kinds: str) -> tuple[Any, ...]:
        return tuple(kinds)


def prepared() -> tuple[MemoryNodes, Any]:
    run = MemoryNodes()
    state = eh.new_state("analysis", "stage", 1, {})
    for kind, payload in (("EstimationContextManifest", manifest()),
                          ("EstimationPlan", plan()), ("PrimaryAnalysisResult", result())):
        run.commit(state, kind, payload.canonical_payload(), ())
    run.commit(state, "NumericalEnvironmentManifest", {"environment": "frozen"}, ())
    run.commit(state, "EstimationEvidenceBundle", {"kind": "sensitivity"}, ())
    run.bundles = {"diagnostic": ref("diagnostics"), "sensitivity": ref("sensitivities")}
    run.frozen = {"masks": (mask(PACK.allowed_mask_rule_ids[0]),),
                  "mask_refs": (ref("mask"),)}
    run.denominators = {"row": 100, "unit": 100}
    return run, state


def test_numerical_close_preserves_measurements_without_claim_or_figure_builders() -> None:
    run, state = prepared()

    def forbidden(*args: Any) -> Any:
        raise AssertionError("reporting must not run inside analysis")

    run.adapter = SimpleNamespace(figures=forbidden)  # type: ignore[assignment]
    run.harvest = {"diagnostic": {"smd": 0.25, "unavailable": float("nan")}}
    assert run._finalise(state) == ("complete", None)
    support = run.saved["AnalysisSupportingData"]
    assert support["measurements"] == {"diagnostic": {"smd": 0.25, "unavailable": None}}
    assert support["unavailable_values"] == ["diagnostic/unavailable"]
    bundle = ec.NumericalBundleV1.model_validate_json(json.dumps(
        run.saved["NumericalBundle"]))
    assert bundle.primary_result.artifact_id == "PrimaryAnalysisResult"
    assert bundle.evidence_bundles == (ref("diagnostics"), ref("sensitivities"))
    assert {"ClaimJudgment", "JudgmentCeiling", "FigureDataArtifact"}.isdisjoint(run.saved)
    assert not {"claim_judgment", "judgment_ceiling", "capacity_report"} & set(
        ec.NumericalBundleV1.model_fields)


def test_a_visual_capacity_failure_cannot_prevent_numerical_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, state = prepared()

    assert run.plan_node(state)["phase"] == "estimate"


def test_failed_diagnostics_are_accounted_for_but_duplicates_and_extras_are_not() -> None:
    wanted = next(iter(plan().required_diagnostics))
    failed = diagnostic(wanted, "failed")
    scoped = plan().model_copy(update={"required_diagnostics": {wanted: failed.severity}})
    assert walls.wall(9, context(plan=scoped, diagnostics=(failed,))).passed
    assert not walls.wall(9, context(plan=scoped, diagnostics=(failed, failed))).passed
    extra = failed.model_copy(update={"diagnostic_id": "unrequested"})
    assert not walls.wall(9, context(plan=scoped, diagnostics=(failed, extra))).passed

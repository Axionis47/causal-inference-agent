"""Numerical provenance is shared; each caller supplies its own permitted data."""

from __future__ import annotations

import pytest

from causal_agent.explore.contracts import Step
from causal_agent.explore.diagnostics import DiagnosticTools
from causal_agent.lane.episode import EpisodeLog, Fact


def tools(tmp_path, *, required=True):
    return DiagnosticTools(csv=tmp_path / "table.csv", out_dir=tmp_path, timeout=30, stdout_tail=4000, required=required)


def test_finite_numerical_probes_are_required_and_bad_evidence_cannot_be_cited(tmp_path):
    diagnostic = tools(tmp_path)
    log = EpisodeLog(node="diagnostic", visit="evidence-test")
    assert diagnostic.completion_errors(log)
    assert diagnostic.evidence_errors([], log)
    cases = [
        (False, "printed before failing", {"n": 1000}),
        (True, "column names and labels", None),
        (True, "arms checked", {"rank_deficiency": 0}),
        (True, "", {"weight": float("inf")}),
        (True, "", {"weight": float("nan")}),
        (True, "", {}),
    ]
    for number, (ok, stdout, facts) in enumerate(cases, 1):
        diagnostic.steps.append(Step(n=number, ok=ok, stdout=stdout, stderr="" if ok else "failure", files=[], facts=facts, kind="docker"))
        log.facts.append(Fact(node=log.node, visit=log.visit, n=number, tool="run_code", args={}, text=stdout))
    log.calls = len(cases)
    failed, schema_only, successful, infinite, nan, empty = [fact.address for fact in log.facts]
    assert diagnostic.evidence_errors(["probe:diagnostic.unknown.1"], log)
    for address in (failed, schema_only, infinite, nan, empty):
        assert any("missing finite numerical facts" in error for error in diagnostic.evidence_errors([address], log))
    assert diagnostic.evidence_errors([successful], log) == []
    assert diagnostic.completion_errors(log) == []
    assert diagnostic.evidence_errors([successful, failed], log)


@pytest.mark.parametrize("attempted", [False, True])
def test_optional_diagnostics_allow_no_code_but_attempts_cannot_be_ignored(tmp_path, attempted):
    diagnostic = tools(tmp_path, required=False)
    log = EpisodeLog(node="diagnostic", calls=int(attempted))
    assert bool(diagnostic.completion_errors(log)) is attempted
    assert bool(diagnostic.evidence_errors([], log)) is attempted


def test_successful_optional_code_must_be_cited(tmp_path):
    diagnostic = tools(tmp_path, required=False)
    log = EpisodeLog(node="diagnostic", calls=1)
    diagnostic.steps.append(Step(n=1, ok=True, stdout="", stderr="", files=["facts.json"], facts={"rows": 1000}, kind="docker"))
    log.facts.append(Fact(node=log.node, visit=log.visit, n=1, tool="run_code", text="rows=1000"))
    assert diagnostic.completion_errors(log) == []
    assert diagnostic.evidence_errors([], log)
    assert diagnostic.evidence_errors([log.facts[0].address], log) == []

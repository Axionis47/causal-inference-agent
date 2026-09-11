"""Authoring, repair, independent review and release against immutable in-memory artifacts."""
from __future__ import annotations

import hashlib
import io
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from causal.post_analysis import graph, review
from causal.post_analysis.contracts import EvidencePacket, ReportDraft
from causal.post_analysis.store import Store
from causal.post_analysis.visualization.contracts import Column, DataTable
from causal.shared.canonical import canonical_bytes
from causal.shared.contracts import ArtifactRef
from causal.shared.events import EventEmitter
from causal.shared.persistence import PersistenceError

PNG = b"\x89PNG\r\n\x1a\nexact-frozen-page-preview"


class MemoryObjects:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def put_if_absent(self, digest: str, data: bytes) -> str:
        locator = "objects/" + digest
        assert self.data.setdefault(locator, data) == data
        return locator

    def get(self, locator: str) -> bytes:
        return self.data[locator]


class MemoryProducts:
    def __init__(self) -> None:
        self.envelopes: dict[str, Any] = {}

    def load_envelope(self, artifact_id: str) -> Any:
        return self.envelopes[artifact_id]


class MemoryConnection:
    """Execute the real Store reservation transaction against a persistent test record."""
    def __init__(self) -> None:
        self.record: dict[str, int] = {}

    def transaction(self) -> Any:
        return nullcontext()

    def execute(self, query: str, parameters: tuple[Any, ...]) -> Any:
        if query.startswith("SELECT"):
            return SimpleNamespace(fetchone=lambda: (dict(self.record),))
        assert query.startswith("UPDATE presentation.runs SET run_record")
        self.record = json.loads(parameters[0])
        return None


class MemoryStore(Store):
    def __init__(self, deps: Any) -> None:
        super().__init__(deps, "analysis-1", "post-run-1")
        self.committed: list[tuple[str, ArtifactRef, tuple[ArtifactRef, ...]]] = []

    def commit(self, kind: str, body: dict[str, Any], parents: tuple[ArtifactRef, ...]) -> ArtifactRef:
        for parent in parents:
            self.read(parent)
        raw = canonical_bytes({"schema_version": kind + ".v1", **body})
        digest = hashlib.sha256(raw).hexdigest()
        ref = ArtifactRef(artifact_id=kind + ":" + digest, content_hash=digest)
        self.deps.products.envelopes[ref.artifact_id] = SimpleNamespace(
            analysis_id=self.analysis_id, content_hash=digest,
            payload_locator=self.deps.objects.put_if_absent(digest, raw))
        self.committed.append((kind, ref, parents))
        return ref


class ScriptedGateway:
    def __init__(self, script: list[tuple[str, Any]]) -> None:
        self.script = script
        self.calls: list[dict[str, Any]] = []

    def invoke(self, envelope: Any, prompt: str, schema: Any, **kwargs: Any) -> Any:
        role, response = self.script[min(len(self.calls), len(self.script) - 1)]
        assert envelope.task_kind == "post_analysis_" + role
        self.calls.append({"role": role, "payload": envelope.payload,
                           "prompt": prompt, "images": kwargs.get("images", ())})
        body = response(envelope.payload) if callable(response) else response
        return SimpleNamespace(text=body if isinstance(body, str) else json.dumps(body))


def action(tool: str, arguments: Any = None) -> dict[str, Any]:
    return {"tool": tool, "arguments": arguments or {},
            "decision_summary": "This action addresses the available evidence."}


def report(payload: dict[str, Any], *, title: str = "Causal findings") -> dict[str, Any]:
    return {"title": title, "sections": [{"title": "Effect and diagnostic limitations",
        "statements": [
            {"text": "The estimate is uncertain.",
             "citations": [{"evidence_id": "primary", "selector": "/estimate"}]},
            {"text": "The failed overlap check qualifies interpretation.",
             "citations": [{"evidence_id": "diagnostic:overlap", "selector": "/status"}]}],
        "visual_ids": list(payload["visuals"])}],
        "coverage": {"primary": "Discussed in effect section.",
                     "diagnostic:overlap": "Failure discussed as a limitation."}}


def reviewed(verdict: str) -> dict[str, Any]:
    return {"verdict": verdict, "issues": ["Clarify the limitation."] if verdict == "revise" else [],
            "decision_summary": "The final pages and the cited evidence were inspected."}


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    deps = SimpleNamespace(objects=MemoryObjects(), products=MemoryProducts(),
        conn=MemoryConnection(), registry=None, emitter=EventEmitter(io.StringIO()),
        clock=lambda: datetime(2026, 9, 10, tzinfo=UTC), gateway=None,
        checkpointer=InMemorySaver(), tracer=None, render_root=tmp_path,
        max_calls=16, max_reviews=3)
    store = MemoryStore(deps)
    outcome = store.commit("AnalysisOutcome", {"status": "complete"}, ())
    evidence = {"primary": {"estimate": 1.25}, "diagnostic:overlap": {"status": "failed"}}
    table = DataTable(table_id="effect", source=outcome, selector="/primary",
        columns=(Column(name="effect", label="Effect", kind="quantitative", quantity="effect"),),
        rows=((1.25,),))
    packet = EvidencePacket("analysis-1", {"outcome": outcome}, evidence,
                            {"effect": table}, tuple(evidence))
    packet_holder = [packet]
    monkeypatch.setattr(graph.Nodes, "packet", property(lambda self: packet_holder[0]))

    def build_report(draft: ReportDraft, visuals: Any, directory: Path) -> dict[str, Any]:
        # Isolate graph authority/revision behavior from the separately tested page renderer.
        directory.mkdir(parents=True, exist_ok=True)
        objects = {"report.html": directory / "report.html", "page-1.png": directory / "page-1.png"}
        objects["report.html"].write_text(draft.model_dump_json())
        objects["page-1.png"].write_bytes(PNG + draft.title.encode())
        return {"objects": {key: str(path) for key, path in objects.items()},
                "object_hashes": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for key, path in objects.items()}, "preview_keys": ["page-1.png"]}

    module = ModuleType("causal.post_analysis.presentation.build")
    module.build_report = build_report  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return SimpleNamespace(store=store, deps=deps, outcome=outcome, packet=packet_holder,
        config={"configurable": {"thread_id": store.thread_id}, "recursion_limit": 100})


def run(harness: Any, script: list[tuple[str, Any]]) -> Any:
    harness.deps.gateway = ScriptedGateway(script)
    return graph.build_graph(harness.store, harness.outcome).invoke(
        {"outcome": harness.outcome.model_dump(mode="json"), "status": "pending"}, harness.config)


def visual_action() -> dict[str, Any]:
    return action("render_visual", {"table_id": "effect", "kind": "table", "title": "Effect"})


def test_author_tool_review_revision_releases_only_the_exact_passed_export(harness: Any) -> None:
    state = run(harness, [
        ("author", visual_action()),
        ("author", lambda payload: action("write_report", report(payload))),
        ("author", action("submit")), ("review", reviewed("revise")),
        ("author", lambda payload: action("write_report", report(payload, title="Qualified findings"))),
        ("author", action("submit")), ("review", reviewed("pass")),
    ])
    assert state["status"] == "complete"
    bundle = harness.store.read(state["bundle"])
    passed_review = harness.store.read(bundle["review"])
    assert passed_review["passed"] is True
    assert passed_review["binding"] == review.review_binding(
        bundle["context"], bundle["draft"], bundle["visuals"], bundle["export"])
    assert harness.store.read(bundle["draft"])["title"] == "Qualified findings"
    calls = harness.deps.gateway.calls
    assert calls[4]["payload"]["observation"]["issues"] == ("Clarify the limitation.",)
    assert calls[3]["images"][0].data == PNG + b"Causal findings"
    assert calls[6]["images"][0].data == PNG + b"Qualified findings"
    assert harness.deps.conn.record == {"calls": 7, "reviews": 2}
    assert sum(kind == "PostAnalysisBundle" for kind, _, _ in harness.store.committed) == 1
    context = harness.store.read(bundle["context"])
    assert context["sources"]["outcome"] == harness.outcome.model_dump(mode="json")
    assert harness.store.read(harness.outcome)["status"] == "complete"


@pytest.mark.parametrize("defect", ["missing_diagnostic", "unknown_citation", "bad_pointer"])
def test_invalid_evidence_coverage_and_citations_do_not_commit_a_draft(harness: Any, defect: str) -> None:
    def invalid(payload: dict[str, Any]) -> dict[str, Any]:
        body = report(payload)
        if defect == "missing_diagnostic":
            del body["coverage"]["diagnostic:overlap"]
        else:
            citation = body["sections"][0]["statements"][0]["citations"][0]
            citation.update({"evidence_id": "invented"} if defect == "unknown_citation"
                            else {"selector": "/does-not-exist"})
        return action("write_report", body)

    state = run(harness, [("author", visual_action()), ("author", invalid),
                          ("author", action("stop", {"reason": "Cannot support this report."}))])
    assert state["status"] == "incomplete"
    assert harness.deps.gateway.calls[2]["payload"]["observation"]["issues"]
    assert not any(kind in {"PostAnalysisDraft", "PostAnalysisBundle"}
                   for kind, _, _ in harness.store.committed)


def test_malformed_model_actions_consume_the_durable_budget_and_stop(harness: Any) -> None:
    harness.deps.max_calls = 3
    state = run(harness, [("author", "not valid JSON")])
    assert state["status"] == "incomplete"
    assert state["error_code"] == "post_analysis_budget_exhausted"
    assert len(harness.deps.gateway.calls) == harness.deps.conn.record["calls"] == 3
    assert "invalid_action" in harness.deps.gateway.calls[1]["payload"]["observation"]["error"]
    assert not state.get("bundle")


def test_artifact_commit_failure_stops_without_spending_calls_on_model_repairs(harness: Any) -> None:
    commit = harness.store.commit

    def unavailable(kind: str, body: dict[str, Any], parents: tuple[ArtifactRef, ...]) -> ArtifactRef:
        if kind == "PostAnalysisVisual":
            raise PersistenceError("artifact transaction was rejected", "artifact_commit_failed")
        return commit(kind, body, parents)

    harness.store.commit = unavailable
    harness.deps.max_calls = 3
    state = run(harness, [("author", visual_action())])
    assert state["status"] == "incomplete" and state["error_code"] == "artifact_commit_failed"
    assert len(harness.deps.gateway.calls) == harness.deps.conn.record["calls"] == 1
    assert not state.get("bundle")


def test_review_budget_exhaustion_never_turns_an_unreviewed_revision_into_a_pass(harness: Any) -> None:
    harness.deps.max_reviews = 1
    state = run(harness, [
        ("author", visual_action()),
        ("author", lambda payload: action("write_report", report(payload))),
        ("author", action("submit")), ("review", reviewed("revise")),
        ("author", lambda payload: action("write_report", report(payload, title="Revised"))),
        ("author", action("submit")), ("review", reviewed("pass")),
    ])
    assert state["status"] == "incomplete"
    assert state["error_code"] == "post_analysis_budget_exhausted"
    assert harness.deps.conn.record == {"calls": 6, "reviews": 1}
    assert len(harness.deps.gateway.calls) == 6
    assert not state.get("bundle")


def test_checkpoint_resume_retains_committed_visual_and_spent_calls(harness: Any) -> None:
    harness.deps.gateway = ScriptedGateway([
        ("author", visual_action()),
        ("author", lambda payload: action("write_report", report(payload))),
        ("author", action("submit")), ("review", reviewed("pass")),
    ])
    compiled = graph.build_graph(harness.store, harness.outcome)
    paused = compiled.invoke(
        {"outcome": harness.outcome.model_dump(mode="json"), "status": "pending"},
        harness.config, interrupt_after=["tools"])
    assert paused["visuals"] and not paused.get("draft")
    assert harness.deps.conn.record == {"calls": 1, "reviews": 0}
    assert compiled.get_state(harness.config).next == ("agent",)
    recovered_store = MemoryStore(harness.deps)
    recovered = graph.build_graph(recovered_store, harness.outcome).invoke(None, harness.config)
    assert recovered["status"] == "complete"
    assert recovered["visuals"] == paused["visuals"]
    assert not any(kind == "PostAnalysisVisual" for kind, _, _ in recovered_store.committed)
    assert harness.deps.conn.record == {"calls": 4, "reviews": 1}
    assert len(harness.deps.gateway.calls) == 4


def passed_state(harness: Any) -> dict[str, Any]:
    return run(harness, [("author", visual_action()),
        ("author", lambda payload: action("write_report", report(payload))),
        ("author", action("submit")), ("review", reviewed("pass"))])


def test_changed_draft_invalidates_review_and_cannot_release(harness: Any) -> None:
    state = passed_state(harness)
    changed = harness.store.commit("PostAnalysisDraft",
        {**harness.store.read(state["draft"]), "title": "A changed report"}, ())
    before = len(harness.store.committed)
    with pytest.raises(ValueError, match="stale_or_failed_review"):
        graph.Nodes(harness.store, harness.outcome).release(
            {**state, "draft": changed.model_dump(mode="json")})
    assert len(harness.store.committed) == before


def test_tampered_export_bytes_and_changed_upstream_sources_block_release(harness: Any) -> None:
    state = passed_state(harness)
    export = harness.store.read(state["export"])
    locator = export["objects"]["report.html"]
    original = harness.deps.objects.get(locator)
    harness.deps.objects.data[locator] = b"a different report"
    with pytest.raises(ValueError, match="export object changed"):
        graph.Nodes(harness.store, harness.outcome).release(state)
    harness.deps.objects.data[locator] = original
    changed = harness.store.commit("AnalysisOutcome", {"status": "changed"}, ())
    harness.packet[0] = replace(harness.packet[0], sources={"outcome": changed})
    with pytest.raises(ValueError, match="source_binding_changed"):
        graph.Nodes(harness.store, harness.outcome).release(state)


def test_artifact_reference_and_stored_bytes_must_both_match(harness: Any) -> None:
    with pytest.raises(PersistenceError) as failure:
        harness.store.read(harness.outcome.model_copy(update={"content_hash": "a" * 64}))
    assert failure.value.code == "artifact_hash_mismatch"
    envelope = harness.deps.products.load_envelope(harness.outcome.artifact_id)
    harness.deps.objects.data[envelope.payload_locator] = b"{}"
    with pytest.raises(PersistenceError):
        harness.store.read(harness.outcome)


def test_negative_array_index_is_not_a_valid_citation_pointer() -> None:
    with pytest.raises(ValueError):
        review.select({"results": [1, 2]}, "/results/-1")

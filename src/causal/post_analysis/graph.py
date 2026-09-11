"""One LangGraph owns the author/tool/review/release transitions."""
from __future__ import annotations

from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from causal.post_analysis import agent, review
from causal.post_analysis.contracts import Action, InputError, ReportDraft, Review
from causal.post_analysis.store import LOGGER, Store
from causal.post_analysis.visualization import render_dag, render_visual
from causal.post_analysis.visualization.contracts import RenderedVisual, VisualSpec
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef
from causal.shared.gateway import GatewayImage
from causal.shared.persistence import PersistenceError
from causal.shared.tracing import trace_span


class State(TypedDict, total=False):
    outcome: dict[str, str]
    context: dict[str, str]
    status: str
    action: dict[str, Any]
    observation: dict[str, Any]
    visuals: dict[str, dict[str, str]]
    draft: dict[str, str]
    export: dict[str, str]
    review: dict[str, str]
    bundle: dict[str, str]
    issues: list[dict[str, Any]]
    error_code: str


class Nodes:
    def __init__(self, store: Store, outcome: ArtifactRef) -> None:
        self.store, self.outcome = store, outcome
        self._packet: Any = None

    @property
    def packet(self) -> Any:
        if self._packet is None:
            from causal.post_analysis.input import load_packet
            deps = self.store.deps
            self._packet = load_packet(deps.products, deps.objects, deps.registry,
                                       self.store.analysis_id, self.outcome)
        return self._packet

    def input_check(self, state: State) -> dict[str, Any]:
        packet = self.packet
        body = {"outcome": self.outcome.model_dump(mode="json"),
                "sources": {k: v.model_dump(mode="json") for k, v in packet.sources.items()},
                "required_evidence": packet.required_evidence,
                "limitations": packet.limitations}
        context = self.store.commit("PostAnalysisContext", body,
                                    tuple(dict.fromkeys(packet.sources.values())))
        return {"context": context.model_dump(mode="json"), "status": "running", "visuals": {}}

    def author(self, state: State) -> dict[str, Any]:
        packet = self.packet
        payload = {"evidence": packet.evidence,
                   "required_evidence": packet.required_evidence,
                   "limitations": packet.limitations,
                   "tables": {key: table.model_dump(mode="json", exclude={"rows"})
                              for key, table in packet.tables.items()},
                   "dag_available": packet.diagram is not None,
                   "visuals": {key: self.store.read(ref) for key, ref in state["visuals"].items()},
                   "current_report": self.store.read(state["draft"]) if state.get("draft") else None,
                   "observation": state.get("observation", {})}
        try:
            result = agent.invoke(self.store, "author", ArtifactRef(**state["context"]), payload)
            return {"action": result.model_dump(mode="json")}
        except ValidationError as error:
            return {"action": {}, "observation": {"error": "invalid_action", "details": str(error)}}

    def tools(self, state: State) -> dict[str, Any]:
        action = Action.model_validate(state["action"])
        args = action.arguments
        context = ArtifactRef(**state["context"])
        if action.tool == "read_evidence":
            key = args["evidence_id"]
            return {"observation": {"evidence_id": key, "content": self.packet.evidence[key]}}
        if action.tool == "write_report":
            draft = ReportDraft.model_validate(args)
            visuals = {key: self.store.read(ref) for key, ref in state["visuals"].items()}
            if issues := review.check_draft(draft, self.packet, visuals):
                return {"observation": {"issues": issues}}
            ref = self.store.commit("PostAnalysisDraft", draft.model_dump(mode="json"), (context,))
            return {"draft": ref.model_dump(mode="json"), "export": {}, "review": {},
                    "observation": {"draft": ref.model_dump(mode="json")}}
        if action.tool == "stop":
            return {"status": "incomplete", "error_code": "agent_stopped",
                    "observation": {"reason": args.get("reason", action.decision_summary)}}
        if action.tool not in {"render_visual", "render_dag"}:
            raise ValueError("unsupported tool")
        if len(state["visuals"]) >= 12:
            raise ValueError("visual budget exhausted; reuse an existing visual")
        visual_id = "visual_" + content_hash({"tool": action.tool, "arguments": args})[:16]
        directory = self.store.deps.render_root / self.store.stage_run_id / visual_id
        if action.tool == "render_visual":
            spec = VisualSpec.model_validate({**args, "visual_id": visual_id})
            rendered = render_visual(spec, self.packet.tables, directory)
        else:
            if self.packet.diagram is None:
                raise ValueError("approved source DAG is unavailable")
            rendered = render_dag(visual_id, self.packet.diagram, directory, **args)
        frozen = self.store.freeze_files(rendered.model_dump(mode="json"))
        ref = self.store.commit("PostAnalysisVisual", frozen, (context, rendered.source))
        return {"visuals": {**state["visuals"], visual_id: ref.model_dump(mode="json")},
                "review": {}, "export": {}, "observation": {"visual_id": visual_id, **frozen}}

    def inspect(self, state: State) -> dict[str, Any]:
        from causal.post_analysis.presentation.build import build_report
        if not state.get("draft"):
            return {"observation": {"issues": ["write a report before submitting"]}}
        draft_body = self.store.read(state["draft"])
        draft = ReportDraft.model_validate({k: v for k, v in draft_body.items() if k != "schema_version"})
        visuals = {key: self.store.read(ref) for key, ref in state["visuals"].items()}
        if issues := review.check_draft(draft, self.packet, visuals):
            return {"observation": {"issues": issues}}
        selected = {key for section in draft.sections for key in section.visual_ids}
        local_visuals = {}
        directory = self.store.deps.render_root / self.store.stage_run_id / content_hash(draft_body)
        directory.mkdir(parents=True, exist_ok=True)
        for key in selected:
            value = {k: v for k, v in visuals[key].items() if k != "schema_version"}
            review.verify_objects(self.store, value)
            paths = {}
            for name, locator in value["objects"].items():
                path = directory / key / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.store.deps.objects.get(locator))
                paths[name] = str(path)
            local_visuals[key] = RenderedVisual.model_validate({**value, "objects": paths})
        exported = self.store.freeze_files(build_report(draft, local_visuals, directory))
        export = self.store.commit("PostAnalysisExport", exported,
            (ArtifactRef(**state["draft"]), *(ArtifactRef(**state["visuals"][key]) for key in sorted(selected))))
        images = tuple(GatewayImage(mime_type="image/png", data=self.store.deps.objects.get(
            exported["objects"][key])) for key in exported["preview_keys"])
        if not images:
            raise ValueError("final previews are required")
        binding = review.review_binding(state["context"], state["draft"], state["visuals"],
                                        export.model_dump(mode="json"))
        result = agent.invoke(self.store, "review", ArtifactRef(**state["context"]),
            {"evidence": self.packet.evidence, "required_evidence": self.packet.required_evidence,
             "limitations": self.packet.limitations, "draft": draft.model_dump(mode="json"),
             "visuals": visuals, "export": exported, "binding": binding}, images)
        assert isinstance(result, Review)
        passed = result.verdict == "pass" and not result.issues
        ref = self.store.commit("PostAnalysisReview", {**result.model_dump(mode="json"),
            "binding": binding, "passed": passed, "preview_hashes": exported["object_hashes"]},
            (ArtifactRef(**state["context"]), ArtifactRef(**state["draft"]), export))
        return {"export": export.model_dump(mode="json"), "review": ref.model_dump(mode="json"),
                "observation": {"review_passed": passed, "issues": result.issues,
                                "decision_summary": result.decision_summary}}

    def release(self, state: State) -> dict[str, Any]:
        checked = self.store.read(state["review"])
        binding = review.review_binding(state["context"], state["draft"], state["visuals"], state["export"])
        if not checked["passed"] or checked["binding"] != binding:
            raise ValueError("stale_or_failed_review")
        review.verify_objects(self.store, self.store.read(state["export"]))
        # Reopen authority immediately before release; a cached view cannot authorize delivery.
        self._packet = None
        actual = {k: v.model_dump(mode="json") for k, v in self.packet.sources.items()}
        if actual != self.store.read(state["context"])["sources"]:
            raise ValueError("source_binding_changed")
        fields = {key: state[key] for key in ("context", "draft", "visuals", "export", "review")}
        bundle = self.store.commit("PostAnalysisBundle", {**fields, "binding": binding,
            "stage_run_id": self.store.stage_run_id},
            tuple(ArtifactRef(**state[key]) for key in ("context", "draft", "export", "review")))
        return {"bundle": bundle.model_dump(mode="json"), "status": "complete"}

    def wrap(self, name: str, function: Any) -> Any:
        def run(state: State) -> dict[str, Any]:
            with trace_span(self.store.deps.tracer, f"post_analysis.{name}",
                    run_type="tool" if name == "tools" else "chain", inputs=dict(state),
                    metadata={"analysis_id": self.store.analysis_id,
                              "graph_thread_id": self.store.thread_id}) as span:
                try:
                    result = cast(dict[str, Any], function(state))
                except InputError as error:
                    unavailable = any(issue.owner == "storage" for issue in error.issues)
                    result = {"status": "incomplete" if unavailable else "blocked",
                              "issues": [i.model_dump(mode="json") for i in error.issues],
                              "error_code": "source_unreachable" if unavailable else "invalid_upstream_input"}
                except PersistenceError as error:
                    LOGGER.error("Post-analysis %s storage/integrity failure: %s", name, error.code)
                    result = {"status": "incomplete", "error_code": error.code}
                except (ValueError, KeyError, TypeError) as error:
                    if name in {"tools", "review"} and str(error) != "post_analysis_budget_exhausted":
                        result = {"observation": {"error": str(error)}, "review": {}}
                    else:
                        result = {"status": "incomplete", "error_code": str(error)}
                    LOGGER.warning("Post-analysis %s rejected: %s", name, error)
                span.finish(result)
                return result
        return run


def build_graph(store: Store, outcome: ArtifactRef) -> Any:
    nodes = Nodes(store, outcome)
    graph = StateGraph(State)
    for name, function in (("input_check", nodes.input_check), ("agent", nodes.author),
                           ("tools", nodes.tools), ("review", nodes.inspect), ("release", nodes.release)):
        graph.add_node(name, nodes.wrap(name, function))
    graph.add_node("finish", lambda state: {})
    graph.add_edge(START, "input_check")
    def terminal(state: State) -> bool:
        return state.get("status") in {"complete", "blocked", "incomplete"}
    graph.add_conditional_edges("input_check", lambda state: "finish" if terminal(state) else "agent")
    graph.add_conditional_edges("agent", lambda state: "finish" if terminal(state) else (
        "agent" if not state.get("action") else "review" if state["action"]["tool"] == "submit" else "tools"))
    graph.add_conditional_edges("tools", lambda state: "finish" if terminal(state) else "agent")
    graph.add_conditional_edges("review", lambda state: "finish" if terminal(state) else (
        "release" if state.get("review") and state["observation"].get("review_passed") else "agent"))
    graph.add_edge("release", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=store.deps.checkpointer)

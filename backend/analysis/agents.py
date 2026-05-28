"""Durable four-agent orchestration over immutable applied inputs and accepted handoffs."""

from typing import Any, TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from sqlalchemy.orm import Session

from backend.analysis.agent_computation import Computation
from backend.analysis.agent_context import Workspace, initial_packet
from backend.analysis.agent_design import Design
from backend.analysis.agent_prompts import SYSTEMS
from backend.analysis.agent_report import Explanation
from backend.analysis.agent_runtime import Agent
from backend.analysis.agent_store import CONTRACT, TITLES, Store
from backend.analysis.agent_study import Study
from backend.analysis.records import approval_hash, programs, requested_revision
from backend.analysis.service import emit
from backend.context.contracts import ContextContent
from backend.context.service import fingerprint, revision_record
from backend.models import AnalysisRun, DatasetVersion, now
from backend.storage import files


class State(TypedDict, total=False):
    run_id: str
    revise_program: bool


WORKSPACES = {"study": Study, "design": Design, "computation": Computation, "report": Explanation}
DEPENDENCIES = {"study": [], "design": ["study"], "computation": ["design"], "report": ["design", "computation"]}


class Runtime:
    def __init__(self, db: Session, run: AnalysisRun):
        self.db, self.run, self.store = db, run, Store(db, run.id)
        version = db.get(DatasetVersion, run.version_id)
        if version is None or run.context_revision_id is None:
            raise ValueError("Analysis requires its saved dataset version and applied context revision.")
        self.version = version
        self.context = ContextContent.model_validate(revision_record(db, version.id, run.context_revision_id).content)

    def bind(self, state: State) -> State:
        if self.run.inputs.get("contract") != CONTRACT:
            raise ValueError("Historical analyses retain their original records and do not execute with a new contract.")
        frozen = self.run.inputs.get("frozen_context")
        if not isinstance(frozen, dict) or (
            frozen["dataset_version_id"] != str(self.version.id)
            or frozen["context_revision_id"] != str(self.run.context_revision_id)
            or frozen["source_sha256"] != self.version.sha256
            or frozen["context_hash"] != fingerprint(self.context.model_dump(mode="json"))
            or frozen["question"] != self.run.question
            or frozen["brief"] != (self.context.analysis_brief.model_dump(mode="json") if self.context.analysis_brief else None)
        ):
            raise ValueError("The saved analysis inputs no longer match their authorized identities and content hashes.")
        # Verify integrity, not context quality. Applied context is already agreed.
        files().read(self.version.storage_key, self.version.sha256)
        self.store.recover()
        return {}

    def workspace(self, name: str) -> Workspace:
        handoffs = []
        for dependency in DEPENDENCIES[name]:
            accepted = self.store.accepted(dependency)
            if accepted is None:
                raise ValueError(f"{name} requires the accepted {dependency} handoff.")
            handoffs.append(accepted)
        task = self.store.open(name, initial_packet(self.run, self.context, handoffs, agent=name))
        return WORKSPACES[name](self.store, task, self.run, self.version, self.context)

    def stage(self, name: str) -> State:
        accepted = self.store.accepted(name)
        if accepted:
            emit(self.db, self.run, f"agent:{name}:accepted", name, "handoff_accepted", accepted["summary"],
                 {"agent_id": accepted["producer"], "handoff_id": accepted["id"], "handoff_hash": accepted["hash"]})
            self.db.commit()
            return {}
        work = self.workspace(name)
        self.run.stage, self.run.updated_at = name, now()
        if name == "computation" and programs(self.run):
            current = programs(self.run)[-1]
            if f"execution:{current['revision']}" not in self.run.receipts and approval_hash(self.run) != current["hash"] and not requested_revision(self.run):
                self.run.status = "awaiting_approval"
        emit(self.db, self.run, f"agent:{name}:start", name, "agent_started", TITLES[name], {"agent_id": str(work.task.id)})
        self.db.commit()
        # Each graph has a private LangGraph namespace. Native conversations and
        # side-effect receipts remain durable in domain task/attempt records.
        systems = cast(dict[str, str], self.run.inputs["agent_prompts"])
        Agent(self.store, work.task, work.tools(), systems[name], str(self.run.inputs["model"])).graph().invoke({}, {"recursion_limit": 128})
        accepted = self.store.accepted(name)
        if accepted:
            emit(self.db, self.run, f"agent:{name}:accepted", name, "handoff_accepted", accepted["summary"],
                 {"agent_id": str(work.task.id), "handoff_id": accepted["id"], "handoff_hash": accepted["hash"]})
        elif self.run.status == "awaiting_approval":
            current = programs(self.run)[-1]
            emit(self.db, self.run, f"approval_requested:{current['revision']}", name, "approval_requested",
                 "Review and approve the exact analysis program.", {"program_hash": current["hash"], "snapshot_hash": self.run.snapshot_hash})
        self.db.commit()
        return {"revise_program": False}

    def approval(self, state: State) -> State:
        current = programs(self.run)[-1]
        answer = interrupt({"snapshot_hash": self.run.snapshot_hash, "program_hash": current["hash"],
                            "revision": current["revision"], "action": "Approve this exact frozen program."})
        self.db.refresh(self.run)
        if isinstance(answer, dict) and answer.get("revise_program") == current["hash"] and requested_revision(self.run) == current["hash"]:
            return {"revise_program": True}
        if not isinstance(answer, dict) or answer.get("program_hash") != current["hash"] or approval_hash(self.run) != current["hash"]:
            raise ValueError("The saved approval must match this exact program and scientific snapshot.")
        return {}

    def execute(self, state: State) -> State:
        work = self.workspace("computation")
        assert isinstance(work, Computation)
        self.run.status, self.run.stage = "running", "computation"
        self.db.commit()
        receipt = work.execute()
        current = programs(self.run)[-1]
        emit(self.db, self.run, f"execution:{current['revision']}", "computation", "execution_settled",
             "Approved computation completed." if receipt["valid"] else "Approved computation failed; its receipt is retained.",
             {"execution_id": receipt["execution_id"], "valid": receipt["valid"]})
        self.db.commit()
        return {}


def build(runtime: Runtime, saver: BaseCheckpointSaver[Any]) -> CompiledStateGraph[Any, Any, Any, Any]:
    graph = StateGraph(State)
    graph.add_node("bind", runtime.bind)
    for name in SYSTEMS:
        graph.add_node(name, lambda state, name=name: runtime.stage(name))
    graph.add_node("approval", runtime.approval)
    graph.add_node("execute", runtime.execute)
    graph.add_edge(START, "bind")
    graph.add_edge("bind", "study")
    for name, following in (("study", "design"), ("design", "computation")):
        graph.add_conditional_edges(name, lambda state, following=following: END if runtime.run.status == "blocked" else following, [END, following])
    graph.add_conditional_edges("computation", lambda state: "report" if runtime.store.accepted("computation") else "approval" if runtime.run.status == "awaiting_approval" else END, [END, "approval", "report"])
    graph.add_conditional_edges("approval", lambda state: "computation" if state.get("revise_program") else "execute", ["computation", "execute"])
    graph.add_edge("execute", "computation")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=saver, name="bounded-analysis-agents")

"""Frozen evidence access and provisional work shared by the four analysis agents."""

from copy import deepcopy
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import Field

from backend.analysis import domain, packets
from backend.analysis.agent_runtime import Tool
from backend.analysis.agent_store import Store
from backend.analysis.contracts import Decision, GateCheck
from backend.context.contracts import ContextContent, brief_source_keys, statement_entries
from backend.context.domain import project
from backend.models import AnalysisRun, AnalysisTask, DatasetVersion
from backend.validation import Contract


class NoArguments(Contract):
    pass


class ReadContext(Contract):
    keys: list[str] = Field(min_length=1, max_length=12, description="Exact keys from context_catalogue, or columns.<column_id> for one complete variable context.")


class Publish(Contract):
    summary: str = Field(min_length=1, max_length=2000)
    unresolved: list[str] = Field(default_factory=list, max_length=30, description="Nonblocking unknowns and limitations that downstream work must preserve.")


class EvidenceTask(Contract):
    objective: str = Field(min_length=1, max_length=2000)
    keys: list[str] = Field(min_length=1, max_length=8, description="Existing frozen context keys for this narrow evidence question.")


class Stop(Contract):
    outcome: Literal["unsupported", "unidentified"] = Field(default="unsupported", description="A final scientific outcome under the applied context, not a request for more context.")
    reason: str = Field(min_length=1, max_length=3000)
    limitations: list[str] = Field(default_factory=list, max_length=30)


CONTEXT_POLICY = {
    "status": "applied_and_authorized",
    "authority": "The applied context pack is the agreed study input. Context preparation and review are complete.",
    "access": "Read only the exact frozen revision. Keyed retrieval reads existing records, it does not add context.",
    "assumption_policy": "Proceed with necessary explicit conditional assumptions and preserve supplied qualifications. Optional unknowns do not block analysis.",
    "boundary": "Do not reopen context review, request enrichment, fetch new sources, or edit the pack during this run. Return unsupported or unidentified only when the requested analysis genuinely cannot be justified or executed under these inputs and explicit assumptions.",
}


def evidence_records(context: ContextContent) -> list[dict[str, Any]]:
    # Native tool submissions use actual frozen keys, not the old response aliases.
    return [{**item, "reference": item["key"]} for item in packets.evidence_records(context)]


def decision(summary: str, **fields: Any) -> dict[str, Any]:
    return {"status": "ready", "summary": summary, "changes": summary,
            "permits": "Continue from this validated record.", "assumptions": [], "unresolved": [],
            "citations": sorted(domain.references(fields)), **fields}


def require_checks(checks: list[GateCheck]) -> None:
    errors = [f"{item.rule}: {item.detail}" for item in checks if not item.passed]
    if errors:
        raise ValueError("\n".join(errors))


def initial_packet(run: AnalysisRun, context: ContextContent, handoffs: list[dict[str, Any]], *, agent: str = "study") -> dict[str, Any]:
    brief = context.analysis_brief
    selected = brief_source_keys(brief) if brief else set()
    records = evidence_records(context)
    bound_columns = {cid for variable in ([brief.target] if brief and brief.target else []) + (list(brief.treatments) if brief else []) for cid in variable.column_ids}
    offered = [item for item in records if item["key"] in selected]
    catalogue = [{"key": item["key"], "source_predicates": item["source_predicates"],
                  "status": item["statement"]["status"], "basis": item["statement"]["basis"]}
                 for item in records if not item["key"].startswith("columns.")]
    if agent in {"study", "design"}:
        catalogue.extend({"key": "columns." + cid, "name": column.identity.name,
                          "fields": [name for name, values in column.model_dump(mode="json").items() if name != "identity" and values]}
                         for cid, column in sorted(context.columns.items()))
    else:
        used = set().union(*(domain.references(handoff["records"]) for handoff in handoffs))
        catalogue = [item for item in catalogue if item["key"] in used]
        offered = [item for item in offered if item["key"] in used]
    frozen = cast(dict[str, Any], run.inputs["frozen_context"])
    identities = {key: frozen[key] for key in ("dataset_version_id", "source_sha256", "context_revision_id", "context_hash", "readiness_fingerprint", "policy_version")}
    return {"question": run.question, "frozen_inputs": identities,
            "context_policy": CONTEXT_POLICY,
            "brief": brief.model_dump(mode="json", exclude={"source"}) if brief else None,
            "column_identities": [context.columns[cid].identity.model_dump(mode="json") for cid in sorted(bound_columns)],
            "column_count": len(context.columns),
            "context_catalogue": catalogue,
            "evidence_records": offered,
            "handoffs": [{"producer": item["producer"], "hash": item["hash"], "agent": item["agent"], "summary": item["summary"]} for item in handoffs],
            "prior_exposure": run.inputs["prior_exposure"]}


class Workspace:
    def __init__(self, store: Store, task: AnalysisTask, run: AnalysisRun, version: DatasetVersion, context: ContextContent):
        self.store, self.task, self.run, self.version, self.context = store, task, run, version, context

    def records(self) -> dict[str, Any]:
        self.store.db.refresh(self.task)
        return deepcopy(cast(dict[str, Any], self.task.draft.get("records", {})))

    def visible_records(self) -> dict[str, Any]:
        return cast(dict[str, Any], packets.model_view({name: packets.science(value) if name in {"frame", "snapshot"} else value
                                   for name, value in self.records().items()}))

    def save(self, name: str, record: Any, *, invalidate: tuple[str, ...] = ()) -> dict[str, Any]:
        self.store.db.refresh(self.task)
        if self.task.handoff is not None:
            raise ValueError("The accepted handoff is immutable.")
        draft: dict[str, Any] = deepcopy(self.task.draft)
        records: dict[str, Any] = draft.setdefault("records", {})
        removed = [key for key in records if key != name and any(key.startswith(prefix) for prefix in invalidate)] if records.get(name) != record else []
        for key in removed:
            del records[key]
        records[name] = deepcopy(record)
        self.task.draft = draft
        self.store.db.flush()
        visible = packets.science(record) if name in {"frame", "snapshot"} else record
        return {"saved": name, "record": packets.model_view(visible), "invalidated_drafts": removed}

    def read(self, args: ReadContext, key: str) -> dict[str, Any]:
        content, _, _ = project(self.context, args.keys)
        records = evidence_records(packets.bounded(content))
        self.store.db.refresh(self.task)
        draft: dict[str, Any] = deepcopy(self.task.draft)
        draft["_read_keys"] = sorted(set(draft.get("_read_keys", [])) | {item["key"] for item in records})
        self.task.draft = draft
        return {"evidence_records": records, "column_identities": [item.identity.model_dump(mode="json") for item in content.columns.values()]}

    def check_exposure(self, value: Any) -> None:
        packet: dict[str, Any] = self.task.packet
        draft: dict[str, Any] = self.task.draft
        offered = {item["key"] for item in packet.get("evidence_records", [])} | set(draft.get("_read_keys", []))
        for handoff in packet.get("handoffs", []):
            material = self.store.handoff(UUID(handoff["producer"]), expected_hash=handoff["hash"])
            offered.update(domain.references(material["records"]))
        roots = {ref for ref in domain.references(value) if ref not in {"question"} and not ref.startswith(("decision:", "assumption:", "diagnostic:"))}
        missing = roots - offered
        if missing:
            raise ValueError("Read the exact sources before relying on them: " + ", ".join(sorted(missing)))

    def stop(self, args: Stop, key: str) -> dict[str, Any]:
        self.run.status, self.task.status = "blocked", "blocked"
        self.run.error = args.reason
        self.task.draft = {**self.task.draft, "blocker": args.model_dump(mode="json")}
        return {"status": "blocked", **args.model_dump(mode="json"), "context_revision_unchanged": True}

    def common_tools(self) -> list[Tool]:
        def delegate(args: EvidenceTask, key: str) -> dict[str, Any]:
            from backend.analysis.agent_review import research
            return research(self, args, key)

        return [
            Tool("read_context", "Read existing records from the already applied, frozen context pack. This is retrieval within accepted inputs, not enrichment or renewed context review. Returns facts with qualifications, scope and source eligibility; never raw dataset values.", ReadContext, self.read),
            Tool("inspect_work", "Inspect the current provisional records before revising or publishing. Returns your saved work; it is not an accepted downstream handoff yet.", NoArguments, lambda args, key: {"records": self.visible_records()}),
            Tool("delegate_evidence", "Delegate a narrow question about existing frozen records to a private evidence reader. It has at most two model calls, charged to your allowance. Returns processed findings with source references, never new context or scientific authority. Use only when it helps your task.", EvidenceTask, delegate),
            Tool("stop_analysis", "Return a final unsupported or unidentified outcome only when the requested analysis cannot be justified or executed with the applied context and explicit necessary assumptions. Explain the specific limitation. Do not request more context or reopen review. Optional unknowns alone do not block analysis.", Stop, self.stop),
        ]

    def tools(self) -> list[Tool]:
        raise NotImplementedError("Each analysis stage owns its explicit tool interface.")

    def context_records(self) -> dict[str, Any]:
        return dict(statement_entries(self.context))

    def validate(self, value: Decision, prior: dict[str, Any]) -> None:
        if value.status == "needs_input" or value.requested_keys:
            raise ValueError("Context preparation is complete. Read existing frozen keys with read_context, use explicit analysis assumptions where valid, or return a final unsupported/unidentified outcome. Do not request context additions.")
        self.check_exposure(value.model_dump(mode="json"))
        require_checks(domain.checks(value, self.context, prior, scientific_version=19))

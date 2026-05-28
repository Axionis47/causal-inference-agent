"""Durable agent turns and immutable handoffs. Domain state, not graph checkpoints."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast
from uuid import UUID, uuid5

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend import model_access
from backend.analysis.domain import references
from backend.context.service import fingerprint
from backend.models import AnalysisRun, AnalysisTask, AnalysisTaskAttempt, now

LIMITS = {"study": 10, "design": 16, "computation": 14, "report": 8}
TITLES = {"study": "Understanding the study", "design": "Establishing the analysis design",
          "computation": "Preparing and running the computation", "report": "Explaining the results"}
CONTRACT = "bounded-agents-1"


class AgentLimit(ValueError):
    pass


class SubtaskLimit(AgentLimit):
    """A delegated task ended; its parent may still have useful work to do."""


class HandoffConflict(ValueError):
    pass


class Store:
    def __init__(self, db: Session, run_id: UUID):
        self.db, self.run_id = db, run_id

    def locked_run(self) -> AnalysisRun:
        run = self.db.scalar(select(AnalysisRun).where(AnalysisRun.id == self.run_id).with_for_update())
        if run is None:
            raise HandoffConflict("Analysis no longer exists.")
        return run

    def open(self, agent: str, packet: dict[str, Any], *, key: str | None = None,
             parent: AnalysisTask | None = None, limit: int | None = None) -> AnalysisTask:
        self.locked_run()
        if agent not in LIMITS or parent is not None and parent.agent != agent:
            raise HandoffConflict("Delegation must retain the owning agent's allowance.")
        key = key or "agent:" + agent
        identity = uuid5(self.run_id, key)
        task = self.db.get(AnalysisTask, identity)
        input_hash = fingerprint(packet)
        if task is not None:
            if task.input_hash != input_hash or task.parent_id != (parent.id if parent else None):
                raise HandoffConflict("A resumed agent cannot substitute different inputs or dependencies.")
            self.db.commit()
            return task
        dependencies = packet.get("handoffs", [])
        for dependency in dependencies:
            self.handoff(UUID(dependency["producer"]), expected_hash=dependency["hash"])
        task = AnalysisTask(id=identity, run_id=self.run_id, key=key, checkpoint=agent, kind="agent",
            budget_kind="shared", column_ids=[], obligation_keys=[agent], scope={"contract": CONTRACT},
            reason=TITLES[agent], dependency_receipt_ids=[item["producer"] for item in dependencies],
            input_hash=input_hash, packet=deepcopy(packet), max_attempts=2, status="running", generation=0,
            parent_id=parent.id if parent else None, agent=agent, model_limit=limit or LIMITS[agent],
            submissions=0, draft={})
        self.db.add(task)
        self.db.commit()
        return task

    def turns(self, task: AnalysisTask) -> list[AnalysisTaskAttempt]:
        return list(self.db.scalars(select(AnalysisTaskAttempt).where(
            AnalysisTaskAttempt.task_id == task.id).order_by(AnalysisTaskAttempt.number)))

    def counts(self, agent: str) -> int:
        return int(self.db.scalar(select(func.count()).select_from(AnalysisTaskAttempt).join(
            AnalysisTask, AnalysisTask.id == AnalysisTaskAttempt.task_id).where(
            AnalysisTask.run_id == self.run_id, AnalysisTask.agent == agent)) or 0)

    def exhaustion(self, run: AnalysisRun) -> str | None:
        """Read-only recovery eligibility; saved unfinished tool work can still settle."""
        if run.stage not in LIMITS:
            return None
        root = self.db.get(AnalysisTask, uuid5(self.run_id, "agent:" + run.stage))
        if root is not None and root.status == "accepted":
            return None
        for attempt in self.db.scalars(select(AnalysisTaskAttempt).join(AnalysisTask).where(
                AnalysisTask.run_id == self.run_id, AnalysisTask.agent == run.stage)):
            if attempt.provider_content:
                parts = cast(list[dict[str, Any]], attempt.provider_content.get("parts", []))
                count = max(1, sum(bool(part.get("function_call")) for part in parts))
                receipts = cast(dict[str, Any], attempt.tool_receipts)
                if any(not receipts.get(str(index), {}).get("complete") for index in range(count)):
                    return None
        if self.counts(run.stage) >= LIMITS[run.stage] or run.calls >= int(run.model_limit or 48):
            return f"{TITLES[run.stage]} exhausted its saved model allowance. Accepted work and spent calls are retained."
        return None

    def reserve(self, task: AnalysisTask, request: model_access.PreparedToolRequest) -> AnalysisTaskAttempt:
        run = self.locked_run()
        self.db.refresh(task)
        if task.status in {"accepted", "blocked", "exhausted"}:
            raise HandoffConflict("A settled agent cannot dispatch another model call.")
        turns = self.turns(task)
        if self.counts(str(task.agent)) >= LIMITS[str(task.agent)] or run.calls >= int(run.model_limit or 48):
            task.status, run.status = "exhausted", "blocked"
            run.error = f"{TITLES[str(task.agent)]} reached its saved model allowance. Accepted work remains available."
            self.db.commit()
            raise AgentLimit(run.error)
        if len(turns) >= int(task.model_limit or 0):
            task.status = "exhausted"
            if task.parent_id is not None:
                self.db.commit()
                raise SubtaskLimit("The delegated task exhausted its allowance without an accepted finding. Its calls remain charged to the parent.")
            run.status, run.error = "blocked", f"{TITLES[str(task.agent)]} reached its saved model allowance."
            self.db.commit()
            raise AgentLimit(run.error)
        active = self.db.scalar(select(func.count()).select_from(AnalysisTaskAttempt).where(
            AnalysisTaskAttempt.run_id == self.run_id, AnalysisTaskAttempt.status == "reserved")) or 0
        if active >= 3:
            raise AgentLimit("Three model calls are already in flight for this analysis.")
        number = len(turns) + 1
        record = request.record()
        attempt = AnalysisTaskAttempt(id=uuid5(task.id, f"turn:{number}"), run_id=self.run_id, task_id=task.id,
            key=f"{task.key}:turn:{number}", number=number, generation=task.generation, status="reserved",
            input_hash=task.input_hash, request=record, request_hash=fingerprint(record), metrics={}, tool_receipts={})
        run.calls += 1
        run.shared_calls += 1
        run.updated_at = now()
        self.db.add(attempt)
        self.db.commit()  # Reservation is durable before provider dispatch.
        return attempt

    def settle(self, attempt: AnalysisTaskAttempt, turn: model_access.ToolTurn | None, error: Exception | None = None) -> None:
        self.locked_run()
        self.db.refresh(attempt)
        if attempt.status != "reserved":
            raise HandoffConflict("The model turn has already been settled.")
        attempt.completed_at = now()
        if turn is not None:
            attempt.provider_content = deepcopy(turn.content)
            attempt.raw_output, attempt.metrics, attempt.status = turn.raw, deepcopy(turn.metrics), "accepted"
        else:
            raw, metrics = model_access.response_artifacts(error)
            attempt.raw_output, attempt.status = raw, "failed"
            attempt.metrics = {**metrics, "error_type": type(error).__name__, "error": str(error)[:3500]}
        self.db.commit()

    def recover(self) -> None:
        """Caller owns the run advisory lock; any old reservations lost their worker."""
        self.locked_run()
        for attempt in self.db.scalars(select(AnalysisTaskAttempt).where(
                AnalysisTaskAttempt.run_id == self.run_id, AnalysisTaskAttempt.status == "reserved")):
            attempt.status, attempt.completed_at = "uncertain", now()
            attempt.metrics = {**attempt.metrics, "recovery": "Worker ended before the provider response was durably saved. Its allowance remains charged."}
        self.db.commit()

    def tool(self, task: AnalysisTask, attempt: AnalysisTaskAttempt, index: int, name: str,
             arguments: dict[str, Any], execute: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
        key = str(index)
        request_hash = fingerprint({"name": name, "arguments": arguments})
        self.locked_run()
        self.db.refresh(attempt)
        receipts: dict[str, Any] = deepcopy(attempt.tool_receipts)
        saved = receipts.get(key)
        if saved and saved["request_hash"] != request_hash:
            raise HandoffConflict("A tool call identity cannot be reused with different arguments.")
        if saved and saved.get("complete"):
            self.db.commit()
            return deepcopy(saved["result"])
        if not saved:
            receipts[key] = {"name": name, "arguments": arguments, "request_hash": request_hash, "complete": False}
            attempt.tool_receipts = receipts
            self.db.commit()
        call_key = f"{attempt.id}:{index}"
        result = execute(call_key)
        # Domain writes and the receipt commit together. External tools use call_key
        # as their idempotency identity and can recover a previously completed job.
        self.db.refresh(attempt)
        receipts = deepcopy(attempt.tool_receipts)
        receipts[key] = {**receipts[key], "complete": True, "result": deepcopy(result)}
        attempt.tool_receipts = receipts
        self.db.commit()
        return result

    def submission(self, task: AnalysisTask, call_key: str) -> None:
        self.locked_run()
        self.db.refresh(task)
        draft: dict[str, Any] = deepcopy(task.draft)
        keys = list(draft.get("_submissions", []))
        if call_key not in keys:
            if len(keys) >= 3:
                raise AgentLimit("This agent has used its three handoff submissions.")
            keys.append(call_key)
            draft["_submissions"] = keys
            task.draft, task.submissions = draft, len(keys)
            self.db.commit()

    def publish(self, task: AnalysisTask, records: dict[str, Any], summary: str, unresolved: list[str]) -> dict[str, Any]:
        self.locked_run()
        self.db.refresh(task)
        dependencies = deepcopy(cast(list[dict[str, Any]], task.packet.get("handoffs", [])))

        def collect_dependencies(value: Any) -> None:
            if isinstance(value, dict):
                if isinstance(value.get("producer"), str) and isinstance(value.get("hash"), str):
                    dependencies.append({"producer": value["producer"], "hash": value["hash"]})
                for item in value.values():
                    collect_dependencies(item)
            elif isinstance(value, list):
                for item in value:
                    collect_dependencies(item)

        collect_dependencies(records)
        draft_records = cast(dict[str, Any], task.draft.get("records", {}))
        collect_dependencies({name: value for name, value in draft_records.items() if name.startswith("evidence:")})
        dependencies = list({(item["producer"], item["hash"]): item for item in dependencies}.values())
        for dependency in dependencies:
            self.handoff(UUID(dependency["producer"]), expected_hash=dependency["hash"])
        assumptions: dict[str, Any] = {}

        def ledger(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    ledger(item)
            elif isinstance(value, dict):
                for item in value.get("assumptions", []):
                    if item["id"] in assumptions and assumptions[item["id"]] != item:
                        raise HandoffConflict("A handoff cannot contain competing versions of one assumption.")
                    assumptions[item["id"]] = item
                for name, item in value.items():
                    if name not in {"assumptions", "admission", "original_proposal"}:
                        ledger(item)
        ledger(records)
        body = {"id": str(uuid5(task.id, "handoff")), "producer": str(task.id), "agent": task.agent,
                "schema_version": CONTRACT, "input_hash": task.input_hash,
                "frozen_inputs": task.packet["frozen_inputs"],
                "dependencies": [{"producer": item["producer"], "hash": item["hash"]} for item in dependencies],
                "records": deepcopy(records), "summary": summary, "unresolved": unresolved,
                "evidence": sorted(ref for ref in references(records) if not ref.startswith(("decision:", "assumption:"))),
                "assumption_ids": sorted(assumptions)}
        digest = fingerprint(body)
        if task.handoff is not None and (task.handoff != body or task.handoff_hash != digest):
            raise HandoffConflict("An accepted handoff cannot be replaced.")
        task.handoff, task.handoff_hash, task.status = body, digest, "accepted"
        task.draft = {name: value for name, value in task.draft.items() if name != "records"}
        # Commit with the publishing tool receipt through tool().
        self.db.flush()
        return {**body, "hash": digest}

    def handoff(self, task_id: UUID, *, expected_hash: str | None = None) -> dict[str, Any]:
        task = self.db.get(AnalysisTask, task_id)
        if task is None or task.run_id != self.run_id or task.status != "accepted" or task.handoff is None:
            raise HandoffConflict("A dependency must name an accepted handoff from this analysis.")
        if fingerprint(task.handoff) != task.handoff_hash or expected_hash is not None and expected_hash != task.handoff_hash:
            raise HandoffConflict("The accepted handoff hash does not match its recorded content.")
        return {**deepcopy(task.handoff), "hash": task.handoff_hash}

    def accepted(self, agent: str) -> dict[str, Any] | None:
        task = self.db.get(AnalysisTask, uuid5(self.run_id, "agent:" + agent))
        return self.handoff(task.id) if task is not None and task.status == "accepted" else None

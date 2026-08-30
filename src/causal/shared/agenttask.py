"""One model task: its response schema and its correction loop (D-063, D-065; T-019 seed)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Final, Protocol, cast

from pydantic import BaseModel, ValidationError

from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef
from causal.shared.envelope import AgentTaskEnvelopeV1, AgentTaskResultV1
from causal.shared.events import Severity
from causal.shared.gateway import GatewayResultV1
from causal.shared.validation import ValidationIssueV1, ValidationReport, parse_strict

__all__ = ["EVIDENCE_HEADING", "PARENT_HEADING", "REQUIREMENT_HEADING", "GatewayProtocol",
           "TaskRunner", "result_schema"]

EVIDENCE_HEADING: Final = "\n\n## allowed_evidence\n"
PARENT_HEADING: Final = "\n\n## parent_artifacts\n"
REQUIREMENT_HEADING: Final = "\n\n## registered_requirement_ids\n"
NONE_LINE: Final = "(none)"


class GatewayProtocol(Protocol):
    """The one model call a harness makes (T-010 `VertexGateway.invoke`)."""

    def invoke(self, envelope: AgentTaskEnvelopeV1, prompt: str,
               response_schema: dict[str, object]) -> GatewayResultV1: ...


@cache
def result_schema(draft: type[BaseModel], *, many: bool = False) -> dict[str, object]:
    """`AgentTaskResultV1`'s schema with `payload` replaced by the draft's, or a list of them."""
    result: dict[str, Any] = deepcopy(AgentTaskResultV1.model_json_schema())
    payload: dict[str, Any] = deepcopy(draft.model_json_schema())
    defs: dict[str, Any] = result.setdefault("$defs", {})
    for name, definition in payload.pop("$defs", {}).items():
        if defs.setdefault(name, definition) != definition:
            raise ValueError(f"{draft.__name__} redefines $defs entry {name!r}")
    result["properties"]["payload"] = {
        "type": "object", "properties": {"items": {"type": "array", "items": payload}},
        "required": ["items"]} if many else payload
    return result


@dataclass(frozen=True)
class TaskRunner:
    """The §16.4 model-task loop over injected harness hooks; `state` is the caller's snapshot."""

    gateway: GatewayProtocol
    tasks: Mapping[str, Any]  # task kind -> its registered spec
    tools: Mapping[str, Sequence[str]]  # task kind -> the tool ids it may call
    evals: Mapping[str, tuple[str, ...]]  # task kind -> the eval ids its events carry
    prompts_root: Path
    envelope: Callable[..., AgentTaskEnvelopeV1]  # the task-envelope assembler
    prompt: Callable[..., str]  # the prompt renderer
    validate: Callable[..., ValidationReport]  # walls 1..n over one result
    context: Callable[..., Any]  # the default validation context for this state
    manifest: Callable[..., ArtifactRef]  # the context manifest every envelope names
    evidence: Callable[..., frozenset[str]]  # the analysis's allowlisted evidence ids
    parents: Callable[..., tuple[ArtifactEnvelopeV1, ...]]  # parent envelopes by artifact kind
    commit: Callable[..., ArtifactEnvelopeV1]  # commit one validated payload
    emit: Callable[..., None]  # one operational event
    record: Callable[..., None]  # the audit row for one delegated task
    exhausted: Callable[..., None]  # routing after the last permitted correction
    upsert: Callable[..., None]  # the requirement rows a result raised
    requirements: Sequence[str] = ()  # every requirement id wall 2 admits; empty where none apply

    def invoke(self, state: Any, spec: Any, task_id: str, attempt: int,
               scope: tuple[str, Sequence[str]], refs: tuple[ArtifactRef, ...],
               payload: Mapping[str, object], draft: type[BaseModel], *, many: bool = False,
               ) -> tuple[AgentTaskEnvelopeV1, AgentTaskResultV1 | None]:
        """One physical attempt: envelope, prompt, gateway, strict `AgentTaskResultV1` parse.

        A `many` task whose payload carries no `items` list parses as `None`, so the
        caller's `schema_invalid` correction fires instead of a `KeyError` (D-066).
        """
        built = self.envelope(
            spec, analysis_id=state["analysis_id"], stage_run_id=state["stage_run_id"],
            task_id=task_id, attempt_id=f"{task_id}:{attempt}", scope_kind=scope[0],
            manifest_ref=self.manifest(state), scope_ids=scope[1],
            parent_artifacts=refs, allowed_evidence_ids=sorted(self.evidence(state)),
            allowed_tool_ids=self.tools[spec.task_kind],
            payload_type=f"{spec.task_kind}-context", payload=payload)
        self.emit(state, "agent.started", self.evals[spec.task_kind], task_id=task_id,
                  attempt_id=built.attempt_id, attempt_number=attempt)
        # Wall 2 admits only these ids, and a result's parents must be exactly these
        # committed refs, so the prompt must carry every closed set it enforces — evidence,
        # parents, and the requirement ids a result may raise (D-065, D-068, D-098).
        rendered = self.prompt(spec, self.prompts_root, dict(payload)) + EVIDENCE_HEADING + (
            "\n".join(sorted(built.allowed_evidence_ids)) or NONE_LINE) + PARENT_HEADING + (
            "\n".join(ref.artifact_id for ref in built.parent_artifacts) or NONE_LINE
            ) + REQUIREMENT_HEADING + ("\n".join(sorted(self.requirements)) or NONE_LINE)
        answer = self.gateway.invoke(built, rendered, result_schema(draft, many=many))
        try:
            parsed = parse_strict(AgentTaskResultV1, answer.parsed or {})
        except ValidationError:
            return built, None
        # D-071: the harness owns these three facts, so a model echo never decides them.
        parsed = parsed.model_copy(update={
            "envelope_id": built.envelope_id, "task_id": built.task_id,
            "validation_target": built.validator_version})
        return built, None if many and not isinstance(parsed.payload.get("items"), list) else parsed

    def run(self, state: Any, kind: str, model: type[BaseModel], *, scope_kind: str,
            scope_ids: Sequence[str], parent_kinds: tuple[str, ...],
            payload: Mapping[str, object], many: bool = False, commits: str | None = None,
            ctx: Any = None) -> tuple[tuple[Any, ArtifactEnvelopeV1], ...] | None:
        """One model task with the §16.4 correction loop; commits every validated payload."""
        spec, evals = self.tasks[kind], self.evals[kind]
        artifact_type = commits or spec.output_artifact_type
        task_id = f"dt:{state['stage_run_id']}:{kind}:{content_hash(dict(payload))[:12]}"
        parents = self.parents(state, *parent_kinds)
        refs = tuple(ArtifactRef(artifact_id=p.artifact_id, content_hash=p.content_hash)
                     for p in parents)
        body, issues = dict(payload), cast(tuple[ValidationIssueV1, ...], ())
        for attempt in range(1, spec.correction_budget + 2):
            built, result = self.invoke(
                state, spec, task_id, attempt, (scope_kind, scope_ids), refs, body, model,
                many=many)
            digest = content_hash(built.canonical_payload())
            if result is None:
                for name in ("agent.schema_failed", "agent.correction_requested"):
                    self.emit(state, name, evals, severity=Severity.ERROR,
                              task_id=task_id, attempt_number=attempt,
                              error_code="schema_invalid")
                state["corrections"][f"{artifact_type}:schema_invalid"] = attempt
                body = dict(payload) | {"correction": {"issues": [{"code": "schema_invalid"}]}}
                continue
            items = ([dict(row) for row in cast(list[Any], result.payload["items"])] if many
                     else [dict(result.payload)])
            issues = tuple(
                issue for row in items
                for issue in self.validate(
                    spec.wall, kind, model, result.model_copy(update={"payload": row}),
                    ctx or self.context(state)).issues)
            self.upsert(
                result.missing_requirements, state["analysis_id"], state["design_revision"])
            state["open_requirement_ids"] = sorted({*state["open_requirement_ids"], *(
                row.requirement_id for row in result.missing_requirements)})
            if not issues:
                done = tuple((parse_strict(model, row), self.commit(
                    state, artifact_type, row, parents)) for row in items)
                self.record(state, task_id, kind, scope_ids, digest, built,
                            result.status.value, attempt, done[-1][1].artifact_id)
                self.emit(state, "task.completed", evals, task_id=task_id,
                          status=result.status.value)
                return done
            for issue in issues:
                state["corrections"][f"{artifact_type}:{issue.code}"] = attempt
            self.emit(state, "artifact.validation_failed", evals, task_id=task_id,
                      severity=Severity.WARNING, error_code=issues[0].code)
            self.emit(state, "agent.correction_requested", evals, task_id=task_id,
                      attempt_number=attempt, error_code=issues[0].code)
            body = dict(payload) | {"correction": {"failing_payload": items[0], "issues": [
                issue.model_dump(mode="json") for issue in issues]}}
        self.exhausted(state, task_id, kind, issues)
        return None

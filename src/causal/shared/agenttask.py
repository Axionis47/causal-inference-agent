"""One model task: its response schema and its correction loop (D-063, D-065; T-019 seed)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, is_dataclass, replace
from functools import cache
from pathlib import Path
from textwrap import indent
from typing import Annotated, Any, Final, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, ReferenceKind
from causal.shared.envelope import (
    AgentTaskEnvelopeV1,
    AgentTaskResultV1,
    AttemptedEvidenceV1,
    ContextRequirementV1,
    TaskStatus,
)
from causal.shared.events import Severity
from causal.shared.gateway import MODEL_OUTPUT_TRUNCATED, GatewayError, GatewayResultV1
from causal.shared.validation import (
    FIX_ACTIONS,
    ValidationIssueV1,
    ValidationReport,
    constrain_reference_schema,
    make_issue,
    parse_strict,
    reference_projection,
    reference_snapshot,
    validate_references,
    walk_reference_fields,
)

__all__ = ["EVIDENCE_CHARS", "EVIDENCE_HEADING", "EVIDENCE_TOTAL", "PARENT_HEADING",
           "REFERENCE_HEADING", "REQUIREMENT_HEADING", "GatewayProtocol", "TaskRunner",
           "evidence_availability", "evidence_block", "in_task_scope", "result_schema"]

EVIDENCE_HEADING: Final = "\n\n## allowed_evidence\n"
PARENT_HEADING: Final = "\n\n## parent_artifacts\n"
REFERENCE_HEADING: Final = "\n\n## reference_catalog\n"
REQUIREMENT_HEADING: Final = "\n\n## registered_requirement_ids\n"
NONE_LINE: Final = "(none)"
# One item's text, then the whole block. Every allowlisted item lands in every task prompt,
# so a dataset with per-column evidence would otherwise grow the prompt without bound.
EVIDENCE_CHARS: Final = 4000
EVIDENCE_TOTAL: Final = 24000
_REFERENCE_RULE: Final = "wall2.typed_reference"


class _ModelTaskDecisionV1(BaseModel):
    """Only fields that require model judgment; the harness owns the result envelope."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    status: TaskStatus
    payload: dict[str, object]
    missing_requirements: tuple[ContextRequirementV1, ...]
    conflicts: Annotated[tuple[str, ...], Field(max_length=16)]
    warnings: Annotated[tuple[str, ...], Field(max_length=16)]


def _schema_issue(path: str, error_type: str) -> dict[str, str]:
    return {"code": "schema_invalid", "json_path": path[:240] or "/",
            "error_type": error_type[:80] or "schema_invalid"}


def _with_correction(payload: Mapping[str, object],
                     correction: Mapping[str, object]) -> dict[str, object]:
    """Keep original task feedback distinct from replaceable response-validation feedback."""
    feedback = payload.get("correction")
    held = ({"task_feedback": deepcopy(dict(feedback))}
            if isinstance(feedback, Mapping) else {})
    return dict(payload) | {"correction": dict(correction) | held}


def _pydantic_issues(error: ValidationError) -> tuple[dict[str, str], ...]:
    errors = error.errors(include_url=False, include_context=False, include_input=False)
    errors.sort(key=lambda item: item["type"] == "extra_forbidden")
    issues = tuple(_schema_issue("/" + "/".join(str(part).replace("~", "~0").replace(
        "/", "~1") for part in item["loc"]), str(item["type"])) for item in errors[:8])
    return tuple(issues) or (_schema_issue("/", "schema_invalid"),)


def _result_contract_issues(
        built: AgentTaskEnvelopeV1, result: AgentTaskResultV1) -> tuple[dict[str, str], ...]:
    checks = ((result.status not in built.allowed_stopping_states, "/status",
               "stopping_state_not_allowed"),
              (result.status.value == "needs_context" and not result.missing_requirements,
               "/missing_requirements", "missing_requirements_required_for_needs_context"),
              (result.status.value == "conflict" and not result.conflicts,
               "/conflicts", "conflicts_required_for_conflict"))
    return tuple(_schema_issue(path, error) for failed, path, error in checks if failed)


class GatewayProtocol(Protocol):
    def invoke(self, envelope: AgentTaskEnvelopeV1, prompt: str,
               response_schema: dict[str, object]) -> GatewayResultV1: ...


def in_task_scope(evidence_id: str, scope_ids: Sequence[str]) -> bool:
    """A column-scoped evidence id belongs only to a task that was assigned that column.

    Covers both families: `ev:kaggle/column/<table>/<column>/<field>` from intake, and the
    `<profile>#/columns/<column>` measured facts the harness computed.
    """
    parts = [part for part in evidence_id.split("/") if part]
    if not scope_ids:
        return True
    if "columns" in parts:  # <profile>#/columns/<column>
        return parts[-1] in scope_ids
    if "column" in parts and len(parts) > 3:  # ev:kaggle/column/<table>/<column>/<field>
        return parts[3] in scope_ids
    return True


def _evidence_rows(evidence: Mapping[str, str] | frozenset[str],
                   scope_ids: Sequence[str] = ()) -> tuple[tuple[str, str, str], ...]:
    """Render-ready evidence rows with harness-owned shown/empty/withheld status."""
    values = evidence if isinstance(evidence, Mapping) else {}
    identifiers_only = not isinstance(evidence, Mapping)
    rows, used = [], 0
    for key in sorted(evidence, key=lambda item: (not item.startswith("ua:"), item)):
        if not in_task_scope(key, scope_ids):
            continue
        item = (values.get(key) or "")[:EVIDENCE_CHARS]
        status = "evidenced"
        if used + len(item) > EVIDENCE_TOTAL:
            item, status = "(withheld: this task's evidence budget is full)", "withheld"
        elif not item and not identifiers_only:
            status = "empty"
        else:
            used += len(item)
        rows.append((key, item, status))
    return tuple(rows)


def evidence_availability(evidence: Mapping[str, str] | frozenset[str],
                          scope_ids: Sequence[str] = ()) -> dict[str, str]:
    """Return deterministic prompt availability; models never author these statuses."""
    return {key: status for key, _, status in _evidence_rows(evidence, scope_ids)}


def evidence_block(evidence: Mapping[str, str] | frozenset[str],
                   scope_ids: Sequence[str] = (), *, group_identical: bool = False) -> str:
    """Every in-scope evidence id with the text it carries, under one budget (D-102, D-103).

    Rendering the id alone let a worker report a document it was never shown as `not_offered`,
    so the gate saw its sources exhausted and escalated to the user for facts the harness was
    holding. Rendering all of them without a bound is the opposite failure, so a task sees the
    dataset-wide sources plus its own columns, and an item dropped for budget says so rather
    than vanishing. A caller with no text — estimation, presentation — still renders bare ids.
    """
    rows = _evidence_rows(evidence, scope_ids)
    if group_identical:
        groups: dict[tuple[str, str], list[str]] = {}
        for key, value, status in rows:
            groups.setdefault((value, status), []).append(key)
        return json.dumps([{"evidence_ids": ids, "text": value, "availability": status}
                           for (value, status), ids in groups.items()], separators=(",", ":"))
    lines = [f"{key}:\n{indent(value, '  ')}" if value else key for key, value, _ in rows]
    return "\n".join(lines) or NONE_LINE


@cache
def result_schema(
    draft: type[BaseModel], *, many: bool = False, item_count: int | None = None,
    requirement_ids: tuple[str, ...] = (),
    evidence_ids: tuple[str, ...] = (), diagnostic_ids: tuple[str, ...] = (),
    diagnostic_result_ids: tuple[str, ...] = (),
    diagnostic_request_limit: int | None = None,
    column_ids: tuple[str, ...] = (), concept_ids: tuple[str, ...] = (),
    graph_edge_ids: tuple[str, ...] = (), alternative_ids: tuple[str, ...] = (),
    method_ids: tuple[str, ...] = (), artifact_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    """Forced schema for model-owned decisions, excluding compiler-owned envelope fields."""
    result: dict[str, Any] = deepcopy(_ModelTaskDecisionV1.model_json_schema())
    payload: dict[str, Any] = deepcopy(draft.model_json_schema())
    if diagnostic_request_limit is not None:
        requests = payload.get("properties", {}).get("requested_diagnostic_ids")
        if isinstance(requests, dict):
            requests["maxItems"] = min(requests.get("maxItems", diagnostic_request_limit),
                                       max(0, diagnostic_request_limit))
    defs: dict[str, Any] = result.setdefault("$defs", {})
    for name, definition in payload.pop("$defs", {}).items():
        if defs.setdefault(name, definition) != definition:
            raise ValueError(f"{draft.__name__} redefines $defs entry {name!r}")
    result["properties"]["payload"] = {
        "type": "object", "properties": {"items": {"type": "array", "items": payload}},
        "required": ["items"]} if many else payload
    if many and item_count is not None:
        result["properties"]["payload"]["properties"]["items"].update(
            minItems=item_count, maxItems=item_count)
    catalogs = {
        ReferenceKind.EVIDENCE: evidence_ids,
        ReferenceKind.DIAGNOSTIC: diagnostic_ids,
        ReferenceKind.DIAGNOSTIC_RESULT: diagnostic_result_ids,
        ReferenceKind.REQUIREMENT: requirement_ids,
        ReferenceKind.COLUMN: column_ids,
        ReferenceKind.CONCEPT: concept_ids,
        ReferenceKind.GRAPH_EDGE: graph_edge_ids,
        ReferenceKind.ALTERNATIVE: alternative_ids,
        ReferenceKind.METHOD: method_ids,
        ReferenceKind.ARTIFACT: artifact_ids,
    }
    constrain_reference_schema(result, catalogs)
    # Availability is a harness observation, never part of the model decision.
    defs["ContextRequirementV1"]["properties"]["attempted_evidence"]["maxItems"] = 0
    return result


def _seal_result(
    built: AgentTaskEnvelopeV1, artifact_type: str, raw: Mapping[str, object]
) -> AgentTaskResultV1:
    """Validate the model decision, then deterministically stamp identity and lineage."""
    decision = parse_strict(_ModelTaskDecisionV1, raw)
    return AgentTaskResultV1(
        envelope_id=built.envelope_id, schema_version="agent-task-result.v1",
        task_id=built.task_id, status=decision.status, artifact_type=artifact_type,
        artifact_schema_version=built.output_schema_version,
        parent_artifact_ids=tuple(ref.artifact_id for ref in built.parent_artifacts),
        payload=dict(decision.payload),
        missing_requirements=decision.missing_requirements, conflicts=decision.conflicts,
        warnings=decision.warnings, validation_target=built.validator_version)


def _requirements_with_availability(
    requirements: Sequence[ContextRequirementV1], availability: Mapping[str, str],
) -> tuple[ContextRequirementV1, ...]:
    """Replace model-authored source status with the exact task evidence surface."""
    attempted = tuple(AttemptedEvidenceV1(evidence_id=key, availability_status=status)
                      for key, status in sorted(availability.items()))
    return tuple(row.model_copy(update={"attempted_evidence": attempted})
                 for row in requirements)


def _context_reference_catalogs(
    ctx: Any, *, evidence_ids: Sequence[str], requirement_ids: Sequence[str],
    diagnostic_ids: Sequence[str], diagnostic_result_ids: Sequence[str],
    artifact_ids: Sequence[str],
) -> dict[ReferenceKind, tuple[str, ...]]:
    """Build task-local catalogs without coupling the shared runner to a stage context type."""
    catalogs: dict[ReferenceKind, tuple[str, ...]] = {
        ReferenceKind.EVIDENCE: tuple(evidence_ids),
        ReferenceKind.REQUIREMENT: tuple(requirement_ids),
        ReferenceKind.DIAGNOSTIC: tuple(diagnostic_ids),
        ReferenceKind.DIAGNOSTIC_RESULT: tuple(diagnostic_result_ids),
        ReferenceKind.ARTIFACT: tuple(artifact_ids),
    }
    context_parents = getattr(ctx, "parents", {})
    if isinstance(context_parents, Mapping):
        catalogs[ReferenceKind.ARTIFACT] = tuple(sorted({
            *catalogs[ReferenceKind.ARTIFACT],
            *(str(item) for item in context_parents),
        }))
    manifest = getattr(ctx, "manifest", None)
    inventory = getattr(manifest, "structural_inventory", ())
    if inventory:
        catalogs[ReferenceKind.COLUMN] = tuple(
            sorted({str(row.column_name) for row in inventory})
        )
    for kind, attribute in (
        (ReferenceKind.CONCEPT, "concept_ids"),
        (ReferenceKind.GRAPH_EDGE, "relationship_ids"),
    ):
        values = getattr(ctx, attribute, ())
        if values:
            catalogs[kind] = tuple(sorted(str(item) for item in values))
    causal_context = getattr(ctx, "causal_context", None)
    if causal_context is not None:
        catalogs[ReferenceKind.CONCEPT] = tuple(sorted({
            *catalogs.get(ReferenceKind.CONCEPT, ()), *causal_context.concept_ids,
        }))
        catalogs[ReferenceKind.GRAPH_EDGE] = tuple(sorted({
            *catalogs.get(ReferenceKind.GRAPH_EDGE, ()),
            *(edge.edge_id for edge in causal_context.edges),
            *(edge.edge_id for alternative in causal_context.alternatives
              for edge in alternative.edges),
        }))
        catalogs[ReferenceKind.ALTERNATIVE] = tuple(sorted(
            alternative.alternative_id for alternative in causal_context.alternatives
        ))
    packs = getattr(ctx, "packs", None)
    if packs is not None:
        catalogs[ReferenceKind.METHOD] = tuple(
            sorted(str(pack.method_id) for pack in packs.all())
        )
    return catalogs


def _decision_projection(
    model: type[BaseModel], items: Sequence[Mapping[str, object]], result: AgentTaskResultV1,
    *, editable_excerpt_paths: frozenset[str] = frozenset(), many: bool = False,
) -> dict[str, Any]:
    """Freeze semantics except typed references and explicitly coupled citation excerpts."""
    normalized: list[Mapping[str, object]] = []
    for row in items:
        try:
            # An omitted declared default is the same typed decision as its explicit value.
            # Invalid drafts keep their raw shape: normalization cannot hide schema failures.
            normalized.append(parse_strict(model, row).model_dump(mode="json"))
        except ValidationError:
            normalized.append(row)
    projected = [reference_projection(model, row) for row in normalized]
    for index, row in enumerate(projected):
        prefix = f"/payload/items/{index}" if many else "/payload"
        interpretations = row.get("source_interpretations", ())
        if not isinstance(interpretations, list | tuple):
            continue
        for position, interpretation in enumerate(interpretations):
            path = f"{prefix}/source_interpretations/{position}/verbatim_excerpt"
            if path in editable_excerpt_paths and isinstance(interpretation, dict):
                interpretation.pop("verbatim_excerpt", None)
    return {
        "status": result.status.value,
        "payload": projected,
        "missing_requirements": [
            reference_projection(ContextRequirementV1, row.model_dump(mode="python"))
            for row in result.missing_requirements
        ],
        "conflicts": list(result.conflicts),
        "warnings": list(result.warnings),
    }


def _citation_excerpt_paths(
    model: type[BaseModel], items: Sequence[Mapping[str, object]],
    editable_references: frozenset[str], *, many: bool,
) -> frozenset[str]:
    """Only an invalid typed source-interpretation evidence id permits its sibling quote."""
    paths = set()
    for index, row in enumerate(items):
        prefix = f"/payload/items/{index}" if many else "/payload"
        typed_evidence = {field.path for field in walk_reference_fields(model, row, prefix=prefix)
                          if field.kind is ReferenceKind.EVIDENCE
                          and field.schema.get("type") == "string"}
        interpretations = row.get("source_interpretations", ())
        if not isinstance(interpretations, list | tuple):
            continue
        for position, interpretation in enumerate(interpretations):
            parent = f"{prefix}/source_interpretations/{position}"
            if (f"{parent}/evidence_id" in editable_references & typed_evidence
                    and isinstance(interpretation, Mapping)
                    and isinstance(interpretation.get("verbatim_excerpt"), str)):
                paths.add(f"{parent}/verbatim_excerpt")
    return frozenset(paths)


def _citation_quote_failed(issues: Sequence[ValidationIssueV1],
                           editable_excerpt_paths: frozenset[str]) -> bool:
    # Design quote-wall paths are local to one payload. Keep the citation freeze until its
    # quote also passes; otherwise a bad quote would release all semantics on the next retry.
    local_paths = {"/source_interpretations/" + path.split("/source_interpretations/", 1)[1]
                   for path in editable_excerpt_paths}
    return any(issue.rule_id == "wall3.evidence_quote"
               and (issue.json_path in editable_excerpt_paths or issue.json_path in local_paths)
               for issue in issues)


def _decision_reference_snapshot(
    model: type[BaseModel], items: Sequence[Mapping[str, object]], result: AgentTaskResultV1,
    *, many: bool, array_paths: set[str],
) -> dict[str, str]:
    refs: dict[str, str] = {}
    for index, row in enumerate(items):
        prefix = f"/payload/items/{index}" if many else "/payload"
        refs.update(reference_snapshot(model, row, prefix=prefix))
        array_paths.update(field.path for field in walk_reference_fields(model, row, prefix=prefix)
                           if field.schema.get("type") == "array")
    for index, requirement in enumerate(result.missing_requirements):
        refs.update(reference_snapshot(
            ContextRequirementV1, requirement.model_dump(mode="python"),
            prefix=f"/missing_requirements/{index}",
        ))
        array_paths.update(field.path for field in walk_reference_fields(
            ContextRequirementV1, requirement.model_dump(mode="python"),
            prefix=f"/missing_requirements/{index}") if field.schema.get("type") == "array")
    return refs


def _reference_array_matches(
    baseline: Sequence[tuple[str, str]], current: Sequence[str], editable: frozenset[str],
) -> bool:
    """Invalid entries may be replaced or deleted; every valid entry must survive in order."""
    positions = {0}
    for path, value in baseline:
        advanced = {index + 1 for index in positions if index < len(current)
                    and (path in editable or current[index] == value)}
        positions = positions | advanced if path in editable else advanced
    return len(current) in positions


def _reference_repair_issues(
    baseline: Mapping[str, str], current: Mapping[str, str], editable_paths: frozenset[str],
    array_paths: frozenset[str],
) -> tuple[ValidationIssueV1, ...]:
    changed = {
        path for path in set(baseline) | set(current)
        if path not in editable_paths and baseline.get(path) != current.get(path)
    }
    for parent in array_paths & {path.rpartition("/")[0] for path in editable_paths}:
        old = sorted(((path, value) for path, value in baseline.items()
                      if path.rpartition("/")[0] == parent),
                     key=lambda item: int(item[0].rpartition("/")[2]))
        new = sorted(((path, value) for path, value in current.items()
                      if path.rpartition("/")[0] == parent),
                     key=lambda item: int(item[0].rpartition("/")[2]))
        if _reference_array_matches(old, [value for _, value in new], editable_paths):
            changed.difference_update(path for path, _ in (*old, *new))
    return tuple(make_issue(
        "reference_repair_changed_unrelated_reference", path,
        "correction.reference_only", FIX_ACTIONS, False,
        detail="Preserve every valid reference in order; only invalid references may be "
               "replaced or deleted from an array.",
    ) for path in sorted(changed))


@dataclass(frozen=True)
class TaskRunner:
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
    evidence: Callable[..., Any]  # allowlisted evidence ids, with their text where there is any
    parents: Callable[..., tuple[ArtifactEnvelopeV1, ...]]  # parent envelopes by artifact kind
    commit: Callable[..., ArtifactEnvelopeV1]  # commit one validated payload
    emit: Callable[..., None]  # one operational event
    record: Callable[..., None]  # the audit row for one delegated task
    exhausted: Callable[..., None]  # routing after the last permitted correction
    upsert: Callable[..., None]  # validated requirement rows, canonicalized with task state
    requirements: Sequence[str] = ()  # every requirement id wall 2 admits; empty where none apply

    def invoke(self, state: Any, spec: Any, task_id: str, attempt: int,
               scope: tuple[str, Sequence[str]], refs: tuple[ArtifactRef, ...],
               payload: Mapping[str, object], draft: type[BaseModel], *, artifact_type: str,
               validation_ctx: Any,
               many: bool = False, evidence_scope_ids: Sequence[str] = (),
               ) -> tuple[
                   AgentTaskEnvelopeV1, AgentTaskResultV1 | None,
                   tuple[dict[str, str], ...], dict[str, str],
                   dict[ReferenceKind, tuple[str, ...]],
               ]:
        """One physical attempt: envelope, prompt, gateway, strict `AgentTaskResultV1` parse.

        A `many` task whose payload carries no `items` list parses as `None`, so the
        caller's `schema_invalid` correction fires instead of a `KeyError` (D-066).
        """
        offered = self.evidence(state)
        surface = ({key: value for key, value in offered.items()
                    if in_task_scope(key, evidence_scope_ids)}
                   if isinstance(offered, Mapping) else frozenset(
                       key for key in offered if in_task_scope(key, evidence_scope_ids)))
        availability = evidence_availability(surface)
        evidence = tuple(key for key, status in availability.items() if status == "evidenced")
        built = self.envelope(
            spec, analysis_id=state["analysis_id"], stage_run_id=state["stage_run_id"],
            task_id=task_id, attempt_id=f"{task_id}:{attempt}", scope_kind=scope[0],
            manifest_ref=self.manifest(state), scope_ids=scope[1],
            parent_artifacts=refs, allowed_evidence_ids=sorted(evidence),
            allowed_tool_ids=self.tools[spec.task_kind],
            payload_type=f"{spec.task_kind}-context", payload=payload)
        self.emit(state, "agent.started", self.evals[spec.task_kind], task_id=task_id,
                  attempt_id=built.attempt_id, attempt_number=attempt)
        # Wall 2 admits only these ids, and a result's parents must be exactly these
        # committed refs, so the prompt must carry every closed set it enforces — evidence,
        # parents, and the requirement ids a result may raise (D-065, D-068, D-098).
        requirements = tuple(getattr(spec, "allowed_requirement_ids", self.requirements))
        diagnostic_results = payload.get("diagnostic_results", ())
        diagnostic_result_ids = tuple(sorted(str(row["diagnostic_result_id"])
            for row in (diagnostic_results if isinstance(diagnostic_results, Sequence) else ())
            if isinstance(row, Mapping) and row.get("diagnostic_result_id")))
        available = payload.get("available_diagnostics")
        diagnostic_ids = tuple(sorted({str(item) for rows in available.values() for item in rows}
            if isinstance(available, Mapping) else set()))
        catalogs = _context_reference_catalogs(
            validation_ctx, evidence_ids=evidence, requirement_ids=requirements,
            diagnostic_ids=diagnostic_ids, diagnostic_result_ids=diagnostic_result_ids,
            artifact_ids=tuple(sorted({*evidence, *(
                ref.artifact_id for ref in built.parent_artifacts)})),
        )
        catalog = {"source_evidence_ids": sorted(evidence),
                   "evidence_availability": availability,
                   "parent_artifact_ids": [ref.artifact_id for ref in built.parent_artifacts],
                   "registered_diagnostic_ids": list(diagnostic_ids),
                   "diagnostic_result_ids": list(diagnostic_result_ids)}
        rendered = (self.prompt(spec, self.prompts_root, dict(payload)) + REFERENCE_HEADING +
            json.dumps(catalog, indent=1, sort_keys=True) + EVIDENCE_HEADING +
            evidence_block(surface, group_identical=spec.task_kind == "causal_context") + PARENT_HEADING +
            ("\n".join(ref.artifact_id for ref in built.parent_artifacts) or NONE_LINE) +
            REQUIREMENT_HEADING + ("\n".join(sorted(requirements)) or NONE_LINE))
        loop = payload.get("agent_loop")
        remaining = loop.get("remaining_tool_calls") if isinstance(loop, Mapping) else None
        diagnostic_limit = (remaining if spec.task_kind == "method_design"
                            and type(remaining) is int and remaining >= 0 else None)
        schema = result_schema(
            draft, many=many, item_count=len(scope[1]) if scope[0] == "column" else None,
            requirement_ids=tuple(sorted(requirements)),
            evidence_ids=tuple(sorted(evidence)), diagnostic_ids=diagnostic_ids,
            diagnostic_result_ids=diagnostic_result_ids,
            diagnostic_request_limit=diagnostic_limit,
            column_ids=catalogs.get(ReferenceKind.COLUMN, ()),
            concept_ids=catalogs.get(ReferenceKind.CONCEPT, ()),
            graph_edge_ids=catalogs.get(ReferenceKind.GRAPH_EDGE, ()),
            alternative_ids=catalogs.get(ReferenceKind.ALTERNATIVE, ()),
            method_ids=catalogs.get(ReferenceKind.METHOD, ()),
            artifact_ids=catalogs.get(ReferenceKind.ARTIFACT, ()))
        try:
            answer = self.gateway.invoke(built, rendered, schema)
        except GatewayError as error:
            if error.code != MODEL_OUTPUT_TRUNCATED:
                raise
            # No partial response is parsed. This is an output correction in the caller's
            # existing budget, not a retry of the nonretryable transport operation.
            return built, None, ({"code": MODEL_OUTPUT_TRUNCATED,
                "json_path": "/response", "error_type": "incomplete_json"},), availability, catalogs
        try:
            parsed = _seal_result(built, artifact_type, answer.parsed or {})
        except ValidationError as error:
            return built, None, _pydantic_issues(error), availability, catalogs
        contract_issues = _result_contract_issues(built, parsed)
        if many and not isinstance(parsed.payload.get("items"), list):
            error_type = "missing" if "items" not in parsed.payload else "list_type"
            contract_issues += (_schema_issue("/payload/items", error_type),)
        elif (many and scope[0] == "column"
              and len(cast(list[object], parsed.payload["items"])) != len(scope[1])):
            contract_issues += (_schema_issue("/payload/items", "assigned_item_count_mismatch"),)
        for index, requirement in enumerate(parsed.missing_requirements):
            if requirement.attempted_evidence:
                contract_issues += (_schema_issue(
                    f"/missing_requirements/{index}/attempted_evidence", "harness_owned"),)
        return (built, None if contract_issues else parsed, contract_issues, availability,
                catalogs)

    def run(self, state: Any, kind: str, model: type[BaseModel], *, scope_kind: str,
            scope_ids: Sequence[str], parent_kinds: tuple[str, ...],
            payload: Mapping[str, object], many: bool = False, commits: str | None = None,
            ctx: Any = None, evidence_scope_ids: Sequence[str] | None = None,
            precommit_admission: Callable[
                [tuple[BaseModel, ...], AgentTaskResultV1], tuple[ValidationIssueV1, ...]
            ] | None = None,
            ) -> tuple[tuple[Any, ArtifactEnvelopeV1], ...] | None:
        spec, evals = self.tasks[kind], self.evals[kind]
        artifact_type = commits or spec.output_artifact_type
        task_id = f"dt:{state['stage_run_id']}:{kind}:{content_hash(dict(payload))[:12]}"
        parents = self.parents(state, *parent_kinds)
        refs = tuple(ArtifactRef(artifact_id=p.artifact_id, content_hash=p.content_hash)
                     for p in parents)
        body, issues = dict(payload), cast(tuple[ValidationIssueV1, ...], ())
        validation_ctx = ctx or self.context(state)
        reference_semantic_baseline: Any = None
        reference_value_baseline: dict[str, str] = {}
        reference_array_paths: frozenset[str] = frozenset()
        editable_reference_paths: frozenset[str] = frozenset()
        editable_excerpt_paths: frozenset[str] = frozenset()
        reference_correction: dict[str, object] = {}
        for attempt in range(1, spec.correction_budget + 2):
            built, result, contract_issues, availability, catalogs = self.invoke(
                state, spec, task_id, attempt, (scope_kind, scope_ids), refs, body, model,
                artifact_type=artifact_type, validation_ctx=validation_ctx, many=many,
                evidence_scope_ids=evidence_scope_ids or scope_ids)
            digest = content_hash(built.canonical_payload())
            if result is None:
                safe = contract_issues[0] if contract_issues else _schema_issue("/", "unknown")
                code = safe["code"]
                issues = tuple(make_issue(row["code"], row["json_path"],
                    "correction.response_schema", FIX_ACTIONS, False,
                    detail=f"Response contract failed: {row['error_type']}.")
                    for row in contract_issues or (safe,))
                for name in ("agent.schema_failed", "agent.correction_requested"):
                    self.emit(state, name, evals, severity=Severity.ERROR,
                              task_id=task_id, attempt_number=attempt,
                              error_code=code, safe_dimensions={
                                  "json_path": safe["json_path"],
                                  "error_type": safe["error_type"]})
                state["corrections"][f"{artifact_type}:{code}"] = attempt
                if code == MODEL_OUTPUT_TRUNCATED:
                    instruction = (
                        "Return one complete compact JSON object matching the required schema, "
                        "with all required fields and no surrounding prose. Keep explanations "
                        "concise; avoid repetition. Preserve all scientific facts, existing "
                        "correction instructions, and any immutable reference-repair baseline.")
                    issues = (make_issue(code, "/response", "correction.complete_output",
                                         FIX_ACTIONS, False, detail=instruction),)
                    prior = body.get("correction")
                    correction = deepcopy(dict(prior)) if isinstance(prior, Mapping) else {}
                    previous = correction.get("issues", [])
                    correction["issues"] = [row for row in previous if row != safe] + [safe]
                    correction["output_constraint"] = instruction
                    body = _with_correction(payload, correction | deepcopy(reference_correction))
                    continue
                body = _with_correction(payload, {
                    "issues": list(contract_issues)} | deepcopy(reference_correction))
                continue
            items = ([dict(row) for row in cast(list[Any], result.payload["items"])] if many
                     else [dict(result.payload)])
            if (is_dataclass(validation_ctx) and not isinstance(validation_ctx, type)
                    and hasattr(validation_ctx, "evidence_ids")):
                available = payload.get("available_diagnostics")
                diagnostic_ids = frozenset(
                    str(item) for rows in available.values() for item in rows
                ) if isinstance(available, Mapping) else frozenset()
                offered = self.evidence(state)
                evidence_text = ({key: str(offered[key]) for key in built.allowed_evidence_ids}
                                 if isinstance(offered, Mapping) else {})
                validation_ctx = replace(
                    validation_ctx, evidence_ids=frozenset(built.allowed_evidence_ids),
                    evidence_text=evidence_text, diagnostic_ids=diagnostic_ids)
            local_reference_issues = tuple(
                issue for index, row in enumerate(items)
                for issue in validate_references(
                    model, row, catalogs,
                    prefix=f"/payload/items/{index}" if many else "/payload",
                )
            ) + tuple(
                issue for index, requirement in enumerate(result.missing_requirements)
                for issue in validate_references(
                    ContextRequirementV1, requirement.model_dump(mode="python"), catalogs,
                    prefix=f"/missing_requirements/{index}",
                )
            )
            wall_issues = tuple(
                issue for row in items
                for issue in self.validate(
                    spec.wall, kind, model, result.model_copy(update={"payload": row}),
                    validation_ctx).issues)
            local_reference_keys = {
                (issue.code, issue.rule_id, issue.artifact_ids)
                for issue in local_reference_issues
            }
            issues = local_reference_issues + tuple(
                issue for issue in wall_issues
                if (issue.code, issue.rule_id, issue.artifact_ids)
                not in local_reference_keys)
            current_projection = _decision_projection(
                model, items, result, editable_excerpt_paths=editable_excerpt_paths, many=many)
            current_arrays: set[str] = set()
            current_references = _decision_reference_snapshot(
                model, items, result, many=many, array_paths=current_arrays)
            if (reference_semantic_baseline is not None
                    and current_projection != reference_semantic_baseline):
                issues += (make_issue(
                    "reference_repair_changed_semantics", "/payload",
                    "correction.reference_only", FIX_ACTIONS, False,
                    detail="Restore the prior decision fields; this correction may change only "
                           "the listed typed references and their explicitly eligible citation "
                           "excerpts."),)
            if reference_semantic_baseline is not None:
                issues += _reference_repair_issues(
                    reference_value_baseline, current_references, editable_reference_paths,
                    reference_array_paths)
            parsed_items: tuple[BaseModel, ...] = ()
            if not issues:
                parsed_items = tuple(parse_strict(model, row) for row in items)
                if precommit_admission is not None:
                    issues = precommit_admission(parsed_items, result)
            if not issues:
                # A rejected draft has no authority to mutate durable requirement state.  In
                # particular, requirements that contradicted that same draft used to survive a
                # successful correction and trigger stale clarification interrupts later.
                requirements = _requirements_with_availability(
                    result.missing_requirements, availability)
                self.upsert(
                    requirements, state["analysis_id"],
                    state["design_revision"], state, items)
                state["open_requirement_ids"] = sorted({*state["open_requirement_ids"], *(
                    row.requirement_id for row in requirements
                    if not self.requirements or row.requirement_id in self.requirements)})
                done = tuple((parsed, self.commit(
                    state, artifact_type, row, parents))
                    for parsed, row in zip(parsed_items, items, strict=True))
                self.record(state, task_id, kind, scope_ids, digest, built,
                            result.status.value, attempt, done[-1][1].artifact_id)
                self.emit(state, "task.completed", evals, task_id=task_id,
                          status=result.status.value)
                return done
            # A successful reference repair can expose a later validation wall. Its
            # requested semantic correction must not inherit the completed repair's freeze.
            # Keep the freeze while references or the repair itself still fail validation.
            if reference_semantic_baseline is not None and not any(
                    issue.rule_id in {_REFERENCE_RULE, "correction.reference_only"}
                    for issue in issues) and not _citation_quote_failed(issues, editable_excerpt_paths):
                reference_semantic_baseline = None
                editable_excerpt_paths = frozenset()
                reference_correction = {}
            for issue in issues:
                state["corrections"][f"{artifact_type}:{issue.code}"] = attempt
            if (reference_semantic_baseline is None and issues and all(
                    issue.rule_id == _REFERENCE_RULE for issue in issues)):
                reference_value_baseline = current_references
                reference_array_paths = frozenset(current_arrays)
                editable_reference_paths = frozenset(
                    issue.json_path for issue in issues)
                editable_excerpt_paths = _citation_excerpt_paths(
                    model, items, editable_reference_paths, many=many)
                reference_semantic_baseline = _decision_projection(
                    model, items, result, editable_excerpt_paths=editable_excerpt_paths, many=many)
                reference_correction = {
                    "baseline_decision": result.model_dump(mode="json", include={
                        "status", "missing_requirements", "conflicts", "warnings"}) | {
                        "payload": deepcopy({"items": items} if many else items[0])},
                    "editable_reference_paths": sorted(editable_reference_paths),
                    "editable_excerpt_paths": sorted(editable_excerpt_paths),
                    "instruction": "Return the complete baseline_decision. Replace only the "
                    "listed editable_reference_paths, or delete their invalid array entries. "
                    "Only for an invalid source_interpretations evidence_id, its sibling quote "
                    "listed in editable_excerpt_paths may change with the citation: copy an "
                    "exact span from the repaired task-local source. Preserve fact_key, value "
                    "and relation; do not delete or reorder source_interpretations rows. "
                    "Preserve all valid reference values in the same order and multiplicity. "
                    "Preserve all other fields exactly, "
                    "including status, missing_requirements, conflicts, and warnings."}
            self.emit(state, "artifact.validation_failed", evals, task_id=task_id,
                      severity=Severity.WARNING, error_code=issues[0].code,
                      safe_dimensions={"json_path": issues[0].json_path,
                                       "rule_id": issues[0].rule_id})
            self.emit(state, "agent.correction_requested", evals, task_id=task_id,
                      attempt_number=attempt, error_code=issues[0].code,
                      safe_dimensions={"json_path": issues[0].json_path,
                                       "rule_id": issues[0].rule_id})
            body = _with_correction(payload, {"issues": [
                issue.model_dump(mode="json") for issue in issues]} | (
                deepcopy(reference_correction) or {
                    "failing_payload": items if many else items[0],
                    "failing_decision": result.model_dump(mode="json", include={
                        "status", "missing_requirements", "conflicts", "warnings"}) | {
                        "payload": deepcopy({"items": items} if many else items[0])}}))
        self.exhausted(state, task_id, kind, issues)
        return None

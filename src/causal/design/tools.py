"""The model-facing design tool surface and its allowlist router (PRD-002 §15; SC §5.4)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Final, Protocol

from pydantic import BaseModel

from causal.design.contracts import AvailabilityRowV1, DesignContextManifestV1
from causal.design.entry import ProductsReader
from causal.shared.envelope import AgentTaskEnvelopeV1
from causal.shared.events import EventEmitter, OperationalEventV1, Severity, Stage, build_event

__all__ = [
    "MEASURED_FACT_KEYS", "RETRIEVAL_TOOL_IDS", "MethodPackSource", "ToolError",
    "ToolHandler", "ToolResult", "ToolRouter", "make_retrieval_handlers",
    "tool_allowlists",
]

TOOL_DENIED: Final = "tool_denied"
TOOL_FAILED: Final = "tool_failed"
UNKNOWN_TOOL: Final = "unknown_tool"

COMPONENT_ID: Final = "design-harness"
TOOL_REGISTRY_VERSION: Final = "design-tools.v1"
TOOL_EVAL_IDS: Final = ("EV-SYS-002",)
RETRIEVAL_TOOL_IDS: Final = (
    "list_intake_inventory", "get_semantic_evidence", "get_measured_facts",
    "get_provenance", "get_method_contract")
# The bounded per-column profile facts a model task may see (PRD-002 §15).
MEASURED_FACT_KEYS: Final = (
    "dtype", "null_count", "null_rate", "cardinality", "all_null", "constant")

ToolHandler = Callable[[AgentTaskEnvelopeV1, Mapping[str, Any]], dict[str, Any]]


class ToolError(ValueError):
    """A tool call was refused or failed. `code` is a stable contract value."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class MethodPackSource(Protocol):
    """The validated method-pack lookup `get_method_contract` reads from."""

    def get(self, method_id: str) -> object: ...


@dataclass(frozen=True)
class ToolResult:
    """One completed tool call: bounded scalars and lists, never rows or documents."""

    tool_id: str
    status: str
    result: dict[str, Any]


def tool_allowlists(
    document: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], dict[str, bool]]:
    """Split a parsed `design-tools.v1` document into allowlist and registration maps."""
    rows = next((value for value in document.values() if isinstance(value, list)), [])
    return (
        {str(r["tool_id"]): tuple(str(k) for k in r["allowed_task_kinds"]) for r in rows},
        {str(r["tool_id"]): bool(r["registered"]) for r in rows},
    )


class ToolRouter:
    """Enforcement order: listed, registered, task-kind allowlisted, envelope, handler."""

    def __init__(
        self,
        allowlists: Mapping[str, tuple[str, ...]],
        registered: Mapping[str, bool],
        handlers: Mapping[str, ToolHandler],
        emitter: EventEmitter,
        clock: Callable[[], datetime],
        component_version: str = "design-harness.v1",
    ) -> None:
        self._allowlists = dict(allowlists)
        self._registered = dict(registered)
        self._handlers = dict(handlers)
        self._emitter = emitter
        self._clock = clock
        self._component_version = component_version
        self._counts: dict[str, int] = {}

    def _event(
        self, envelope: AgentTaskEnvelopeV1, name: str, tool_id: str, **overrides: object
    ) -> OperationalEventV1:
        count = self._counts.get(envelope.envelope_id, 0) + 1
        self._counts[envelope.envelope_id] = count
        return build_event(
            occurred_at_utc=self._clock(), event_name=name,
            event_id=f"evt:{envelope.envelope_id}:tool:{count}",
            analysis_id=envelope.analysis_id, stage=Stage.DESIGN,
            stage_run_id=envelope.stage_run_id, task_id=envelope.task_id,
            attempt_id=envelope.attempt_id, component_id=COMPONENT_ID,
            component_version=self._component_version,
            versions={"tool": TOOL_REGISTRY_VERSION}, required_eval_ids=TOOL_EVAL_IDS,
            safe_dimensions={"tool_id": tool_id, "task_kind": envelope.task_kind},
            **overrides,
        )

    def _deny(
        self, envelope: AgentTaskEnvelopeV1, tool_id: str, code: str, reason: str
    ) -> ToolError:
        self._emitter.emit(self._event(
            envelope, "tool.denied", tool_id, severity=Severity.ERROR, status="denied",
            error_code=code, retryable=False))
        return ToolError(f"tool {tool_id!r} denied: {reason}", code)

    def call(
        self, envelope: AgentTaskEnvelopeV1, tool_id: str, arguments: Mapping[str, Any]
    ) -> ToolResult:
        """Run one allowlisted tool; every refusal emits `tool.denied` and calls no handler."""
        allowed = self._allowlists.get(tool_id)
        if allowed is None:
            raise self._deny(envelope, tool_id, UNKNOWN_TOOL, "not listed in the registry")
        if not self._registered.get(tool_id, False):
            raise self._deny(envelope, tool_id, TOOL_DENIED, "not registered in this version")
        if envelope.task_kind not in allowed:
            raise self._deny(
                envelope, tool_id, TOOL_DENIED,
                f"task kind {envelope.task_kind!r} is not allowlisted")
        if tool_id not in envelope.allowed_tool_ids:
            raise self._deny(envelope, tool_id, TOOL_DENIED, "absent from the envelope")
        handler = self._handlers.get(tool_id)
        if handler is None:
            raise self._deny(envelope, tool_id, UNKNOWN_TOOL, "no handler is bound")
        self._emitter.emit(self._event(envelope, "tool.started", tool_id))
        try:
            result = handler(envelope, arguments)
        except Exception as error:  # any handler failure is exactly one tool failure
            self._emitter.emit(self._event(
                envelope, "tool.failed", tool_id, severity=Severity.ERROR, status="failed",
                error_code=TOOL_FAILED, retryable=False,
                exception_class=type(error).__name__))
            raise ToolError(f"tool {tool_id!r} failed: {error}", TOOL_FAILED) from error
        self._emitter.emit(self._event(envelope, "tool.completed", tool_id, status="completed"))
        return ToolResult(tool_id=tool_id, status="completed", result=result)


def _surface(rows: tuple[AvailabilityRowV1, ...], table_name: str) -> list[dict[str, Any]]:
    return [
        {
            "scope_kind": row.scope_kind, "table_name": row.table_name,
            "column_name": row.column_name, "field_or_slot_name": row.field_or_slot_name,
            "status": row.status, "evidence_count": row.evidence_count,
            "json_pointer": row.json_pointer,
        }
        for row in rows
        if row.table_name is None or row.table_name == table_name
    ]


def _pack_dict(pack: object) -> dict[str, Any]:
    """One method-pack row as plain scalars and lists."""
    if isinstance(pack, BaseModel):
        return dict(pack.model_dump(mode="json"))
    if isinstance(pack, Mapping):
        return {str(key): value for key, value in pack.items()}
    raise ToolError(f"unusable method pack of type {type(pack).__name__}", TOOL_FAILED)


def make_retrieval_handlers(
    manifest: DesignContextManifestV1,
    products: ProductsReader,
    profiles: Mapping[str, Mapping[str, Any]],
    evidence_payload: Mapping[str, Any],
    packs: MethodPackSource,
) -> dict[str, ToolHandler]:
    """The five PRD-002 §15 retrieval handlers, each bounded by an already frozen surface."""
    items = evidence_payload.get("items") or ()
    context = RetrievalContext(
        manifest=manifest, products=products, profiles=profiles,
        evidence={str(i["evidence_id"]): i for i in items if isinstance(i, Mapping)},
        packs=packs,
    )
    return {
        "list_intake_inventory": partial(_list_intake_inventory, context),
        "get_semantic_evidence": partial(_get_semantic_evidence, context),
        "get_measured_facts": partial(_get_measured_facts, context),
        "get_provenance": partial(_get_provenance, context),
        "get_method_contract": partial(_get_method_contract, context),
    }


@dataclass(frozen=True)
class RetrievalContext:
    """The frozen surfaces the retrieval handlers may read; nothing else is reachable."""

    manifest: DesignContextManifestV1
    products: ProductsReader
    profiles: Mapping[str, Mapping[str, Any]]
    evidence: Mapping[str, Mapping[str, Any]]
    packs: MethodPackSource


def _list_intake_inventory(
    ctx: RetrievalContext, envelope: AgentTaskEnvelopeV1, arguments: Mapping[str, Any]
) -> dict[str, Any]:
    table = str(arguments["table_name"])
    if table != ctx.manifest.selected_table:
        raise ValueError(f"{table!r} is not the selected table")
    return {
        "table_name": table,
        "structural": [
            {"column_name": row.column_name, "dtype": row.dtype, "ordinal": row.ordinal}
            for row in ctx.manifest.structural_inventory
            if row.table_name == table
        ],
        "available": _surface(ctx.manifest.semantic_available, table),
        "missing": _surface(ctx.manifest.semantic_missing, table),
        "measured": _surface(ctx.manifest.measured_surface, table),
        "provenance": _surface(ctx.manifest.provenance_surface, table),
    }


def _get_semantic_evidence(
    ctx: RetrievalContext, envelope: AgentTaskEnvelopeV1, arguments: Mapping[str, Any]
) -> dict[str, Any]:
    requested = [str(value) for value in arguments["evidence_ids"]]
    allowed = set(envelope.allowed_evidence_ids)
    readable = [key for key in requested if key in allowed and key in ctx.evidence]
    return {
        "items": [dict(ctx.evidence[key]) for key in readable],
        "missing": [key for key in requested if key not in readable],
    }


def _get_measured_facts(
    ctx: RetrievalContext, envelope: AgentTaskEnvelopeV1, arguments: Mapping[str, Any]
) -> dict[str, Any]:
    table = str(arguments["table_name"])
    profile = ctx.profiles.get(table)
    if profile is None:
        raise ValueError(f"no committed profile for {table!r}")
    raw = profile.get("columns")
    columns: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    requested = [str(value) for value in arguments["column_names"]]
    facts = [
        {"column_name": name}
        | {key: columns[name].get(key) for key in MEASURED_FACT_KEYS}
        | {"hypotheses": [
            str(item.get("kind")) for item in columns[name].get("hypotheses") or ()
        ]}
        for name in requested
        if name in columns
    ]
    return {
        "table_name": table, "facts": facts,
        "missing": [name for name in requested if name not in columns],
    }


def _get_provenance(
    ctx: RetrievalContext, envelope: AgentTaskEnvelopeV1, arguments: Mapping[str, Any]
) -> dict[str, Any]:
    found = ctx.products.load_envelope(str(arguments["artifact_id"]))
    return {
        "artifact_id": found.artifact_id, "artifact_type": found.artifact_type,
        "schema_version": found.schema_version, "content_hash": found.content_hash,
        "producer_component": found.producer_component,
        "producer_version": found.producer_version,
        "sensitivity_class": found.sensitivity_class.value,
        "payload_locator": found.payload_locator,
        "parent_artifact_ids": [ref.artifact_id for ref in found.parent_artifacts],
        "parent_content_hashes": [ref.content_hash for ref in found.parent_artifacts],
    }


def _get_method_contract(
    ctx: RetrievalContext, envelope: AgentTaskEnvelopeV1, arguments: Mapping[str, Any]
) -> dict[str, Any]:
    return _pack_dict(ctx.packs.get(str(arguments["method_id"])))

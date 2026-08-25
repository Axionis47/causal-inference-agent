"""The model-facing design retrieval handlers (PRD-002 §15; router shared per D-049)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Final, Protocol

from pydantic import BaseModel

from causal.design.contracts import AvailabilityRowV1, DesignContextManifestV1
from causal.shared.envelope import AgentTaskEnvelopeV1
from causal.shared.readers import ProductsReader
from causal.shared.toolrouter import (
    TOOL_FAILED,
    ToolError,
    ToolHandler,
    ToolResult,
    ToolRouter,
    tool_allowlists,
)

__all__ = [
    "MEASURED_FACT_KEYS", "RETRIEVAL_TOOL_IDS", "MethodPackSource", "ToolError",
    "ToolHandler", "ToolResult", "ToolRouter", "make_retrieval_handlers",
    "tool_allowlists",
]

RETRIEVAL_TOOL_IDS: Final = (
    "list_intake_inventory", "get_semantic_evidence", "get_measured_facts",
    "get_provenance", "get_method_contract")
# The bounded per-column profile facts a model task may see (PRD-002 §15).
MEASURED_FACT_KEYS: Final = (
    "dtype", "null_count", "null_rate", "cardinality", "all_null", "constant")


class MethodPackSource(Protocol):
    """The validated method-pack lookup `get_method_contract` reads from."""

    def get(self, method_id: str) -> object: ...


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

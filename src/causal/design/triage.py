"""Deterministic column triage `triage.v1` and its committed record (T-011 §2–§3; D-046)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from causal.design.contracts import (
    AvailabilityRowV1,
    ConceptProposalV1,
    DesignContextManifestV1,
    DesignIntentV1,
    _Payload,
    _require_exact_keys,
    _Row,
)
from causal.shared.contracts import Identity

__all__ = [
    "BATCH_COLUMN_LIMIT", "MAX_BATCHES", "TIER2_HYPOTHESIS_KINDS", "TIER_KEYS",
    "TRIAGE_RULE_VERSION", "ColumnTriageRecordV1", "TriageBatchV1", "build_batches",
    "normalize_column_name", "triage",
]

TRIAGE_RULE_VERSION: Final = "triage.v1"
TIER_KEYS: Final = ("critical", "plausible_adjustment", "supporting", "unused")
MAX_BATCHES: Final = 8
# Columns per batch before the cap forces wider batches (T-011 §3 "one batch per ⌈n/limit⌉").
BATCH_COLUMN_LIMIT: Final = 4
# D-025: the profiler emits only these two hypothesis kinds; either one holds a column at tier 2.
TIER2_HYPOTHESIS_KINDS: Final = ("missing_sentinel", "identifier")


class TriageBatchV1(_Row):
    """One frozen worker batch over critical and plausible-adjustment columns."""

    batch_id: Identity
    column_names: Annotated[tuple[Identity, ...], Field(min_length=1)]


class ColumnTriageRecordV1(_Payload):
    """The frozen tier assignment for one table (`column-triage.v1`, D-046)."""

    schema_version: Literal["column-triage.v1"] = "column-triage.v1"
    triage_rule_version: Literal["triage.v1"] = "triage.v1"
    table_name: Identity
    tiers: dict[str, tuple[Identity, ...]]
    batches: tuple[TriageBatchV1, ...]
    deferred: tuple[Identity, ...]
    match_trace: dict[str, str]

    @model_validator(mode="after")
    def _tiers_partition_the_inventory(self) -> Self:
        _require_exact_keys(self.tiers, TIER_KEYS, "tiers")
        seen: set[str] = set()
        for key in TIER_KEYS:
            members = self.tiers[key]
            if len(set(members)) != len(members) or tuple(sorted(members)) != members:
                raise ValueError(f"tier {key} must be a sorted tuple of distinct columns")
            overlap = sorted(seen & set(members))
            if overlap:
                raise ValueError(f"tier {key} overlaps an earlier tier: {overlap}")
            seen |= set(members)
        if set(self.match_trace) != seen:
            raise ValueError("match_trace must name exactly the triaged columns")
        return self

    @model_validator(mode="after")
    def _batches_and_deferral(self) -> Self:
        if len(self.batches) > MAX_BATCHES:
            raise ValueError(f"at most {MAX_BATCHES} batches, got {len(self.batches)}")
        batchable = set(self.tiers["critical"]) | set(self.tiers["plausible_adjustment"])
        batched: set[str] = set()
        for batch in self.batches:
            members = set(batch.column_names)
            if members - batchable:
                raise ValueError(
                    f"batch {batch.batch_id} carries non-batchable columns: "
                    f"{sorted(members - batchable)}"
                )
            if members & batched:
                repeated = sorted(members & batched)
                raise ValueError(f"batch {batch.batch_id} repeats columns: {repeated}")
            batched |= members
        if self.deferred != tuple(sorted((*self.tiers["supporting"], *self.tiers["unused"]))):
            raise ValueError("deferred must equal sorted(supporting + unused)")
        return self


def normalize_column_name(name: str) -> str:
    """Fold a column or candidate name to its triage comparison key (T-011 §3)."""
    return name.strip().casefold().replace(" ", "_").replace("-", "_")


def _intent_columns(intent: DesignIntentV1) -> frozenset[str]:
    """Every column named by an intent concept proposal, normalized."""
    proposals: tuple[ConceptProposalV1, ...] = (
        intent.treatment, intent.outcome, intent.population, intent.comparator,
        intent.unit, intent.timeframe, *intent.mandatory_concepts,
    )
    return frozenset(
        normalize_column_name(column)
        for proposal in proposals for column in proposal.candidate_columns
    )


def _surface_columns(
    rows: tuple[AvailabilityRowV1, ...], slot: str | None = None, status: str | None = None
) -> frozenset[str]:
    """Normalized column keys on one availability surface, optionally slot/status filtered."""
    return frozenset(
        normalize_column_name(row.column_name)
        for row in rows
        if row.scope_kind == "column" and row.column_name is not None
        and (slot is None or row.field_or_slot_name == slot)
        and (status is None or row.status == status)
    )


def _classify(
    column: str, intent_columns: frozenset[str], evidenced: frozenset[str],
    flagged: frozenset[str], measured: frozenset[str],
) -> tuple[str, str]:
    """Apply the five `triage.v1` rules in order; first match wins."""
    if column in intent_columns:
        return "intent_candidate", "critical"
    if column in evidenced:
        return "evidenced_meaning", "plausible_adjustment"
    if column in flagged:
        return "flagged_by_profile", "plausible_adjustment"
    if column in measured:
        return "profiled_only", "supporting"
    return "no_signal", "unused"


def _batch(table_name: str, members: tuple[str, ...]) -> TriageBatchV1:
    """One batch with a deterministic `tb:` id over its table and sorted members (D-031)."""
    material = table_name + "\x1f" + "\x1f".join(sorted(members))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return TriageBatchV1(batch_id=f"tb:{digest[:16]}", column_names=members)


def build_batches(
    table_name: str, critical: Sequence[str], plausible_adjustment: Sequence[str]
) -> tuple[TriageBatchV1, ...]:
    """Group critical then plausible columns, inventory order kept, into ≤ 8 batches."""
    members = (*critical, *plausible_adjustment)
    if not members:
        return ()
    size = max(BATCH_COLUMN_LIMIT, -(-len(members) // MAX_BATCHES))
    return tuple(
        _batch(table_name, members[start:start + size]) for start in range(0, len(members), size)
    )


def triage(
    intent: DesignIntentV1,
    manifest: DesignContextManifestV1,
    hypothesis_columns: Mapping[str, tuple[str, ...]] | None = None,
) -> ColumnTriageRecordV1:
    """Assign every structural column of the selected table to a tier (no model, no data read)."""
    intent_columns = _intent_columns(intent)
    evidenced = _surface_columns(manifest.semantic_available, "meaning", "evidenced")
    measured = _surface_columns(manifest.measured_surface)
    flagged = frozenset(
        normalize_column_name(column)
        for column, kinds in (hypothesis_columns or {}).items()
        if any(kind in TIER2_HYPOTHESIS_KINDS for kind in kinds)
    )
    tiers: dict[str, list[str]] = {key: [] for key in TIER_KEYS}
    ordered: dict[str, list[str]] = {"critical": [], "plausible_adjustment": []}
    match_trace: dict[str, str] = {}
    inventory = sorted(
        (row for row in manifest.structural_inventory if row.table_name == manifest.selected_table),
        key=lambda row: row.ordinal,
    )
    for row in inventory:
        rule, tier = _classify(
            normalize_column_name(row.column_name), intent_columns, evidenced, flagged, measured
        )
        tiers[tier].append(row.column_name)
        match_trace[row.column_name] = rule
        if tier in ordered:
            ordered[tier].append(row.column_name)
    frozen = {key: tuple(sorted(columns)) for key, columns in tiers.items()}
    return ColumnTriageRecordV1(
        table_name=manifest.selected_table, tiers=frozen,
        batches=build_batches(
            manifest.selected_table, ordered["critical"], ordered["plausible_adjustment"]
        ),
        deferred=tuple(sorted((*frozen["supporting"], *frozen["unused"]))),
        match_trace=match_trace,
    )

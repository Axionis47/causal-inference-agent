"""Task table, prompt rendering, envelope assembly, MeasurementMap compiler (T-013 §1 items 1–3)."""

# PRD-002 §9.3 worker envelope, §9.5 deterministic MeasurementMap compiler, §12.2 concept graph,
# §16.1 result contract. Error codes: invalid_registry_file (from `_parse`), duplicate_task,
# unknown_task_kind.

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from causal.design.contracts import (
    ConceptProposalV1,
    DesignContextManifestV1,
    DesignIntentV1,
    _Payload,
    _Row,
)
from causal.design.packs import TASK_KINDS, PackRegistryError, TaskKind, _parse
from causal.design.semantics import (
    ColumnSemanticCardV1,
    ConceptStatus,
    ConceptV1,
    MeasurementLinkV1,
    MeasurementMapV1,
    MeasurementRelation,
)
from causal.shared.contracts import ArtifactRef, Identity
from causal.shared.envelope import AgentTaskEnvelopeV1, TaskBudgets, TaskStatus

__all__ = ["BATCH_COLUMN_LIMIT", "MAX_BATCHES", "TIER2_HYPOTHESIS_KINDS",
           "ColumnTriageRecordV1", "TaskSpecV1", "TriageBatchV1",
           "build_batches", "build_task_envelope", "compile_measurement_map", "load_task_table",
           "normalize_column_name", "render_prompt", "triage"]

# Even two 12-slot cards can exhaust the provider output budget (NHEFS education/exercise).
# Bound each response to one card and retain the existing total capacity of 24 columns.
MAX_BATCHES: Final = 24
BATCH_COLUMN_LIMIT: Final = 1
TIER2_HYPOTHESIS_KINDS: Final = ("missing_sentinel", "identifier")


class TriageBatchV1(_Row):
    batch_id: Identity
    column_names: Annotated[
        tuple[Identity, ...], Field(min_length=1, max_length=BATCH_COLUMN_LIMIT)]


class ColumnTriageRecordV1(_Payload):
    """The bounded semantic-work scope for one selected table."""

    schema_version: Literal["column-triage.v1"] = "column-triage.v1"
    table_name: Identity
    batches: tuple[TriageBatchV1, ...]
    deferred: tuple[Identity, ...]

    @model_validator(mode="after")
    def _bounded_partition(self) -> Self:
        if len(self.batches) > MAX_BATCHES:
            raise ValueError(f"at most {MAX_BATCHES} batches, got {len(self.batches)}")
        batched = tuple(name for batch in self.batches for name in batch.column_names)
        if len(batched) != len(set(batched)) or set(batched) & set(self.deferred):
            raise ValueError("batched and deferred columns must be distinct")
        if self.deferred != tuple(sorted(set(self.deferred))):
            raise ValueError("deferred columns must be sorted and distinct")
        return self


def normalize_column_name(name: str) -> str:
    return name.strip().casefold().replace(" ", "_").replace("-", "_")


def _intent_columns(intent: DesignIntentV1) -> frozenset[str]:
    proposals: tuple[ConceptProposalV1, ...] = (
        intent.treatment, intent.outcome, intent.population, intent.comparator,
        intent.unit, intent.timeframe, *intent.mandatory_concepts)
    return frozenset(normalize_column_name(column) for proposal in proposals
                     for column in proposal.candidate_columns)


def _batch(table_name: str, members: tuple[str, ...]) -> TriageBatchV1:
    material = table_name + "\x1f" + "\x1f".join(sorted(members))
    digest = hashlib.sha256(material.encode()).hexdigest()
    return TriageBatchV1(batch_id=f"tb:{digest[:16]}", column_names=members)


def build_batches(table_name: str, critical: Sequence[str],
                  plausible_adjustment: Sequence[str]) -> tuple[TriageBatchV1, ...]:
    # Both dimensions are hard ceilings: never preserve the fan-out cap by silently widening
    # a model response. Overflow remains deferred in the triage record for downstream routing.
    members = (*critical, *plausible_adjustment)[:MAX_BATCHES * BATCH_COLUMN_LIMIT]
    if not members:
        return ()
    return tuple(_batch(table_name, members[start:start + BATCH_COLUMN_LIMIT])
                 for start in range(0, len(members), BATCH_COLUMN_LIMIT))


def triage(intent: DesignIntentV1, manifest: DesignContextManifestV1,
           hypothesis_columns: Mapping[str, tuple[str, ...]] | None = None,
           ) -> ColumnTriageRecordV1:
    """Assign every selected-table column to one bounded processing tier."""
    intent_columns = _intent_columns(intent)
    evidenced = frozenset(normalize_column_name(row.column_name) for row
                          in manifest.semantic_available if row.scope_kind == "column"
                          and row.column_name is not None and row.field_or_slot_name == "meaning"
                          and row.status == "evidenced")
    flagged = frozenset(normalize_column_name(column)
                        for column, kinds in (hypothesis_columns or {}).items()
                        if any(kind in TIER2_HYPOTHESIS_KINDS for kind in kinds))
    inventory = sorted((row for row in manifest.structural_inventory
                        if row.table_name == manifest.selected_table), key=lambda row: row.ordinal)
    critical = tuple(row.column_name for row in inventory
                     if normalize_column_name(row.column_name) in intent_columns)
    plausible = tuple(row.column_name for row in inventory if row.column_name not in critical
                      and normalize_column_name(row.column_name) in evidenced | flagged)
    batches = build_batches(manifest.selected_table, critical, plausible)
    batched = {name for batch in batches for name in batch.column_names}
    return ColumnTriageRecordV1(
        table_name=manifest.selected_table,
        batches=batches,
        deferred=tuple(sorted(row.column_name for row in inventory
                              if row.column_name not in batched)))


class TaskSpecV1(_Row):
    """One model task kind: its prompt, output contract, highest wall, and spend ceilings."""

    task_kind: TaskKind
    prompt_path: str
    prompt_version: str
    output_artifact_type: str
    output_schema_version: str
    wall: int
    allowed_stopping_states: tuple[TaskStatus, ...]
    allowed_requirement_ids: Annotated[tuple[Identity, ...], Field(min_length=1)]
    token_budget: int
    tool_call_budget: int
    correction_budget: int


class _TaskFileV1(_Row):
    registry_version: Literal["design-tasks.v1"]
    tasks: tuple[TaskSpecV1, ...]


def load_task_table(path: Path) -> dict[str, TaskSpecV1]:
    """The five task specs keyed by task kind; a repeated, missing, or surplus row fails closed."""
    specs = _parse(path, _TaskFileV1).tasks
    by_kind: dict[str, TaskSpecV1] = {spec.task_kind: spec for spec in specs}
    if len(by_kind) != len(specs):
        raise PackRegistryError("duplicate task kind in the task table", "duplicate_task")
    if sorted(by_kind) != sorted(TASK_KINDS):
        raise PackRegistryError(f"expected {sorted(TASK_KINDS)}, got {sorted(by_kind)}",
                                "unknown_task_kind")
    return by_kind


def render_prompt(spec: TaskSpecV1, prompts_root: Path, sections: Mapping[str, object]) -> str:
    """Template text plus one JSON section per key, sorted by key; `prompt_path` is root-relative."""
    template = (prompts_root / spec.prompt_path).read_text(encoding="utf-8")
    compact = spec.task_kind == "causal_context"
    rendered = template + "".join(
        f"\n\n## {key}\n{json.dumps(sections[key], indent=None if compact else 1, sort_keys=True,
                                   separators=(',', ':') if compact else None)}"
        for key in sorted(sections)
    )
    return rendered + ("\n\n## prerequisite_rule\nSettled requirements are authoritative. "
        "Use cited evidence for `resolved`; preserve uncertainty without asking again for "
        "`unknown_accepted`; never return a settled id and canonical scope in "
        "`missing_requirements`." if sections.get("prerequisite_context") else "")


def build_task_envelope(
    spec: TaskSpecV1, *, analysis_id: str, stage_run_id: str, task_id: str, attempt_id: str,
    manifest_ref: ArtifactRef, scope_kind: str, scope_ids: Sequence[str],
    parent_artifacts: Sequence[ArtifactRef], allowed_evidence_ids: Sequence[str],
    allowed_tool_ids: Sequence[str], payload_type: str, payload: Mapping[str, object],
    validator_version: str = "design-validators.v1",
    model_profile_version: str = "vertex-model-profile.v1",
) -> AgentTaskEnvelopeV1:
    """One closed task envelope: the spec's contract and budgets over the harness's allowlists."""
    return AgentTaskEnvelopeV1(
        envelope_id=f"env:{task_id}:{attempt_id}", schema_version="agent-task-envelope.v1",
        analysis_id=analysis_id, stage_run_id=stage_run_id, task_id=task_id, attempt_id=attempt_id,
        context_manifest=manifest_ref, task_kind=spec.task_kind, scope_kind=scope_kind,
        scope_ids=tuple(scope_ids), parent_artifacts=tuple(parent_artifacts),
        allowed_evidence_ids=tuple(allowed_evidence_ids), allowed_tool_ids=tuple(allowed_tool_ids),
        allowed_retrieval_ids=tuple(allowed_tool_ids), prompt_version=spec.prompt_version,
        output_schema_version=spec.output_schema_version, validator_version=validator_version,
        model_profile_version=model_profile_version, payload_type=payload_type,
        budgets=TaskBudgets(token_budget=spec.token_budget, tool_call_budget=spec.tool_call_budget,
                            correction_budget=spec.correction_budget),
        allowed_stopping_states=spec.allowed_stopping_states, payload=dict(payload),
        error_vocabulary=("schema_invalid", "tool_denied", "correction_exhausted"),
        forbidden_payload_classes=("raw_rows", "dataframe", "archive_bytes", "provider_response",
                                   "credentials"),
    )


def _link(concept_id: str, card: ColumnSemanticCardV1) -> MeasurementLinkV1:
    """One column-to-concept link: `measures` when the card names the concept, else `proxies`."""
    direct = card.concept_id == concept_id
    return MeasurementLinkV1(concept_id=concept_id, table_name=card.table_name,
                             column_name=card.column_name, notes="direct" if direct else "proxy",
                             timing=card.timing,
                             relation=MeasurementRelation.MEASURES if direct
                             else MeasurementRelation.PROXIES)


def compile_measurement_map(intent: DesignIntentV1,
                            cards: Sequence[ColumnSemanticCardV1]) -> MeasurementMapV1:
    """Concepts and links from validated cards and intent proposals; no model, no claims (§9.5)."""
    pairs = [(cid, card) for card in cards if (cid := card.concept_id) is not None]
    concepts = {cid: ConceptV1(concept_id=cid, name=cid, status=ConceptStatus.OBSERVED,
                               description="named by a validated column semantic card")
                for cid, _ in pairs}
    # One concept per intent proposal, keyed by a slug of its name; unmeasured ones are kept.
    for proposal in (intent.treatment, intent.outcome, intent.population, intent.comparator,
                     intent.unit, intent.timeframe, *intent.mandatory_concepts):
        concept_id = "c:" + normalize_column_name(proposal.name)
        matched = [card for card in cards if card.column_name in proposal.candidate_columns]
        pairs += [(concept_id, card) for card in matched]
        concepts[concept_id] = ConceptV1(
            concept_id=concept_id, name=proposal.name, description=proposal.description,
            status=(ConceptStatus.OBSERVED if any(c.concept_id == concept_id for c in matched)
                    else ConceptStatus.PROXY_MEASURED if matched else ConceptStatus.UNMEASURED))
    links = sorted({_link(cid, card) for cid, card in pairs},
                   key=lambda k: (k.concept_id, k.table_name, k.column_name, k.relation))
    return MeasurementMapV1(concepts=tuple(concepts[key] for key in sorted(concepts)),
                            links=tuple(links))

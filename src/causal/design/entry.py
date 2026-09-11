"""Design entry gate: CSV candidates, table selection, context manifest (PRD-002 §5, §6)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from langgraph.types import interrupt
from psycopg import Connection

from causal.design.contracts import (
    AvailabilityRowV1,
    DesignContextManifestV1,
    InterruptKind,
    SelectionSource,
    StructuralFieldV1,
    TableSelectionDecisionV1,
    TableSelectionV1,
)
from causal.design.harness_base import (
    ALLOWED_INTAKE,
    COMPONENT,
    EVAL_STAGE,
    REGISTRY_VERSIONS,
    HarnessBase,
)
from causal.shared import handoff
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, HandoffManifestV1
from causal.shared.readers import CatalogReader, ProductsReader, SqlCatalogReader
from causal.shared.validation import parse_strict

__all__ = [
    "CSV_MEDIA_TYPES",
    "RETRIEVAL_SURFACES",
    "CatalogReader",
    "CsvCandidate",
    "EntryError",
    "EntryNodes",
    "ProductsReader",
    "PsycopgCatalogReader",
    "SelectionRequired",
    "compile_manifest",
    "list_admitted_non_csv",
    "list_csv_candidates",
    "resolve_selection",
    "validate_entry",
]

HANDOFF_UNAVAILABLE: Final = "handoff_unavailable"
ENTRY_VALIDATION_FAILED: Final = "entry_validation_failed"
NO_ANALYSIS_CSV: Final = "NO_ANALYSIS_CSV"
UNSUPPORTED_ANALYSIS_FORMAT_V1: Final = "UNSUPPORTED_ANALYSIS_FORMAT_V1"
# The five PRD-001 §9.3 retrieval views; there is deliberately no all-context view.
RETRIEVAL_SURFACES: Final = (
    "structural_manifest", "semantic_available", "semantic_missing",
    "measured_fact_manifest", "provenance_manifest",
)
CSV_MEDIA_TYPES: Final = frozenset({"csv", "text/csv", "application/csv"})
SUPPORTED_HANDOFF_VERSION: Final = "handoff.v1"
ADMITTED_PARSE_STATUS: Final = "parsed"
# Provider-declared type slots the catalog indexes for a column (PRD-001 §5.7).
_DTYPE_SLOTS: Final = frozenset({"type", "originalType"})


class EntryError(ValueError):
    """A design entry-gate step failed. `code` is a stable contract value."""

    def __init__(
        self, message: str, code: str, detail_codes: tuple[str, ...] = ()
    ) -> None:
        super().__init__(message)
        self.code = code
        self.detail_codes = detail_codes


def PsycopgCatalogReader(conn: Connection[Any]) -> SqlCatalogReader:
    """The design-stage catalog reader: the five PRD-001 views, failing closed (D-049)."""
    return SqlCatalogReader(conn, views=RETRIEVAL_SURFACES, error=lambda view: EntryError(
        f"unsupported retrieval surface {view!r}", ENTRY_VALIDATION_FAILED))


@dataclass(frozen=True)
class CsvCandidate:
    """One admitted CSV resource that could be the analysis table."""

    logical_name: str
    resource_object_locator: str
    sha256: str
    media_type: str
    parse_status: str


@dataclass(frozen=True)
class SelectionRequired:
    """Several admitted CSVs and no decision: the caller owns the durable interrupt."""

    candidates: tuple[CsvCandidate, ...]


def _admitted(catalog_reader: CatalogReader, dataset_id: str) -> tuple[dict[str, Any], ...]:
    return tuple(
        row
        for row in catalog_reader.resources(dataset_id)
        if row["parse_status"] == ADMITTED_PARSE_STATUS
    )


def _is_csv(row: Mapping[str, Any]) -> bool:
    media_type = row["media_type"]
    return isinstance(media_type, str) and media_type.strip().lower() in CSV_MEDIA_TYPES


def list_csv_candidates(
    catalog_reader: CatalogReader, dataset_id: str
) -> tuple[CsvCandidate, ...]:
    """Admitted (`parse_status = 'parsed'`) CSV resources, ordered by logical name."""
    return tuple(
        CsvCandidate(
            logical_name=str(row["logical_name"]),
            resource_object_locator=str(row["object_key"]),
            sha256=str(row["object_sha256"]),
            media_type=str(row["media_type"]),
            parse_status=str(row["parse_status"]),
        )
        for row in _admitted(catalog_reader, dataset_id)
        if _is_csv(row)
    )


def list_admitted_non_csv(catalog_reader: CatalogReader, dataset_id: str) -> tuple[str, ...]:
    """Logical names of admitted resources that are not analysis CSVs."""
    return tuple(
        str(row["logical_name"])
        for row in _admitted(catalog_reader, dataset_id)
        if not _is_csv(row)
    )


def _selection(
    candidate: CsvCandidate, dataset_id: str, count: int, source: SelectionSource,
    decision_artifact_id: str | None,
) -> TableSelectionV1:
    return TableSelectionV1(
        dataset_id=dataset_id,
        logical_name=candidate.logical_name,
        resource_object_locator=candidate.resource_object_locator,
        resource_sha256=candidate.sha256,
        candidate_count=count,
        selection_source=source,
        decision_artifact_id=decision_artifact_id,
    )


def resolve_selection(
    candidates: tuple[CsvCandidate, ...],
    decision: TableSelectionDecisionV1 | None,
    dataset_id: str,
    *,
    other_admitted: tuple[str, ...] = (),
    decision_artifact_id: str | None = None,
) -> TableSelectionV1 | SelectionRequired:
    """Route zero, one, or many admitted CSVs to a selection, an interrupt, or a refusal."""
    if not candidates:
        raise EntryError(f"no admitted analysis CSV in {dataset_id!r}", NO_ANALYSIS_CSV)
    if decision is None:
        if len(candidates) == 1:
            return _selection(
                candidates[0], dataset_id, 1, SelectionSource.ONLY_CANDIDATE, None
            )
        return SelectionRequired(candidates)
    chosen = next(
        (row for row in candidates if row.logical_name == decision.selected_table), None
    )
    if chosen is not None:
        return _selection(
            chosen, dataset_id, len(candidates), SelectionSource.USER_DECISION,
            decision_artifact_id,
        )
    if decision.selected_table in other_admitted:
        raise EntryError(
            f"{decision.selected_table!r} is admitted but is not an analysis CSV",
            UNSUPPORTED_ANALYSIS_FORMAT_V1,
        )
    raise EntryError(
        f"{decision.selected_table!r} names no admitted resource",
        ENTRY_VALIDATION_FAILED, ("unknown_selected_table",),
    )


def _envelope_of(products: ProductsReader, artifact_id: str) -> ArtifactEnvelopeV1 | None:
    try:
        return products.load_envelope(artifact_id)
    except Exception:  # noqa: BLE001 -- any read failure is a missing artifact
        return None


def _referenced_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    single = (
        "question_artifact_id", "source_manifest_artifact_id",
        "evidence_bundle_artifact_id", "semantic_map_artifact_id",
    )
    profiles = payload.get("table_profile_artifact_ids") or ()
    ids = [payload.get(key) for key in single] + list(profiles)
    return tuple(value for value in ids if isinstance(value, str))


def validate_entry(
    handoff: HandoffManifestV1,
    outcome_payload: Mapping[str, Any],
    products: ProductsReader,
    *,
    selection: TableSelectionV1 | None = None,
) -> None:
    """PRD-002 §5 conditions 1–5; every failing condition is reported at once (fail-closed)."""
    codes: set[str] = set()
    if outcome_payload.get("status") not in ("usable", "partial"):
        codes.add("unsupported_intake_status")
    for entry in handoff.entries:
        envelope = _envelope_of(products, entry.artifact_id)
        if envelope is None:
            codes.add("missing_artifact")
        elif envelope.content_hash != entry.content_hash:
            codes.add("artifact_hash_mismatch")
    for artifact_id in _referenced_ids(outcome_payload):
        if _envelope_of(products, artifact_id) is None:
            codes.add("missing_artifact")
    if (
        handoff.schema_version != SUPPORTED_HANDOFF_VERSION
        or outcome_payload.get("handoff_contract_version") != SUPPORTED_HANDOFF_VERSION
    ):
        codes.add("unsupported_handoff_version")
    surfaces = outcome_payload.get("retrieval_surfaces")
    if not isinstance(surfaces, Mapping) or set(surfaces) != set(RETRIEVAL_SURFACES):
        codes.add("unsupported_retrieval_surface")
    if selection is None or selection.dataset_id != outcome_payload.get("dataset_id"):
        codes.add("no_table_selection")
    question = outcome_payload.get("question_artifact_id")
    envelope = _envelope_of(products, question) if isinstance(question, str) else None
    if envelope is None or envelope.artifact_type != "QuestionRecord":
        codes.add("question_unreadable")
    if codes:
        raise EntryError(
            "design entry gate refused the intake handoff",
            ENTRY_VALIDATION_FAILED, tuple(sorted(codes)),
        )


def _structural_inventory(
    rows: tuple[dict[str, Any], ...], table_name: str
) -> tuple[StructuralFieldV1, ...]:
    """One row per column of the selected table, ordered by the catalog's stable ordering."""
    slots: dict[str, set[str]] = {}
    for row in rows:
        if row["scope_kind"] != "column" or row["table_name"] != table_name:
            continue
        column = row["column_name"]
        if column is None:
            continue
        declared = slots.setdefault(str(column), set())
        if row["status"] == "evidenced":
            declared.add(str(row["field_or_slot_name"]))
    return tuple(
        StructuralFieldV1(
            table_name=table_name,
            column_name=column,
            # The catalog indexes slot availability, not slot values (PRD-001 §9.3), so the
            # inventory records whether the source declared a type, not the type literal.
            dtype="declared" if declared & _DTYPE_SLOTS else "undeclared",
            ordinal=ordinal,
        )
        for ordinal, (column, declared) in enumerate(sorted(slots.items()))
    )


def _availability(
    rows: tuple[dict[str, Any], ...], table_name: str
) -> tuple[AvailabilityRowV1, ...]:
    """Dataset-scope and selected-table rows of one availability view."""
    return tuple(
        AvailabilityRowV1(
            scope_kind=row["scope_kind"],
            table_name=row["table_name"],
            column_name=row["column_name"],
            field_or_slot_name=row["field_or_slot_name"],
            status=row["status"],
            # semantic_missing carries neither counter nor pointer: a missing slot has no
            # evidence and no object to point into.
            evidence_count=int(row.get("evidence_count") or 0),
            json_pointer=str(row.get("json_pointer") or ""),
        )
        for row in rows
        if row["table_name"] is None or row["table_name"] == table_name
    )


def compile_manifest(
    catalog_reader: CatalogReader,
    selection: TableSelectionV1,
    *,
    question_ref: ArtifactRef,
    outcome_ref: ArtifactRef,
    selection_ref: ArtifactRef,
    design_revision: int,
    registry_versions: Mapping[str, str],
) -> DesignContextManifestV1:
    """The one immutable context surface every design task reads from (PRD-002 §5)."""
    table = selection.logical_name
    rows = {
        name: catalog_reader.view_rows(name, selection.dataset_id)
        for name in RETRIEVAL_SURFACES
    }
    return DesignContextManifestV1(
        design_revision=design_revision,
        question_artifact=question_ref,
        intake_outcome_artifact=outcome_ref,
        table_selection_artifact=selection_ref,
        selected_table=table,
        structural_inventory=_structural_inventory(rows["structural_manifest"], table),
        semantic_available=_availability(rows["semantic_available"], table),
        semantic_missing=_availability(rows["semantic_missing"], table),
        measured_surface=_availability(rows["measured_fact_manifest"], table),
        provenance_surface=_availability(rows["provenance_manifest"], table),
        retrieval_surfaces=RETRIEVAL_SURFACES,
        registry_versions=dict(registry_versions),
    )


class EntryNodes(HarnessBase):
    """The intake-handoff, table-selection, and context-manifest graph nodes."""

    def entry(self, state: Any) -> dict[str, Any]:
        self._emit(state, "stage.started", EVAL_STAGE)
        manifest = self._handoff_manifest(state)
        payload = self._payload(state["artifacts"]["IntakeOutcome"])
        gate = handoff.HandoffGate(self.deps.objects, self.deps.products,
                                   handoff.HandoffStore(self.deps.conn), self.deps.registry,
                                   self.deps.emitter)
        opened = gate.accept(manifest, COMPONENT, ALLOWED_INTAKE, lambda verdict, codes: self._event(
            state, f"handoff.{verdict}", EVAL_STAGE, status=verdict,
            error_code=codes[0] if codes else None))
        for found in (manifest.entries[0].artifact_id, str(payload["question_artifact_id"])):
            state["hashes"][found] = self.deps.products.load_envelope(found).content_hash
        state["artifacts"]["QuestionRecord"] = str(payload["question_artifact_id"])
        if not opened.accepted:
            return self._fail(state, "handoff_unavailable", opened.error_codes)
        return self._out(state, stage="entry", dataset_id=str(payload["dataset_id"]))

    def selection(self, state: Any) -> dict[str, Any]:
        dataset = state["dataset_id"]
        candidates = list_csv_candidates(self.deps.catalog, dataset)
        try:
            routed = resolve_selection(candidates, None, dataset)
            if isinstance(routed, SelectionRequired):
                anchor = self._ref(state, "IntakeOutcome")
                self._emit(state, "user_interrupt.created", EVAL_STAGE, status="table_selection")
                decision = parse_strict(TableSelectionDecisionV1, interrupt({
                    "kind": InterruptKind.TABLE_SELECTION.value,
                    "interrupt_artifact_id": anchor.artifact_id,
                    "interrupt_hash": anchor.content_hash,
                    "design_revision": state["design_revision"],
                    "candidates": [row.logical_name for row in routed.candidates]}))
                chosen = self._commit(state, "TableSelectionDecision", decision.canonical_payload(),
                                      self._parents(state, "IntakeOutcome"))
                self._emit(state, "user_interrupt.resumed", EVAL_STAGE, status="table_selection")
                routed = resolve_selection(
                    candidates, decision, dataset, decision_artifact_id=chosen.artifact_id,
                    other_admitted=list_admitted_non_csv(self.deps.catalog, dataset))
        except EntryError as error:
            return self._out(state, status="needs_data", error_code=error.code)
        assert isinstance(routed, TableSelectionV1)
        self._commit(state, "TableSelection", routed.canonical_payload(),
                     self._parents(state, "IntakeOutcome", "QuestionRecord"))
        return self._out(state, stage="selection")

    def manifest(self, state: Any) -> dict[str, Any]:
        selection = self._model(state, "TableSelection", TableSelectionV1)
        try:
            validate_entry(self._handoff_manifest(state),
                           self._payload(state["artifacts"]["IntakeOutcome"]),
                           self.deps.products, selection=selection)
        except EntryError as error:
            return self._fail(state, error.code, error.detail_codes)
        compiled = compile_manifest(
            self.deps.catalog, selection, question_ref=self._ref(state, "QuestionRecord"),
            outcome_ref=self._ref(state, "IntakeOutcome"),
            selection_ref=self._ref(state, "TableSelection"),
            design_revision=state["design_revision"], registry_versions=REGISTRY_VERSIONS)
        self._commit(state, "DesignContextManifest", compiled.canonical_payload(),
                     self._parents(state, "TableSelection"))
        return self._out(state, stage="manifest")

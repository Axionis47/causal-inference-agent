"""Design-harness pipeline nodes: entry through method design (T-013 §1.5 split, D-055)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, cast

from langgraph.types import interrupt

from causal.design import askgate, contracts, entry, frame, semantics
from causal.design import compile as compiler
from causal.design.capacity import check_capacity
from causal.design.diagnostics import DIAGNOSTIC_SPECS, run_diagnostic
from causal.design.harness_base import (
    ALLOWED_INTAKE,
    COMPONENT,
    EVAL_ASK,
    EVAL_STAGE,
    REGISTRY_VERSIONS,
    DesignState,
    HarnessBase,
)
from causal.design.triage import ColumnTriageRecordV1, triage
from causal.shared import envelope as agent
from causal.shared import handoff
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactEnvelopeV1
from causal.shared.frames import ROW_UNIT_COLUMN
from causal.shared.readers import CsvObjectFrameSource
from causal.shared.validation import parse_strict

__all__ = ["PipelineNodes"]

CAPACITY_KIND: Final = "DeliveryCapacityCheck"
# The three payloads that must name the row unit identifier once the harness binds it.
ROW_UNIT_KINDS: Final = ("RoleLedger", "RunnableFrameContract", "ExperimentDesign")
ROW_UNIT_ASSUMPTION: Final = (
    "No column identifies a unit and the intent states one row per unit, so each row is treated "
    f"as one unit and {ROW_UNIT_COLUMN!r} is derived from the source row order.")
# `hypothesis`, never `evidenced`: no document names this column, because it does not exist yet.
ROW_UNIT_CLAIM: Final[dict[str, Any]] = {
    "role": semantics.RoleName.UNIT_IDENTIFIER.value, "concept_id": "c:row_unit",
    "column_refs": [ROW_UNIT_COLUMN], "evidence_ids": [], "graph_edge_ids": [],
    "timing": semantics.TimingClass.PRE_TREATMENT.value, "alternatives": [], "methods": [],
    "support_class": agent.SupportClass.CORROBORATED_SOURCE_INFERENCE.value,
    "status": agent.EpistemicStatus.HYPOTHESIS.value}


def _bind_row_unit(kind: str, body: dict[str, Any]) -> dict[str, Any]:
    """Name the derived row identifier in whichever payload carries it."""
    if kind == "RoleLedger":
        return body | {"claims": [*body["claims"], ROW_UNIT_CLAIM]}
    if kind == "RunnableFrameContract":
        return body | {"key_columns": [*body["key_columns"], ROW_UNIT_COLUMN]}
    return body | {"assumptions": [*body["assumptions"], ROW_UNIT_ASSUMPTION]}


class PipelineNodes(HarnessBase):
    """Entry, selection, manifest, and the model-task pipeline through method design."""

    def entry(self, state: DesignState) -> dict[str, Any]:
        """Open the intake handoff through the shared T-006 gate (PRD-002 §5 conditions 2–3)."""
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

    def selection(self, state: DesignState) -> dict[str, Any]:
        """One admitted CSV, or the durable table-selection interrupt (PRD-002 §5, §11.1)."""
        dataset = state["dataset_id"]
        candidates = entry.list_csv_candidates(self.deps.catalog, dataset)
        try:
            routed = entry.resolve_selection(candidates, None, dataset)
            if isinstance(routed, entry.SelectionRequired):
                anchor = self._ref(state, "IntakeOutcome")
                self._emit(state, "user_interrupt.created", EVAL_STAGE, status="table_selection")
                decision = parse_strict(contracts.TableSelectionDecisionV1, interrupt({
                    "kind": contracts.InterruptKind.TABLE_SELECTION.value,
                    "interrupt_artifact_id": anchor.artifact_id,
                    "interrupt_hash": anchor.content_hash,
                    "design_revision": state["design_revision"],
                    "candidates": [row.logical_name for row in routed.candidates]}))
                chosen = self._commit(state, "TableSelectionDecision", decision.canonical_payload(),
                                      self._parents(state, "IntakeOutcome"))
                self._emit(state, "user_interrupt.resumed", EVAL_STAGE, status="table_selection")
                routed = entry.resolve_selection(
                    candidates, decision, dataset, decision_artifact_id=chosen.artifact_id,
                    other_admitted=entry.list_admitted_non_csv(self.deps.catalog, dataset))
        except entry.EntryError as error:
            return self._out(state, status="refused", refusal_code=error.code)
        assert isinstance(routed, contracts.TableSelectionV1)
        self._commit(state, "TableSelection", routed.canonical_payload(),
                     self._parents(state, "IntakeOutcome", "QuestionRecord"))
        return self._out(state, stage="selection")

    def manifest(self, state: DesignState) -> dict[str, Any]:
        """Check §5 conditions 1–5, then compile the one immutable context surface."""
        selection = self._model(state, "TableSelection", contracts.TableSelectionV1)
        try:
            entry.validate_entry(self._handoff_manifest(state),
                                 self._payload(state["artifacts"]["IntakeOutcome"]),
                                 self.deps.products, selection=selection)
        except entry.EntryError as error:
            return self._fail(state, error.code, error.detail_codes)
        compiled = entry.compile_manifest(
            self.deps.catalog, selection, question_ref=self._ref(state, "QuestionRecord"),
            outcome_ref=self._ref(state, "IntakeOutcome"),
            selection_ref=self._ref(state, "TableSelection"),
            design_revision=state["design_revision"], registry_versions=REGISTRY_VERSIONS,
            recipient_map=self.deps.tool_registry.recipient_map())
        self._commit(state, "DesignContextManifest", compiled.canonical_payload(),
                     self._parents(state, "TableSelection"))
        return self._out(state, stage="manifest")

    def intent(self, state: DesignState) -> dict[str, Any]:
        """The single bounded intent task (PRD-002 §8.1)."""
        book = self._model(state, "DesignContextManifest", contracts.DesignContextManifestV1)
        done = self._run_task(
            state, "intent", contracts.DesignIntentV1, scope_kind="design",
            scope_ids=(book.selected_table,), parent_kinds=("DesignContextManifest",),
            payload={"question": self._payload(state["artifacts"]["QuestionRecord"]),
                     "selected_table": book.selected_table,
                     # What the harness counted, so grain and unit identity are read off the
                     # table rather than guessed from its shape (D-103).
                     "table_facts": self._grain(state), "columns": [
                         row.column_name for row in book.structural_inventory]})
        return self._out(state) if done is None else self._out(state, stage="intent")

    def triage_node(self, state: DesignState) -> dict[str, Any]:
        """Deterministic `triage.v1` over the committed table profile; no model runs here."""
        book = self._model(state, "DesignContextManifest", contracts.DesignContextManifestV1)
        hypotheses = {name: tuple(str(item["kind"]) for item in col.get("hypotheses") or ())
                      for name, col in self._profile(state)[1].items()}
        record = triage(
            self._model(state, "DesignIntent", contracts.DesignIntentV1), book, hypotheses)
        self._commit(state, "ColumnTriageRecord", record.canonical_payload(),
                     self._parents(state, "DesignIntent"))
        return self._out(state, stage="triage", pending_task_ids=[
            batch.batch_id for batch in record.batches], counts={
                **state["counts"], "columns_triaged": len(record.match_trace),
                "deferred_columns": len(record.deferred)})

    def semantic(self, state: DesignState) -> dict[str, Any]:
        """One sequential task per frozen batch, one card per assigned column (D-050)."""
        record = self._model(state, "ColumnTriageRecord", ColumnTriageRecordV1)
        cards = list(state["card_ids"])
        for batch in record.batches:
            done = self._run_task(
                state, "semantic_batch", semantics.ColumnSemanticCardV1, scope_kind="column",
                scope_ids=batch.column_names, parent_kinds=("DesignIntent",), many=True,
                ctx=self._ctx(state, triage=record),
                payload={"batch_id": batch.batch_id, "table_name": record.table_name,
                         "columns": list(batch.column_names)})
            if done is None:
                return self._out(state)
            cards.extend(found.artifact_id for _, found in done)
        return self._out(state, stage="semantic", card_ids=cards, counts={
            **state["counts"], "columns_carded": len(cards)})

    def measurement(self, state: DesignState) -> dict[str, Any]:
        """The deterministic MeasurementMap compiler over the validated cards (§9.5)."""
        built = compiler.compile_measurement_map(
            self._model(state, "DesignIntent", contracts.DesignIntentV1),
            [parse_strict(semantics.ColumnSemanticCardV1, self._payload(found))
             for found in state["card_ids"]])
        self._commit(state, "MeasurementMap", built.canonical_payload(),
                     self._parents(state, "DesignIntent"))
        return self._out(state, stage="measurement")

    def roles(self, state: DesignState) -> dict[str, Any]:
        """One role task per frozen batch scope; a worker never widens it (SC §7)."""
        record = self._model(state, "ColumnTriageRecord", ColumnTriageRecordV1)
        mapped = self._model(state, "MeasurementMap", semantics.MeasurementMapV1)
        found_ids = list(state["role_ids"])
        for batch in record.batches:
            concepts = sorted({link.concept_id for link in mapped.links
                               if link.column_name in batch.column_names})
            done = self._run_task(
                state, "role_evidence", semantics.RoleEvidenceV1, scope_kind="relationship",
                scope_ids=concepts or [record.table_name], parent_kinds=("MeasurementMap",),
                # PRD-002 §9.5 routes "assigned concepts/relationships, timing, cited cards" here.
                # The cards hold the meaning, units, levels and timing of these exact columns and
                # were committed one node ago; sending their names alone starved the worker (D-104).
                payload={"batch_id": batch.batch_id, "concept_ids": concepts,
                         "columns": list(batch.column_names),
                         "cards": [card for found in state["card_ids"]
                                   if (card := self._payload(found))["column_name"]
                                   in batch.column_names]})
            if done is None:
                return self._out(state)
            found_ids.extend(found.artifact_id for _, found in done)
        return self._out(state, stage="roles", role_ids=found_ids, counts={
            **state["counts"], "role_tasks": len(found_ids)})

    def synthesis(self, state: DesignState) -> dict[str, Any]:
        """CausalContext then RoleLedger, both behind the wall-5 causal validator (§8.4)."""
        payload: dict[str, object] = {
            "measurement_map": self._payload(state["artifacts"]["MeasurementMap"]),
            "hypotheses": [self._payload(found) for found in state["role_ids"]]}
        done = self._run_task(
            state, "causal_synthesis", semantics.CausalContextV1, scope_kind="design",
            scope_ids=("causal_context",), parent_kinds=("MeasurementMap",), payload=payload)
        if done is None:
            return self._out(state)
        ledger = self._run_task(
            state, "causal_synthesis", semantics.RoleLedgerV1, scope_kind="design",
            scope_ids=("role_ledger",), parent_kinds=("CausalContext",), commits="RoleLedger",
            ctx=self._ctx(state, causal_context=done[0][0]),
            payload=payload | {"causal_context": self._ref(state, "CausalContext").model_dump(
                mode="json")})
        return self._out(state) if ledger is None else self._out(state, stage="synthesis")

    def requirements_gate(self, state: DesignState) -> dict[str, Any]:
        """Freeze, route, and ask at most twice per revision (SC §6.2; PRD-002 §11)."""
        frozen = askgate.freeze_requirements(self._open_requirements(state))
        round_number = state["clarification_round"] + 1
        decisions = askgate.gate(frozen, {}, round_number, self.deps.templates)
        settled = {askgate.GateRoute.RESOLVED: askgate.RequirementState.RESOLVED,
                   askgate.GateRoute.RECORD_SENSITIVITY: askgate.RequirementState.UNKNOWN_ACCEPTED}
        for row in decisions:
            if row.route in settled:
                self.requirements.set_state(state["analysis_id"], state["design_revision"],
                                            row.requirement_id, row.scope_id, settled[row.route],
                                            None)
        asking = {row.requirement_id for row in decisions if row.route is askgate.GateRoute.ASK}
        if not asking:
            terminal = [row for row in decisions if row.route.value.startswith("terminal")]
            if not terminal:
                return self._out(state, stage="gate")
            refused = terminal[0].route is askgate.GateRoute.TERMINAL_REFUSED
            return self._out(state, status="refused" if refused else "needs_context",
                             refusal_code="UNSUPPORTED_IDENTIFICATION" if refused else None)
        return self._ask(state, frozen, asking, round_number)

    def _ask(self, state: DesignState, frozen: tuple[agent.ContextRequirementV1, ...],
             asking: set[str], round_number: int) -> dict[str, Any]:
        """One clarification round: packet, durable interrupt, typed answers (PRD-002 §11.1)."""
        book = self._model(state, "DesignContextManifest", contracts.DesignContextManifestV1)
        columns = [row.column_name for row in book.structural_inventory]
        rows = [row for row in frozen if row.requirement_id in asking]
        packet = askgate.build_packet(rows, state["design_revision"], round_number, columns)
        opened = self._commit(state, "UserQuestionPacket", packet.canonical_payload(),
                              self._parents(state, "DesignContextManifest"))
        self._emit(state, "user_interrupt.created", EVAL_ASK, status="clarification",
                   artifact_refs=(self._ref(state, "UserQuestionPacket"),))
        answer = parse_strict(contracts.UserContextAnswerV1, interrupt({
            "kind": contracts.InterruptKind.CLARIFICATION.value,
            "interrupt_artifact_id": opened.artifact_id, "interrupt_hash": opened.content_hash,
            "design_revision": state["design_revision"], "round_number": round_number,
            "packet": packet.canonical_payload()}))
        stored = self._commit(state, "UserContextAnswer", answer.canonical_payload(), (opened,))
        # Grouped, not keyed: `frozen` is unique by (requirement_id, scope_id), so a dict keyed by
        # requirement id alone would settle the last-sorted scope and silently drop the rest (D-100).
        grouped = askgate.group(rows, columns)
        for outcome in askgate.validate_answers(packet, answer, grouped, columns):
            for scope_id in outcome.scope_ids:
                self.requirements.set_state(
                    state["analysis_id"], state["design_revision"], outcome.requirement_id,
                    scope_id, outcome.state, stored.artifact_id)
        self._emit(state, "user_interrupt.resumed", EVAL_ASK, status="clarification")
        return self._out(state, stage="gate_again", clarification_round=round_number,
                         answer_ids=[*state["answer_ids"], stored.artifact_id])

    def method(self, state: DesignState) -> dict[str, Any]:
        """Pack eligibility, the method-design task, prerepair diagnostics, and the frame (§13)."""
        ledger = self._model(state, "RoleLedger", semantics.RoleLedgerV1)
        held = {row.role.value for row in ledger.claims
                if row.status is not agent.EpistemicStatus.UNKNOWN}
        eligible = [row.method_id for row in self.deps.packs.all()
                    if set(row.required_roles) <= held]
        if not eligible:
            return self._out(state, status="refused", refusal_code="UNSUPPORTED_METHOD",
                             candidate_method_ids=[])
        payload: dict[str, object] = {
            "eligible_methods": eligible, "parents": {
                kind: self._ref(state, kind).model_dump(mode="json") for kind in
                ("TableSelection", "MeasurementMap", "CausalContext", "RoleLedger")},
            "role_ledger": self._payload(state["artifacts"]["RoleLedger"]),
            # Without this an answer only unblocks the gate: no model ever reads what the
            # analyst said, so the design is drawn without it (D-100).
            "user_answers": [self._payload(found) for found in state["answer_ids"]],
            "method_contracts": [self.deps.packs.get(name).model_dump(mode="json")
                                 for name in eligible]}
        common = {"role_ledger": ledger, "causal_context": self._model(
            state, "CausalContext", semantics.CausalContextV1)}
        done = self._run_task(
            state, "method_design", frame.ExperimentDesignV1, scope_kind="design",
            scope_ids=("experiment_design",), parent_kinds=("RoleLedger",), payload=payload,
            ctx=self._ctx(state, method_id=eligible[0], **common))
        if done is None:
            return self._out(state, candidate_method_ids=eligible)
        design = cast(frame.ExperimentDesignV1, done[0][0])
        contract = self._run_task(
            state, "method_design", frame.RunnableFrameContractV1, scope_kind="design",
            scope_ids=("runnable_frame_contract",), parent_kinds=("ExperimentDesign",),
            commits="RunnableFrameContract",
            ctx=self._ctx(state, method_id=design.method_id, design=design, **common),
            payload=payload | {"experiment_design": self._ref(
                state, "ExperimentDesign").model_dump(mode="json")})
        if contract is None:
            return self._out(state, candidate_method_ids=eligible)
        return self._out(state, stage="method", method_id=design.method_id,
                         candidate_method_ids=eligible)

    def _row_is_unit(self, state: DesignState) -> bool:
        """True when no column identifies a unit and the intent asserts one row per unit.

        The assertion is load-bearing. "No column is unique" alone would turn a panel file that
        lost its id column into one independent unit per row, and every interval would be too
        narrow; the grain must be stated before the row number becomes an identifier (D-105).
        """
        intent = self._model(state, "DesignIntent", contracts.DesignIntentV1)
        return (intent.candidate_grain == "one_row_per_unit"
                and not self._grain(state)["unique_single_columns"])

    # PRD-003 §4 reads two facts off the committed design that no worker can know: the identity
    # of its delivery-capacity check, and the pre-repair report in its lineage. Both are bound
    # here, before the design is written, because a D-031 identity is a function of the payload
    # alone — so the check can be named before `capacity_node` commits it (D-077). The row unit
    # identifier is the same class (D-105): the column does not exist until preparation makes it,
    # and wall 2 admits only `manifest.structural_inventory` names, so no worker could name it.
    def _commit(self, state: DesignState, kind: str, payload: Mapping[str, object],
                parents: tuple[ArtifactEnvelopeV1, ...]) -> ArtifactEnvelopeV1:
        """Commit one artifact, bound with the facts no worker can know."""
        body = dict(payload)
        if kind in ROW_UNIT_KINDS and self._row_is_unit(state):
            body = _bind_row_unit(kind, body)
        if kind != "ExperimentDesign":
            return super()._commit(state, kind, body, parents)
        draft = parse_strict(frame.ExperimentDesignV1, body)
        bound = body | {"capacity_check": self._capacity_ref(state, draft)}
        return super()._commit(state, kind, bound, parents + self._prerepair(state, draft))

    def _capacity(self, design: frame.ExperimentDesignV1) -> frame.DeliveryCapacityCheckV1:
        """The exact-cardinality delivery preflight over one design; a pure function (§13.5)."""
        contrasts = max(1, len(design.primary_contrasts))
        counts = dict.fromkeys(frame.CAPACITY_DIMENSIONS, 0) | {
            "arms": contrasts + 1, "contrasts": contrasts, "series": contrasts + 1,
            "evidence_items": len(design.required_visual_evidence) + len(
                design.required_postrepair_diagnostics)}
        return check_capacity(self.deps.packs.get(design.method_id), counts,
                              design.required_visual_evidence,
                              registry=self.deps.capacity_registry)

    def _capacity_ref(self, state: DesignState,
                      design: frame.ExperimentDesignV1) -> dict[str, str]:
        """The capacity check's D-031 identity, computed from its payload before it is written."""
        digest = content_hash(self._capacity(design).canonical_payload())
        return {"artifact_id": f"{CAPACITY_KIND.lower()}:{state['analysis_id']}:{digest[:16]}",
                "content_hash": digest}

    def _prerepair(self, state: DesignState,
                   design: frame.ExperimentDesignV1) -> tuple[ArtifactEnvelopeV1, ...]:
        """The design's required read-only diagnostics over the selected CSV (PRD-002 §14)."""
        selection = self._model(state, "TableSelection", contracts.TableSelectionV1)
        ledger = self._model(state, "RoleLedger", semantics.RoleLedgerV1)
        source = CsvObjectFrameSource(self.deps.objects, selection.resource_object_locator,
                                      self._ref(state, "TableSelection"))
        columns = {row.role.value: list(row.column_refs) for row in ledger.claims}
        params = {
            "columns": columns.get("treatment", []), "by": columns.get("group", []),
            "key_columns": columns.get("unit_identifier", []),
            "target": next(iter(columns.get("outcome", [])), None),
            "column": next(iter(columns.get("treatment", [])), None),
            "running_column": next(iter(columns.get("running_variable", [])), None)}
        results = tuple(run_diagnostic(name, source, params)
                        for name in design.required_prerepair_diagnostics
                        if name in DIAGNOSTIC_SPECS)
        if not results:
            return ()
        return (self._commit(
            state, "PreRepairFeasibilityReport", frame.PreRepairFeasibilityReportV1(
                method_id=design.method_id, results=results).canonical_payload(),
            self._parents(state, "RoleLedger")),)


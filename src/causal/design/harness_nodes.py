"""Design-harness pipeline nodes: entry through method design (T-013 §1.5 split, D-055)."""
# ruff: noqa: PLR0402 -- module namespaces keep compiler/model ownership explicit.

from __future__ import annotations

from typing import Any, Final, cast

from langgraph.types import interrupt

import causal.design.askgate as askgate
import causal.design.compiler_v2 as compiler_v2
import causal.design.contracts as contracts
import causal.design.diagnostics as diagnostics
import causal.design.entry as entry
import causal.design.resolution_v2 as resolution_v2
import causal.design.semantics as semantics
import causal.design.v2 as v2
import causal.design.validators as validators
import causal.shared.agenttask as agenttask
from causal.design import compile as compiler
from causal.design.compile import ColumnTriageRecordV1, triage
from causal.design.harness_base import EVAL_ASK, EVAL_METHOD, REGISTRY_VERSIONS, DesignState
from causal.design.packs import MethodPackV1
from causal.shared import envelope as agent
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef
from causal.shared.events import Severity
from causal.shared.readers import CsvObjectFrameSource
from causal.shared.validation import make_issue, parse_strict

_FACT_REQUIREMENTS: Final = {
    "assignment_mechanism": "design.assignment_mechanism", "estimand": "design.estimand",
    "grain": "design.table_grain", "defined_comparator": "design.population_comparator", "comparator": "design.population_comparator",
    "fixed_cutoff": "design.cutoff", "cutoff": "design.cutoff", "sharp_assignment_at_cutoff": "design.sharp_assignment", "sharp_assignment": "design.sharp_assignment",
    "adoption_time_defined": "design.adoption_time", "adoption_time": "design.adoption_time",
    "treatment": "design.treatment_meaning", "outcome": "design.outcome_window", "unit_identifier": "design.unit_identity",
    "cluster": "design.unit_identity", "time": "design.table_grain", "group": "design.population_comparator", "running_variable": "design.concept_mapping"}


def _diagnostic_budget_issues(
    proposal: v2.AgentDesignProposalV2, observed_count: int,
) -> tuple[validators.ValidationIssueV1, ...]:
    remaining = max(0, diagnostics.DIAGNOSTIC_TOOL_CALL_LIMIT - observed_count)
    if len(proposal.requested_diagnostic_ids) <= remaining:
        return ()
    return (make_issue(
        diagnostics.DIAGNOSTIC_TOOL_BUDGET_EXHAUSTED,
        "/payload/requested_diagnostic_ids", "admission.method_diagnostic_budget",
        ("revise_field",), detail=(
            f"Only {remaining} diagnostic requests remain. Reduce requested_diagnostic_ids "
            "to that limit (empty when zero); preserve all assessments of observed results. "
            "The compiler will still run the selected method's required diagnostics.")),)


def _observed_comparator_issues(
    pack: MethodPackV1, comparator: str, plan: v2.DiagnosticPlanV2,
    observed: tuple[contracts.DiagnosticResultV1, ...],
) -> tuple[validators.ValidationIssueV1, ...]:
    if pack.method_id not in {"randomized_experiment", "aipw"} or not observed:
        return ()
    report = resolution_v2.evaluate_diagnostics(plan, observed)
    _, issues = compiler_v2.compile_contrasts(pack, comparator, report)
    return tuple(make_issue(
        issue.code, "/payload/comparator", "admission.observed_comparator",
        ("revise_field",), detail=(
            f"Comparator {comparator!r} does not identify one exact observed treatment level. "
            f"Observed levels: {issue.candidate_values!r}. Use the intended exact level in "
            "comparator and its matching source interpretation; preserve the method, estimand, "
            "ranking, and diagnostic assessments."))
        for issue in issues if issue.code == "comparator_level_unresolved")


class PipelineNodes(entry.EntryNodes):
    def intent(self, state: DesignState) -> dict[str, Any]:
        book = self._model(state, "DesignContextManifest", contracts.DesignContextManifestV1)
        done = self._run_task(state, "intent", contracts.DesignIntentV1, scope_kind="design", scope_ids=(book.selected_table,), parent_kinds=("DesignContextManifest",),
            payload={"question": self._payload(state["artifacts"]["QuestionRecord"]),
                     "selected_table": book.selected_table,
                     # What the harness counted, so grain and unit identity are read off the
                     # table rather than guessed from its shape (D-103).
                     "table_facts": self._grain(state), "columns": [row.column_name for row in book.structural_inventory]},
            evidence_scope_ids=tuple(row.column_name for row in book.structural_inventory))
        return self._out(state) if done is None else self._out(state, stage="intent")

    def triage_node(self, state: DesignState) -> dict[str, Any]:
        book = self._model(state, "DesignContextManifest", contracts.DesignContextManifestV1)
        hypotheses = {name: tuple(str(item["kind"]) for item in col.get("hypotheses") or ())
                      for name, col in self._profile(state)[1].items()}
        record = triage(self._model(state, "DesignIntent", contracts.DesignIntentV1), book,
                        hypotheses)
        self._commit(state, "ColumnTriageRecord", record.canonical_payload(), self._parents(
            state, "DesignIntent"))
        return self._out(state, stage="triage")

    def semantic(self, state: DesignState) -> dict[str, Any]:
        record = self._model(state, "ColumnTriageRecord", ColumnTriageRecordV1)
        cards: list[str] = []
        for batch in record.batches:
            def admit(drafts: tuple[Any, ...], result: agent.AgentTaskResultV1,
                      columns: tuple[str, ...] = batch.column_names,
                      ) -> tuple[validators.ValidationIssueV1, ...]:
                if sorted(draft.column_name for draft in drafts) == sorted(columns):
                    return ()
                return (make_issue(
                    "semantic_batch_scope_mismatch", "/payload/items", "admission.semantic_scope",
                    ("revise_field",), detail="Return exactly one card for each assigned column."),)

            done = self._run_task(state, "semantic_batch", semantics.ColumnSemanticCardV1,
                scope_kind="column",
                scope_ids=batch.column_names, parent_kinds=("DesignIntent",), many=True,
                payload={"batch_id": batch.batch_id, "table_name": record.table_name,
                         "columns": list(batch.column_names)}, precommit_admission=admit)
            if done is None:
                return self._out(state)
            cards.extend(found.artifact_id for _, found in done)
        return self._out(state, stage="semantic", card_ids=cards)

    def measurement(self, state: DesignState) -> dict[str, Any]:
        built = compiler.compile_measurement_map(
            self._model(state, "DesignIntent", contracts.DesignIntentV1),
            [parse_strict(semantics.ColumnSemanticCardV1, self._payload(found)) for found in state["card_ids"]])
        self._commit(state, "MeasurementMap", built.canonical_payload(), self._parents(
            state, "DesignIntent"))
        return self._out(state, stage="measurement")

    def roles(self, state: DesignState) -> dict[str, Any]:
        record = self._model(state, "ColumnTriageRecord", ColumnTriageRecordV1)
        mapped = self._model(state, "MeasurementMap", semantics.MeasurementMapV1)
        found_ids: list[str] = []
        for batch in record.batches:
            concepts = sorted({link.concept_id for link in mapped.links if link.column_name in batch.column_names})
            done = self._run_task(state, "role_evidence", semantics.RoleEvidenceV1,
                scope_kind="relationship",
                scope_ids=concepts or [record.table_name], parent_kinds=("MeasurementMap",),
                # PRD-002 §9.5 routes "assigned concepts/relationships, timing, cited cards" here.
                # The cards hold the meaning, units, levels and timing of these exact columns and
                # were committed one node ago; sending their names alone starved the worker (D-104).
                payload={"batch_id": batch.batch_id, "concept_ids": concepts,
                         "columns": list(batch.column_names),
                         "design_intent": self._payload(state["artifacts"]["DesignIntent"]),
                         "cards": [card for found in state["card_ids"] if
                                   (card := self._payload(found))["column_name"] in batch.column_names]},
                evidence_scope_ids=batch.column_names)
            if done is None:
                return self._out(state)
            found_ids.extend(found.artifact_id for _, found in done)
        return self._out(state, stage="roles", role_ids=found_ids)

    def synthesis(self, state: DesignState) -> dict[str, Any]:
        payload: dict[str, object] = {"design_intent": self._payload(state["artifacts"]["DesignIntent"]), "measurement_map": self._payload(state["artifacts"]["MeasurementMap"]),
            "hypotheses": [self._payload(found) for found in state["role_ids"]]}
        done = self._run_task(state, "causal_context", semantics.CausalContextV1,
            scope_kind="design",
            scope_ids=("causal_context",), parent_kinds=("MeasurementMap",), payload=payload,
            evidence_scope_ids=tuple(link.column_name for link in self._model(
                state, "MeasurementMap", semantics.MeasurementMapV1).links))
        if done is None:
            return self._out(state)
        ledger = self._run_task(state, "role_ledger", semantics.RoleLedgerV1, scope_kind="design",
            scope_ids=("role_ledger",), parent_kinds=("CausalContext",), commits="RoleLedger",
            ctx=self._ctx(state, causal_context=done[0][0]),
            payload=payload | {"causal_context": done[0][0].model_dump(mode="json")},
            evidence_scope_ids=tuple(link.column_name for link in self._model(
                state, "MeasurementMap", semantics.MeasurementMapV1).links))
        return self._out(state) if ledger is None else self._out(state, stage="synthesis")

    def requirements_gate(self, state: DesignState) -> dict[str, Any]:
        frozen = askgate.freeze_requirements(self._open_requirements(state))
        accepted: dict[tuple[str, str], askgate.AcceptedFact] = dict(
            self._accepted_facts(state))
        if "DesignFactSet" in state["artifacts"]:
            for fact in self._model(state, "DesignFactSet", v2.DesignFactSetV2).facts:
                requirement_id = _FACT_REQUIREMENTS.get(fact.fact_id)
                template = self.deps.templates.get(requirement_id or "")
                if fact.executable and template is not None and template.scope_kind.value == "design":
                    admitted = self._persist_design_fact(state, fact)
                    if admitted is not None:
                        accepted[(template.requirement_id, "design")] = admitted
        intent = self._model(state, "DesignIntent", contracts.DesignIntentV1)
        unique = set(self._grain(state).get("unique_single_columns") or ())
        if (intent.candidate_grain == "one_row_per_unit"
                and set(intent.unit.candidate_columns) & unique):
            measured = v2.DesignFactV2(
                fact_id="grain", requirement_id="design.table_grain", scope_id="design",
                value=intent.candidate_grain, source=v2.FactSource.MEASUREMENT,
                source_artifact_ids=(self._profile(state)[0],),
                evidence_class=agent.EvidenceClass.MEASURED_OBSERVATION,
                relation=v2.EvidenceRelation.DIRECT,
                epistemic_status=agent.EpistemicStatus.EVIDENCED,
                acceptance_status=v2.FactAcceptanceStatus.ACCEPTED, executable=True)
            admitted = self._persist_design_fact(state, measured)
            if admitted is not None:
                accepted[("design.table_grain", "design")] = admitted
        round_number = state["clarification_round"] + 1
        decisions = askgate.gate(frozen, accepted, round_number, self.deps.templates)
        settled = {askgate.GateRoute.RESOLVED: askgate.RequirementState.RESOLVED,
            askgate.GateRoute.RECORD_SENSITIVITY: askgate.RequirementState.UNKNOWN_ACCEPTED}
        for row in decisions:
            if row.route in settled:
                settled_fact = accepted.get((row.requirement_id, row.scope_id))
                self.requirements.set_state(state["analysis_id"], state["design_revision"],
                    row.requirement_id, row.scope_id, settled[row.route], None,
                    getattr(settled_fact, "accepted_fact_id", None))
        open_rows = self._open_requirements(state)
        state["open_requirement_ids"] = sorted({row.requirement_id for row in open_rows})
        asking = {row.requirement_id for row in decisions if row.route is askgate.GateRoute.ASK}
        if not asking:
            terminal = [row for row in decisions if row.route.value.startswith("terminal")]
            if not terminal:
                return self._out(state, stage="gate")
            refused = terminal[0].route is askgate.GateRoute.TERMINAL_REFUSED
            return self._out(state, status="unsupported" if refused else "needs_context",
                error_code="unsupported_identification" if refused else terminal[0].reason)
        return self._ask(state, frozen, asking, round_number)

    def _ask(self, state: DesignState, frozen: tuple[agent.ContextRequirementV1, ...], asking: set[str], round_number: int) -> dict[str, Any]:
        book = self._model(state, "DesignContextManifest", contracts.DesignContextManifestV1)
        columns = [row.column_name for row in book.structural_inventory]
        rows = [row for row in frozen if row.requirement_id in asking]
        packet = askgate.build_packet(rows, state["design_revision"], round_number, columns)
        opened = self._commit(state, "UserQuestionPacket", packet.canonical_payload(), self._parents(
            state, "DesignContextManifest"))
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
        outcomes = askgate.validate_answers(packet, answer, grouped, columns)
        for outcome in outcomes:
            for scope_id in outcome.scope_ids:
                accepted_fact_id = None
                if outcome.state is askgate.RequirementState.RESOLVED:
                    template = self.deps.templates[outcome.requirement_id]
                    value = askgate.accepted_answer_value(
                        template.expected_answer_schema, str(outcome.value), scope_id)
                    fact = self.accepted_facts.accept(
                        analysis_id=state["analysis_id"],
                        design_revision=state["design_revision"],
                        requirement_id=outcome.requirement_id, scope_id=scope_id,
                        value=value, value_schema=template.expected_answer_schema,
                        source_kind="user", evidence_ids=(f"ua:{stored.artifact_id}",),
                        evidence_class=agent.EvidenceClass.USER_CONFIRMATION,
                        relation="direct", origin_reference_id=stored.artifact_id,
                        origin_reference_hash=stored.content_hash,
                        created_at=self.deps.clock())
                    accepted_fact_id = fact.accepted_fact_id
                self.requirements.set_state(state["analysis_id"], state["design_revision"],
                    outcome.requirement_id,
                    scope_id, outcome.state, stored.artifact_id, accepted_fact_id)
        open_rows = self._open_requirements(state)
        open_ids = sorted({row.requirement_id for row in open_rows})
        self._emit(state, "user_interrupt.resumed", EVAL_ASK, status="clarification")
        if any(outcome.state is askgate.RequirementState.OPEN for outcome in outcomes):
            return self._out(state, status="needs_context", error_code="user_answer_unknown",
                             clarification_round=round_number,
                             open_requirement_ids=open_ids,
                             answer_ids=[*state["answer_ids"], stored.artifact_id])
        return self._out(state, stage="gate", clarification_round=round_number,
                         open_requirement_ids=open_ids,
                         answer_ids=[*state["answer_ids"], stored.artifact_id])

    def method_proposal(self, state: DesignState) -> dict[str, Any]:
        ledger = self._model(state, "RoleLedger", semantics.RoleLedgerV1)
        proposal = self._investigate(state, ledger)
        if proposal is None:
            return self._out(state)
        facts = self._compile_facts(state, proposal)
        if facts is None:
            if not self._open_requirements(state):
                self._upsert_fact_requirements(state, ["grain"])
            return self._out(state)
        self._upsert_fact_requirements(state, [name for name in ("assignment_mechanism", "estimand", "comparator") if facts.fact(name) is None])
        return self._out(state, stage="method_proposal" if not self._open_requirements(state)
                         else "gate_again")

    def _investigate(self, state: DesignState, ledger: semantics.RoleLedgerV1) -> v2.AgentDesignProposalV2 | None:
        packs = self.deps.packs.all()
        base: dict[str, object] = {
            "parents": {kind: self._ref(state, kind).model_dump(mode="json") for kind in ("TableSelection", "MeasurementMap", "CausalContext", "RoleLedger")},
            "role_ledger": self._payload(state["artifacts"]["RoleLedger"]),
            "method_contracts": [pack.model_dump(mode="json", exclude={"required_visual_evidence_ids"}) for pack in packs],
            "available_diagnostics": {p.method_id: list(p.allowed_prerepair_diagnostic_ids)
                                      for p in packs}}
        observed: tuple[contracts.DiagnosticResultV1, ...] = ()
        previous, repairs = "", 0
        correction: dict[str, str] | None = None
        for turn in range(1, diagnostics.DIAGNOSTIC_TOOL_CALL_LIMIT + 4):
            result_ids = tuple(diagnostics.diagnostic_result_id(row) for row in observed)
            context = self._ctx(
                state, role_ledger=ledger,
                causal_context=self._model(state, "CausalContext", semantics.CausalContextV1),
                diagnostic_result_ids=frozenset(result_ids))
            body = base | {
                "diagnostic_results": [
                    {"diagnostic_result_id": result_id,
                     "registered_diagnostic_id": row.diagnostic_id,
                     "observation": row.model_dump(mode="json")}
                    for result_id, row in zip(result_ids, observed, strict=True)],
                "agent_loop": {"turn": turn, "must_investigate": not observed,
                    "remaining_tool_calls": diagnostics.DIAGNOSTIC_TOOL_CALL_LIMIT - len(observed)}}
            if correction:
                body["correction"] = correction

            def admit(
                drafts: tuple[Any, ...], result: agent.AgentTaskResultV1,
                observed_results: tuple[contracts.DiagnosticResultV1, ...] = observed,
            ) -> tuple[validators.ValidationIssueV1, ...]:
                draft = cast(v2.AgentDesignProposalV2, drafts[0])
                if issues := _diagnostic_budget_issues(draft, len(observed_results)):
                    return issues
                if result.status is not agent.TaskStatus.COMPLETE:
                    return ()
                if not observed_results and not draft.requested_diagnostic_ids:
                    return (make_issue(
                        "diagnostic_required", "/payload/requested_diagnostic_ids",
                        "admission.method_diagnostic", ("revise_field",), detail=(
                            "Request at least one registered diagnostic before finalizing.")),)
                facts = self._fact_set(
                    state, draft, source_artifact_id="pending:AgentDesignProposal")
                missing = tuple(name for name in (
                    "assignment_mechanism", "estimand", "comparator", "grain")
                    if facts is None or facts.fact(name) is None)
                if missing or facts is None:
                    return (make_issue(
                        "unsupported_causal_fact", "/payload/source_interpretations",
                        "admission.causal_facts", ("revise_field", "request_context"),
                        True, missing, "Supply explicit source interpretations or request the "
                        "registered missing context; unsupported model assertions are inert."),)
                mechanism = facts.fact("assignment_mechanism")
                pack = next((p for p in packs if p.method_id in draft.ranked_method_ids
                             and mechanism in p.compatible_assignment_mechanisms), None)
                if pack is None:
                    return (make_issue(
                        "method_assignment_incompatible", "/payload/ranked_method_ids/0",
                        "admission.method_compatibility", ("revise_field",), detail=(
                            "Rank a registered method compatible with the admitted assignment "
                            "mechanism.")),)
                plan = compiler_v2.compile_diagnostic_plan(pack, facts)
                if plan.issues:
                    issue = plan.issues[0]
                    return (make_issue(
                        issue.code, issue.json_path, issue.rule_id,
                        ("revise_field", "request_context"),
                        issue.category is v2.ResolutionCategory.HUMAN_INPUT,
                        issue.required_input_ids, issue.why_blocking),)
                return _observed_comparator_issues(
                    pack, str(facts.fact("comparator")), plan, observed_results)

            done = self._run_task(
                state, "method_design", v2.AgentDesignProposalV2, scope_kind="design",
                scope_ids=("agent_design_proposal",),
                parent_kinds=(("RoleLedger", "DiagnosticObservationSet") if observed
                              else ("RoleLedger",)), payload=body, ctx=context,
                precommit_admission=admit)
            if done is None:
                return None
            proposal = cast(v2.AgentDesignProposalV2, done[0][0])
            current = content_hash(proposal.canonical_payload())
            task_id = f"dt:{state['stage_run_id']}:method_design:{content_hash(body)[:12]}"
            if previous:
                self._emit(state, "agent.design_revised", EVAL_METHOD, task_id=task_id, attempt_number=turn,
                    safe_dimensions={"prior_proposal_hash": previous,
                    "revised_proposal_hash": current, "triggering_diagnostic_ids": ",".join(
                        row.diagnostic_id for row in observed)})
            try:
                facts = self._fact_set(
                    state, proposal, source_artifact_id=done[0][1].artifact_id)
                mechanism = facts.fact("assignment_mechanism") if facts is not None else None
                pack = next((p for p in packs if p.method_id in proposal.ranked_method_ids
                             and mechanism in p.compatible_assignment_mechanisms), None)
                if facts is None or pack is None:
                    raise diagnostics.DiagnosticError("the proposal cannot bind a diagnostic plan", "diagnostic_binding_unavailable")
                plan = compiler_v2.compile_diagnostic_plan(pack, facts)
                selection = self._model(state, "TableSelection", contracts.TableSelectionV1)
                source = CsvObjectFrameSource(self.deps.objects, selection.resource_object_locator, self._ref(state, "TableSelection"))
                results = diagnostics.run_requested_diagnostics(proposal, plan, source, observed)
            except diagnostics.DiagnosticError as error:
                repairs += 1
                correction = {"code": error.code, "detail": str(error)}
                if repairs > self.deps.task_table["method_design"].correction_budget:
                    self._emit(state, "agent.escalated", EVAL_METHOD, severity=Severity.ERROR, task_id=task_id, error_code=error.code,
                        safe_dimensions={"responsible_actor": "model", "exhausted_budget": True})
                    state["status"], state["error_code"] = "system_failure", error.code
                    return None
                continue
            for result in results:
                self._emit(state, "agent.diagnostic_requested", EVAL_METHOD, task_id=task_id, attempt_number=turn,
                    safe_dimensions={"diagnostic_id": result.diagnostic_id,
                    "remaining_tool_calls": diagnostics.DIAGNOSTIC_TOOL_CALL_LIMIT - len(observed) - len(results),
                    "request_hash": content_hash({"diagnostic_id": result.diagnostic_id, "proposal_hash": current})})
            if not results:
                return proposal
            for result in results:
                self._emit(state, "diagnostic.completed", EVAL_METHOD, status=result.status.value, safe_dimensions={"diagnostic_id": result.diagnostic_id,
                    "result_hash": content_hash(result.model_dump(mode="json")),
                    "warning_count": len(result.warnings), "used_row_count": result.used_rows})
            observed, previous, correction = (*observed, *results), current, None
            observations = v2.DiagnosticObservationSetV2(
                selected_csv=self._ref(state, "TableSelection"), observations=tuple(
                    v2.DiagnosticObservationV2(
                        diagnostic_result_id=diagnostics.diagnostic_result_id(row), result=row)
                    for row in observed))
            self._commit(
                state, "DiagnosticObservationSet", observations.canonical_payload(),
                self._parents(state, "TableSelection", "AgentDesignProposal"))
        return None

    def _fact_set(
        self, state: DesignState, proposal: v2.AgentDesignProposalV2, *,
        source_artifact_id: str | None = None,
    ) -> v2.DesignFactSetV2 | None:
        intent = self._model(state, "DesignIntent", contracts.DesignIntentV1)
        profile_id, _ = self._profile(state)
        unique = set(self._grain(state).get("unique_single_columns") or ())
        measured_grain = (profile_id if intent.candidate_grain == "one_row_per_unit"
                          and set(intent.unit.candidate_columns) & unique else None)
        facts = resolution_v2.compile_proposal_facts(
            proposal, evidence=self._evidence(state),
            requirement_templates=self.deps.templates,
            accepted_facts=tuple(self._accepted_facts(state).values()),
            candidate_grain=intent.candidate_grain,
            source_interpretations=intent.source_interpretations,
            source_artifact_id=(source_artifact_id or state["artifacts"].get(
                "AgentDesignProposal", "pending:AgentDesignProposal")),
            verified_grain_source=measured_grain)
        return None if not any(row.fact_id == "grain" and row.executable for row in facts) else (
            resolution_v2.compile_fact_set(selected_csv=self._ref(state, "TableSelection"), facts=facts,
                ledger=self._model(state, "RoleLedger", semantics.RoleLedgerV1), ledger_ref=self._ref(state, "RoleLedger")))

    def _compile_facts(self, state: DesignState, proposal: v2.AgentDesignProposalV2) -> v2.DesignFactSetV2 | None:
        compiled = self._fact_set(state, proposal)
        if compiled is not None:
            current = self._accepted_facts(state)
            conflicts = [fact for fact in compiled.facts if fact.executable
                         and (held := current.get((fact.requirement_id, fact.scope_id)))
                         is not None and held.value != fact.value]
            if conflicts:
                for fact in conflicts:
                    self.accepted_facts.mark_conflicting(
                        state["analysis_id"], state["design_revision"],
                        fact.requirement_id, fact.scope_id)
                self._upsert_fact_requirements(
                    state, [fact.fact_id for fact in conflicts], "method_proposal")
                return None
            for fact in compiled.facts:
                self._persist_design_fact(state, fact)
            self._commit(state, "DesignFactSet", compiled.canonical_payload(), self._parents(state, "AgentDesignProposal", "RoleLedger"))
        return compiled

    def _persist_design_fact(
        self, state: DesignState, fact: v2.DesignFactV2,
    ) -> askgate.AcceptedFactV1 | None:
        """Admit a compiler-accepted value without disguising unsupported model text as fact."""
        template = self.deps.templates.get(fact.requirement_id)
        if (not fact.executable or fact.value is None or template is None
                or fact.evidence_class is None
                or fact.relation not in {v2.EvidenceRelation.DIRECT,
                                         v2.EvidenceRelation.CORROBORATING}):
            return None
        reference_id = fact.source_artifact_ids[0]
        origin_hash = state["hashes"].get(reference_id)
        if origin_hash is None:
            origin_hash = content_hash({
                "reference_id": reference_id,
                "evidence": self._evidence(state).get(reference_id, "")})
        return self.accepted_facts.accept(
            analysis_id=state["analysis_id"], design_revision=state["design_revision"],
            requirement_id=fact.requirement_id, scope_id=fact.scope_id,
            value=fact.value,
            value_schema=template.expected_answer_schema,
            source_kind=fact.source.value, evidence_ids=fact.supporting_evidence_ids,
            evidence_class=fact.evidence_class, relation=fact.relation.value,
            origin_reference_id=reference_id, origin_reference_hash=origin_hash,
            created_at=self.deps.clock())

    def _upsert_fact_requirements(self, state: DesignState, required: list[str],
                                  resume_node: str = "method_proposal") -> None:
        availability = agenttask.evidence_availability(self._evidence(state))
        rows = []
        for fact_id in dict.fromkeys(required):
            template = self.deps.templates.get(_FACT_REQUIREMENTS.get(fact_id, ""))
            if template is None:
                continue
            scopes: tuple[str, ...] = ("design",)
            if template.scope_kind is agent.RequirementScopeKind.CONCEPT:
                ledger = self._model(state, "RoleLedger", semantics.RoleLedgerV1)
                scopes = tuple(claim.concept_id for claim in ledger.claims
                               if claim.role.value == fact_id)
            for scope_id in scopes:
                rows.append(agent.ContextRequirementV1(
                    requirement_id=template.requirement_id,
                    registry_version="context-requirements.v1",
                    scope_kind=template.scope_kind, scope_id=scope_id,
                    fact_required=template.fact_required, why_required=template.why_required,
                    decisions_blocked=("method_eligibility",),
                    criticality=template.criticality,
                    acceptable_evidence_types=template.acceptable_evidence_types,
                    required_support=template.required_support,
                    methods_required_for=template.methods_required_for,
                    attempted_evidence=tuple(agent.AttemptedEvidenceV1(
                        evidence_id=found, availability_status=status)
                        for found, status in availability.items()),
                    user_may_know=template.user_may_know,
                    missing_action=template.missing_action,
                    expected_answer_schema=template.expected_answer_schema))
        self._upsert(rows, state["analysis_id"], state["design_revision"], state)
        state["open_requirement_ids"] = sorted({*state["open_requirement_ids"], *(row.requirement_id for row in rows)})
        if rows:
            state["stage"], state["resume_node"] = "gate_again", resume_node

    def _route_compiler(self, state: DesignState, issues: tuple[Any, ...]) -> dict[str, Any]:
        previous = tuple(str(row.get("fingerprint")) for row in state.get("compiler_issues", ()))
        decision = resolution_v2.route_issues(issues, previous_fingerprints=previous)
        stored = [issue.model_dump(mode="json") for issue in decision.issues]
        if decision.action == "ask_human":
            required = [item for issue in decision.issues for item in issue.required_input_ids]
            self._upsert_fact_requirements(state, required, "method")
            if not any(item in _FACT_REQUIREMENTS for item in required):
                return self._out(state, status="system_failure", compiler_issues=stored,
                                 error_code="missing_requirement_mapping")
            return self._out(state, stage="gate_again", error_code=decision.error_code or "",
                             compiler_issues=stored)
        if decision.action == "retry_model":
            return self._out(state, status="system_failure", error_code="agent_output_invalid",
                             compiler_issues=stored)
        return self._out(state, status=decision.terminal_status or "system_failure",
                         error_code=decision.error_code or "", compiler_issues=stored)

    def method(self, state: DesignState) -> dict[str, Any]:
        proposal = self._model(state, "AgentDesignProposal", v2.AgentDesignProposalV2)
        facts = self._compile_facts(state, proposal)
        if facts is None:
            issue = v2.ValidationIssueV2.build(
                code="grain_unresolved", category=v2.ResolutionCategory.HUMAN_INPUT,
                path="/facts/grain", rule_id="compiler.grain", actual="unsupported", required=("grain",),
                expected="an evidenced or user-confirmed table grain", actor=v2.ResponsibleActor.USER,
                why="row identity and method eligibility depend on grain", actions=("provide_context",))
            return self._route_compiler(state, (issue,))
        profile = self._payload(self._profile(state)[0])
        candidates = compiler_v2.assess_candidates(self.deps.packs, facts, profile)
        eligible = {row.method_id for row in candidates if row.eligible}
        if not eligible:
            return self._route_compiler(state, resolution_v2.route_candidates(candidates).issues)
        method_id = next(name for name in proposal.ranked_method_ids if name in eligible)
        pack = self.deps.packs.get(method_id)
        plan = compiler_v2.compile_diagnostic_plan(pack, facts)
        self._commit(state, "DiagnosticPlan", plan.canonical_payload(), self._parents(state, "DesignFactSet"))
        if plan.issues:
            return self._route_compiler(state, plan.issues)
        selection = self._model(state, "TableSelection", contracts.TableSelectionV1)
        source = CsvObjectFrameSource(self.deps.objects, selection.resource_object_locator, self._ref(state, "TableSelection"))
        try:
            results = tuple(diagnostics.run_diagnostic(item.diagnostic_id, source,
                compiler_v2.diagnostic_parameters(item)) for item in plan.items)
        except (diagnostics.DiagnosticError, TypeError, ValueError) as error:
            issue = v2.ValidationIssueV2.build(
                code=getattr(error, "code", "diagnostic_execution_invalid"), path="/diagnostics",
                category=v2.ResolutionCategory.SYSTEM_FAILURE, actions=("inspect_diagnostic",),
                rule_id="compiler.diagnostic_execution", actual=type(error).__name__,
                actor=v2.ResponsibleActor.SYSTEM,
                expected="all bound diagnostics execute deterministically",
                why="empirical eligibility cannot be established")
            return self._route_compiler(state, (issue,))
        report = resolution_v2.evaluate_diagnostics(plan, results)
        self._commit(state, "DiagnosticReport", report.canonical_payload(), self._parents(state, "DiagnosticPlan"))
        if report.issues:
            return self._route_compiler(state, report.issues)
        contrasts, contrast_issues = compiler_v2.compile_contrasts(pack, str(facts.fact("comparator")), report)
        if contrast_issues:
            return self._route_compiler(state, contrast_issues)
        intent = self._model(state, "DesignIntent", contracts.DesignIntentV1)
        rejected = {row.method_id: "; ".join(issue.code for issue in row.issues) or
                    "ranked lower by the proposal" for row in candidates if row.method_id != method_id}
        design = compiler_v2.compile_design(
            pack=pack, facts=facts, proposal=proposal,
            frame=self._model(state, "CausalContext", semantics.CausalContextV1).frame,
            causal_question=intent.causal_claim, intended_decision=intent.intended_decision,
            comparator=str(facts.fact("comparator")), unit=str(facts.fact("unit")),
            contrasts=contrasts, rejected_methods=rejected, method_structure=resolution_v2.compile_method_structure(facts, report),
            column_cards=tuple((ArtifactRef(artifact_id=found, content_hash=state["hashes"][found]),
                parse_strict(semantics.ColumnSemanticCardV1, self._payload(found)))
                for found in state["card_ids"]),
            registry_versions=dict(REGISTRY_VERSIONS) | {"visualization_catalog": "visualization-catalog.v1"})
        self._commit(state, "CompiledDesign", design.canonical_payload(), self._parents(
            state, "DesignFactSet", "DiagnosticPlan", "DiagnosticReport", "MeasurementMap",
            "CausalContext", "RoleLedger") + tuple(
                self.deps.products.load_envelope(found) for found in state["card_ids"]))
        return self._out(state, stage="method", method_id=method_id, error_code="", compiler_issues=[])

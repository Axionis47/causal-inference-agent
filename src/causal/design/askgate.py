"""Requirement freeze, the deterministic ask gate, packets, and typed answers (SC §6.2)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import StrEnum
from math import isfinite
from typing import Annotated, Any, Final, Literal, Protocol

from psycopg import Connection
from psycopg.types.json import Json
from pydantic import Field

from causal.design.contracts import (
    AnswerItemV1,
    AnswerKind,
    QuestionItemV1,
    UserContextAnswerV1,
    UserQuestionPacketV1,
    _Row,
)
from causal.design.packs import RequirementTemplateV1
from causal.shared.canonical import content_hash
from causal.shared.contracts import Identity, Sha256Hex
from causal.shared.envelope import (
    ContextRequirementV1,
    Criticality,
    EvidenceClass,
    MissingAction,
    RequirementScopeKind,
)

__all__ = [
    "CHOICE_PREFIX", "CLOSED_ANSWER_SCHEMAS", "MAX_QUESTIONS", "MAX_ROUNDS", "AcceptedFactV1",
    "AnswerOutcome", "AskGateError", "GateDecision", "GateRoute", "PsycopgAcceptedFactStore",
    "PsycopgRequirementStore", "RequirementState", "RequirementStore",
    "accepted_answer_value", "build_packet", "freeze_requirements", "gate", "group",
    "schema_declared", "validate_answers",
]

MAX_QUESTIONS: Final = 8
MAX_ROUNDS: Final = 6
# A known empty/not_offered/unreadable/withheld slot is exhausted and never re-queried; only an
# `evidenced` row can satisfy a requirement (PRD-002 §11).
EXHAUSTED_STATUSES: Final = frozenset({"empty", "not_offered", "unreadable", "withheld"})
TERMINAL_STATUSES: Final = EXHAUSTED_STATUSES | {"evidenced"}
CHOICE_PREFIX: Final = "choice:"
CLOSED_ANSWER_SCHEMAS: Final = frozenset({
    "free_text", "iso_date", "boolean", "number", "column_name", "free-text.v1", "column-list.v1",
    "duration-window.v1", "level-map.v1", "timing-class.v1", "mapping-list.v1"})
_DURATION: Final = re.compile(r"^\d+ (day|week|month|year)s?$")
_PAIRS: Final = re.compile(r"^[^=,\s]+=[^=,]+(,[^=,\s]+=[^=,]+)*$")
# A pair answer carries one value per scope, so it resolves only the scopes it names.
_PAIR_SCHEMAS: Final = frozenset({"level-map.v1", "mapping-list.v1"})
class AskGateError(ValueError):
    """An ask-gate operation failed; `code` is a stable contract value."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class RequirementState(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"
    UNKNOWN_ACCEPTED = "unknown_accepted"
    REFUSED = "refused"


class GateRoute(StrEnum):
    RESOLVED = "resolved"
    RECORD_SENSITIVITY = "record_sensitivity"
    ASK = "ask"
    TERMINAL_NEEDS_CONTEXT = "terminal_needs_context"
    TERMINAL_REFUSED = "terminal_refused"


class GateDecision(_Row):
    """One requirement's route out of the deterministic gate, with its reason code."""

    requirement_id: Identity
    scope_id: Identity
    route: GateRoute
    reason: Identity


class AnswerOutcome(_Row):
    """One requirement's state after the user's typed answer, at every scope it settled."""

    requirement_id: Identity
    scope_ids: tuple[Identity, ...]
    state: RequirementState
    value: str | None


class AcceptedFact(Protocol):
    @property
    def requirement_id(self) -> str: ...
    @property
    def scope_id(self) -> str: ...
    @property
    def value(self) -> object: ...
    @property
    def evidence_class(self) -> EvidenceClass | None: ...
    @property
    def relation(self) -> object: ...
    @property
    def acceptance_status(self) -> object: ...
    @property
    def executable(self) -> bool: ...


AcceptedValue = str | int | float | bool | tuple[str, ...]


class AcceptedFactV1(_Row):
    """One admitted factual value; availability and unreviewed model text never create it."""

    accepted_fact_id: Identity
    analysis_id: Identity
    design_revision: Annotated[int, Field(ge=1)]
    requirement_id: Identity
    scope_id: Identity
    value: AcceptedValue
    value_schema: Identity
    source_kind: Literal["user", "document", "measurement", "interpretation", "derived"]
    evidence_ids: tuple[Identity, ...]
    evidence_class: EvidenceClass
    relation: Literal["direct", "corroborating"]
    acceptance_status: Literal["accepted"] = "accepted"
    executable: Literal[True] = True
    origin_revision: Annotated[int, Field(ge=1)]
    origin_reference_id: Identity
    origin_reference_hash: Sha256Hex
    inherited_from_fact_id: Identity | None = None
    supersedes_fact_id: Identity | None = None


def accepted_answer_value(schema_id: str, value: str, scope_id: str) -> AcceptedValue:
    """Normalize one already-validated human answer into its exact scoped value."""
    if schema_id == "boolean":
        return value == "true"
    if schema_id == "number":
        number = float(value)
        if not isfinite(number):
            raise ValueError("an accepted numeric answer must be finite")
        return int(number) if number.is_integer() else number
    if schema_id == "column-list.v1":
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if schema_id in _PAIR_SCHEMAS:
        pairs = dict(part.split("=", 1) for part in value.split(","))
        return pairs[scope_id]
    return value


def freeze_requirements(collected: Iterable[ContextRequirementV1],
                        ) -> tuple[ContextRequirementV1, ...]:
    """Dedup by (requirement_id, scope_id), merging blocked decisions and attempted evidence."""
    merged: dict[tuple[str, str], ContextRequirementV1] = {}
    for item in collected:
        seen = merged.get(key := (item.requirement_id, item.scope_id))
        merged[key] = item if seen is None else seen.model_copy(update={
            "decisions_blocked": tuple(sorted({*seen.decisions_blocked, *item.decisions_blocked})),
            "attempted_evidence": tuple(sorted(
                {*seen.attempted_evidence, *item.attempted_evidence},
                key=lambda row: (row.evidence_id, row.availability_status)))})
    return tuple(merged[key] for key in sorted(merged))


def schema_declared(schema_id: str) -> bool:
    """SC §6.2 condition 4: the expected answer schema is one of the closed set."""
    return schema_id in CLOSED_ANSWER_SCHEMAS or (
        schema_id.startswith(CHOICE_PREFIX) and len(schema_id) > len(CHOICE_PREFIX))


def _valid_answer(schema_id: str, value: str, columns: Sequence[str]) -> bool:
    """Check one submitted value against the closed answer-schema vocabulary."""
    if schema_id.startswith(CHOICE_PREFIX):
        return schema_declared(schema_id) and value in schema_id[len(CHOICE_PREFIX):].split("|")
    if schema_id == "iso_date":
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
        return True
    if schema_id == "number":
        try:
            return isfinite(float(value))
        except ValueError:
            return False
    if schema_id == "column-list.v1":
        parts = [part.strip() for part in value.split(",")]
        return bool(parts) and all(part in columns for part in parts)
    return {"free_text": bool(value.strip()), "free-text.v1": bool(value.strip()),
            "boolean": value in ("true", "false"), "column_name": value in columns,
            "duration-window.v1": _DURATION.match(value) is not None,
            "level-map.v1": _PAIRS.match(value) is not None,
            "mapping-list.v1": _PAIRS.match(value) is not None,
            "timing-class.v1": value in ("pre_treatment", "concurrent", "post_treatment",
                                         "unknown")}.get(schema_id, False)


def _ask_refusal(requirement: ContextRequirementV1, statuses: tuple[tuple[str, str], ...],
                 round_number: int, max_rounds: int = MAX_ROUNDS) -> str | None:
    """The first unmet SC §6.2 ask condition, or None when a question is permitted."""
    if any(status not in TERMINAL_STATUSES for _, status in statuses):
        return "sources_unexhausted"  # condition 2
    if not requirement.user_may_know:
        return "user_cannot_know"  # condition 3
    if not schema_declared(requirement.expected_answer_schema):
        return "schema_undeclared"  # condition 4
    # Condition 5 (the answer changes a legitimate design decision without depending on observed
    # results) holds by construction: requirements are raised only by validated design-stage
    # workers, which run before any estimate exists, and `decisions_blocked` names design
    # decisions. Nothing reachable from this layer can depend on a result.
    return "round_cap_exhausted" if round_number > max_rounds else None


def _fact_satisfies(requirement: ContextRequirementV1, fact: AcceptedFact | None,
                    template: RequirementTemplateV1 | None) -> bool:
    if fact is None or not fact.executable or fact.value is None:
        return False
    if getattr(fact.acceptance_status, "value", fact.acceptance_status) != "accepted":
        return False
    admitted = set(template.acceptable_evidence_types if template
                   else requirement.acceptable_evidence_types)
    if fact.evidence_class not in admitted:
        return False
    relation = getattr(fact.relation, "value", fact.relation)
    support = template.required_support if template else requirement.required_support
    return (relation == "direct" if support.value == "direct" else
            relation in {"direct", "corroborating"}
            if support.value == "direct_or_corroborated" else
            relation not in {"conflicting", "speculative"})


def gate(requirements: Sequence[ContextRequirementV1],
         accepted_facts: Mapping[tuple[str, str], AcceptedFact],
         round_number: int, templates: Mapping[str, RequirementTemplateV1],
         max_rounds: int = MAX_ROUNDS,
         ) -> tuple[GateDecision, ...]:
    """Route every frozen requirement by SC §6.2; condition 1 is the `blocking` branch."""
    decisions: list[GateDecision] = []
    for requirement in requirements:
        statuses = tuple((row.evidence_id, row.availability_status)
                         for row in requirement.attempted_evidence)
        key = (requirement.requirement_id, requirement.scope_id)
        if _fact_satisfies(requirement, accepted_facts.get(key),
                           templates.get(requirement.requirement_id)):
            route, reason = GateRoute.RESOLVED, "accepted_fact_satisfies_requirement"
        elif requirement.criticality is not Criticality.BLOCKING:
            route, reason = GateRoute.RECORD_SENSITIVITY, "supporting_criticality"
        elif requirement.missing_action is MissingAction.RETAIN_AS_SENSITIVITY:
            route, reason = GateRoute.RECORD_SENSITIVITY, "retain_as_sensitivity"
        elif requirement.missing_action is MissingAction.REFUSE:
            route, reason = GateRoute.TERMINAL_REFUSED, "refuse_by_registered_action"
        elif (refusal := _ask_refusal(requirement, statuses, round_number, max_rounds)) is None:
            route, reason = GateRoute.ASK, "ask_permitted"
        else:
            route, reason = GateRoute.TERMINAL_NEEDS_CONTEXT, refusal
        decisions.append(GateDecision(requirement_id=requirement.requirement_id,
                                      scope_id=requirement.scope_id, route=route, reason=reason))
    return tuple(decisions)


def _priority(req: ContextRequirementV1, columns: Sequence[str]) -> tuple[int, int, str]:
    """Blocking design scope first, then column scope in manifest order, then the rest."""
    if req.criticality is Criticality.BLOCKING and req.scope_kind is RequirementScopeKind.DESIGN:
        return (0, 0, req.requirement_id)
    if req.scope_kind is RequirementScopeKind.COLUMN:
        ordinal = columns.index(req.scope_id) if req.scope_id in columns else len(columns)
        return (1, ordinal, req.requirement_id)
    return (2, 0, req.requirement_id)


def group(asks: Sequence[ContextRequirementV1], column_order: Sequence[str] = (),
          ) -> dict[str, list[ContextRequirementV1]]:
    """One entry per requirement id, holding every scope it is open at, in ask priority order."""
    grouped: dict[str, list[ContextRequirementV1]] = {}
    for req in sorted(asks, key=lambda item: _priority(item, column_order)):
        grouped.setdefault(req.requirement_id, []).append(req)
    return grouped


def build_packet(asks: Sequence[ContextRequirementV1], design_revision: int, round_number: int,
                 column_order: Sequence[str] = (), max_questions: int = MAX_QUESTIONS,
                 max_rounds: int = MAX_ROUNDS) -> UserQuestionPacketV1:
    """A bounded consolidated question packet; overflow stays open for the next round.

    A question names every scope its requirement is open at, because the packet schema has no
    scope field and an unnamed column is a question no analyst can answer (D-100).
    """
    if not 1 <= round_number <= max_rounds:
        raise AskGateError(f"round {round_number} exceeds the configured bound", "round_cap")
    grouped = group(asks, column_order)
    if not grouped:
        raise AskGateError("no requirement routed to ask", "empty_packet")
    return UserQuestionPacketV1(
        packet_id=f"qp:{design_revision}:{round_number}", design_revision=design_revision,
        round_number=round_number,
        questions=tuple(QuestionItemV1(
            question_id=f"q:{rows[0].requirement_id}", requirement_ids=(rows[0].requirement_id,),
            question_text=f"{rows[0].fact_required} \u2014 for: "
                          f"{', '.join(row.scope_id for row in rows)}",
            why_it_matters=rows[0].why_required, blocked_decisions=rows[0].decisions_blocked,
            expected_answer_schema=rows[0].expected_answer_schema)
            for rows in list(grouped.values())[:max_questions]))


# An `ask_user` unknown stays open: the coordinator terminates it after round two (SC §6.2).
_UNKNOWN_STATES: Final[dict[MissingAction, RequirementState]] = {
    MissingAction.ASK_USER: RequirementState.OPEN,
    MissingAction.RETAIN_AS_SENSITIVITY: RequirementState.UNKNOWN_ACCEPTED,
    MissingAction.REFUSE: RequirementState.REFUSED}


def _outcome(question: QuestionItemV1, item: AnswerItemV1,
             rows: Sequence[ContextRequirementV1], columns: Sequence[str]) -> AnswerOutcome:
    """Settle every scope the answer covers, or route an `unknown` by the missing action.

    A pair answer carries one value per scope, so it settles only the scopes it names and the
    rest stay open for the next round; any other schema settles them all (D-100).
    """
    requirement_id = question.requirement_ids[0]
    if not rows:
        raise AskGateError(f"no requirement {requirement_id!r}", "unknown_requirement")
    scopes = tuple(row.scope_id for row in rows)
    if item.answer_kind is AnswerKind.UNKNOWN:
        return AnswerOutcome(requirement_id=requirement_id, scope_ids=scopes, value=None,
                             state=_UNKNOWN_STATES[rows[0].missing_action])
    schema = question.expected_answer_schema
    if not _valid_answer(schema, item.value or "", columns):
        raise AskGateError(f"{item.value!r} fails {schema!r}", "schema_invalid")
    if schema in _PAIR_SCHEMAS:
        named = {pair.split("=", 1)[0] for pair in (item.value or "").split(",")}
        scopes = tuple(scope for scope in scopes if scope in named)
    return AnswerOutcome(requirement_id=requirement_id, state=RequirementState.RESOLVED,
                         scope_ids=scopes, value=item.value)


def validate_answers(packet: UserQuestionPacketV1, answer: UserContextAnswerV1,
                     requirements: Mapping[str, Sequence[ContextRequirementV1]],
                     column_order: Sequence[str] = ()) -> tuple[AnswerOutcome, ...]:
    """Every packet question answered exactly once and typed; `unknown` is always allowed."""
    if answer.packet_id != packet.packet_id:
        raise AskGateError(f"answer targets packet {answer.packet_id!r}", "packet_mismatch")
    given = {item.question_id: item for item in answer.answers}
    if len(given) != len(answer.answers):
        raise AskGateError("a question is answered more than once", "duplicate_answer")
    asked = {question.question_id: question for question in packet.questions}
    if extra := sorted(set(given) - set(asked)):
        raise AskGateError(f"answers outside the packet: {extra}", "unknown_question")
    if absent := sorted(set(asked) - set(given)):
        raise AskGateError(f"unanswered packet questions: {absent}", "missing_answer")
    return tuple(
        _outcome(question, given[question_id],
                 requirements.get(question.requirement_ids[0], ()), column_order)
        for question_id, question in asked.items())


class RequirementStore(Protocol):
    """Narrow port over design.context_requirements; T-013 owns transactions and events."""

    def upsert(self, requirements: Sequence[ContextRequirementV1], analysis_id: str,
               design_revision: int) -> None: ...

    def set_state(self, analysis_id: str, design_revision: int, requirement_id: str,
                  scope_id: str, state: RequirementState,
                  answer_artifact_id: str | None,
                  accepted_fact_id: str | None = None) -> None: ...


_CURRENT_FACTS: Final = (
    "SELECT accepted_fact_id, analysis_id, design_revision, requirement_id, scope_id, value,"
    " value_schema, source_kind, evidence_ids, evidence_class, support_relation,"
    " origin_revision, origin_reference_id, origin_reference_hash, inherited_from_fact_id,"
    " supersedes_fact_id FROM design.accepted_facts WHERE analysis_id = %s"
    " AND design_revision = %s AND is_current ORDER BY requirement_id, scope_id")
_INSERT_FACT: Final = (
    "INSERT INTO design.accepted_facts (accepted_fact_id, analysis_id, design_revision,"
    " requirement_id, scope_id, value, value_schema, source_kind, evidence_ids, evidence_class,"
    " support_relation, acceptance_status, is_current, origin_revision, origin_reference_id,"
    " origin_reference_hash, inherited_from_fact_id, supersedes_fact_id, created_at) VALUES"
    " (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'accepted', true, %s, %s, %s, %s, %s, %s)"
    " ON CONFLICT (accepted_fact_id) DO NOTHING")


class PsycopgAcceptedFactStore:
    """Append-only fact admission with one immutable current value per revision and scope."""

    def __init__(self, conn: Connection[Any]) -> None:
        self._conn = conn

    @staticmethod
    def _model(row: Sequence[Any]) -> AcceptedFactV1:
        value = tuple(row[5]) if isinstance(row[5], list) else row[5]
        return AcceptedFactV1.model_validate({
            "accepted_fact_id": str(row[0]), "analysis_id": str(row[1]),
            "design_revision": int(row[2]), "requirement_id": str(row[3]),
            "scope_id": str(row[4]), "value": value, "value_schema": str(row[6]),
            "source_kind": str(row[7]), "evidence_ids": tuple(row[8]),
            "evidence_class": EvidenceClass(str(row[9])), "relation": str(row[10]),
            "origin_revision": int(row[11]), "origin_reference_id": str(row[12]),
            "origin_reference_hash": str(row[13]),
            "inherited_from_fact_id": None if row[14] is None else str(row[14]),
            "supersedes_fact_id": None if row[15] is None else str(row[15])})

    def current(self, analysis_id: str, design_revision: int,
                ) -> dict[tuple[str, str], AcceptedFactV1]:
        rows = self._conn.execute(_CURRENT_FACTS, (analysis_id, design_revision)).fetchall()
        return {(fact.requirement_id, fact.scope_id): fact
                for row in rows if (fact := self._model(row))}

    def accept(
        self, *, analysis_id: str, design_revision: int, requirement_id: str, scope_id: str,
        value: AcceptedValue, value_schema: str, source_kind: str,
        evidence_ids: Sequence[str], evidence_class: EvidenceClass, relation: str,
        origin_reference_id: str, origin_reference_hash: str, created_at: datetime,
        origin_revision: int | None = None, inherited_from_fact_id: str | None = None,
        supersedes_fact_id: str | None = None,
    ) -> AcceptedFactV1:
        existing = self.current(analysis_id, design_revision).get((requirement_id, scope_id))
        if existing is not None:
            if existing.value == value:
                return existing
            raise AskGateError(
                f"accepted fact {requirement_id!r} at {scope_id!r} has a different value",
                "accepted_fact_conflict")
        if supersedes_fact_id is None:
            prior = self._conn.execute(
                "SELECT accepted_fact_id FROM design.accepted_facts WHERE analysis_id = %s"
                " AND design_revision = %s AND requirement_id = %s AND scope_id = %s"
                " AND acceptance_status = 'conflicting' ORDER BY created_at DESC LIMIT 1",
                (analysis_id, design_revision, requirement_id, scope_id)).fetchone()
            supersedes_fact_id = None if prior is None else str(prior[0])
        body = {
            "analysis_id": analysis_id, "design_revision": design_revision,
            "requirement_id": requirement_id, "scope_id": scope_id, "value": value,
            "value_schema": value_schema, "source_kind": source_kind,
            "evidence_ids": tuple(evidence_ids), "evidence_class": evidence_class.value,
            "relation": relation, "origin_revision": origin_revision or design_revision,
            "origin_reference_id": origin_reference_id,
            "origin_reference_hash": origin_reference_hash,
            "inherited_from_fact_id": inherited_from_fact_id,
            "supersedes_fact_id": supersedes_fact_id}
        fact = AcceptedFactV1.model_validate(body | {
            "accepted_fact_id": f"af:{content_hash(body)[:28]}",
            "evidence_class": evidence_class})
        self._conn.execute(_INSERT_FACT, (
            fact.accepted_fact_id, fact.analysis_id, fact.design_revision,
            fact.requirement_id, fact.scope_id, Json(fact.value), fact.value_schema,
            fact.source_kind, Json(list(fact.evidence_ids)), fact.evidence_class.value,
            fact.relation, fact.origin_revision, fact.origin_reference_id,
            fact.origin_reference_hash, fact.inherited_from_fact_id,
            fact.supersedes_fact_id, created_at))
        return fact

    def mark_conflicting(self, analysis_id: str, design_revision: int,
                         requirement_id: str, scope_id: str) -> str | None:
        """Retain a prior value as history but remove its authority until a person resolves it."""
        row = self._conn.execute(
            "UPDATE design.accepted_facts SET acceptance_status = 'conflicting',"
            " is_current = false WHERE analysis_id = %s AND design_revision = %s"
            " AND requirement_id = %s AND scope_id = %s AND is_current"
            " RETURNING accepted_fact_id",
            (analysis_id, design_revision, requirement_id, scope_id)).fetchone()
        return None if row is None else str(row[0])

    def inherit(self, analysis_id: str, from_revision: int, to_revision: int,
                created_at: datetime) -> tuple[AcceptedFactV1, ...]:
        inherited = []
        for fact in self.current(analysis_id, from_revision).values():
            inherited.append(self.accept(
                analysis_id=analysis_id, design_revision=to_revision,
                requirement_id=fact.requirement_id, scope_id=fact.scope_id, value=fact.value,
                value_schema=fact.value_schema, source_kind=fact.source_kind,
                evidence_ids=fact.evidence_ids, evidence_class=fact.evidence_class,
                relation=fact.relation, origin_reference_id=fact.origin_reference_id,
                origin_reference_hash=fact.origin_reference_hash, created_at=created_at,
                origin_revision=fact.origin_revision,
                inherited_from_fact_id=fact.accepted_fact_id))
        return tuple(inherited)


# `state` is deliberately absent from the conflict update: a re-freeze never reopens a settled row.
_UPSERT: Final = (
    "INSERT INTO design.context_requirements (analysis_id, design_revision, requirement_id,"
    " scope_kind, scope_id, criticality, missing_action, state, attempted_evidence) VALUES"
    " (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (analysis_id, design_revision,"
    " requirement_id, scope_id) DO UPDATE SET scope_kind = EXCLUDED.scope_kind, criticality ="
    " EXCLUDED.criticality, missing_action = EXCLUDED.missing_action, attempted_evidence ="
    " EXCLUDED.attempted_evidence")
_SET_STATE: Final = (
    "UPDATE design.context_requirements SET state = %s, resolving_answer_artifact_id = %s,"
    " resolving_fact_id = %s"
    " WHERE analysis_id = %s AND design_revision = %s AND requirement_id = %s"
    " AND scope_id = %s")


class PsycopgRequirementStore:
    """`RequirementStore` over Postgres; rows carry state, never the requirement payload."""

    def __init__(self, conn: Connection[Any]) -> None:
        self._conn = conn

    def upsert(self, requirements: Sequence[ContextRequirementV1], analysis_id: str,
               design_revision: int) -> None:
        """Insert or refresh one row per frozen requirement, preserving any settled state."""
        for req in requirements:
            self._conn.execute(_UPSERT, (
                analysis_id, design_revision, req.requirement_id, req.scope_kind.value,
                req.scope_id, req.criticality.value, req.missing_action.value,
                RequirementState.OPEN.value,
                Json([row.model_dump(mode="json") for row in req.attempted_evidence])))

    def set_state(self, analysis_id: str, design_revision: int, requirement_id: str,
                  scope_id: str, state: RequirementState,
                  answer_artifact_id: str | None,
                  accepted_fact_id: str | None = None) -> None:
        """Move one requirement to a settled state, naming the answer artifact when there is one."""
        self._conn.execute(_SET_STATE, (state.value, answer_artifact_id, accepted_fact_id, analysis_id,
                                        design_revision, requirement_id, scope_id))

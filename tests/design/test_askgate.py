"""Ask-gate freeze, routing, packet bounds, and answer validation (T-012 §3, §6; SC §6.2)."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

from causal.design.askgate import (
    AnswerOutcome,
    AskGateError,
    GateRoute,
    PsycopgRequirementStore,
    RequirementState,
    build_packet,
    freeze_requirements,
    gate,
    validate_answers,
)
from causal.design.contracts import AnswerItemV1, AnswerKind, UserContextAnswerV1
from causal.design.packs import RequirementTemplateV1
from causal.shared.envelope import (
    AttemptedEvidenceV1,
    ContextRequirementV1,
    Criticality,
    EvidenceClass,
    MissingAction,
    RequirementScopeKind,
    SupportRequirement,
)
from tests.conftest import requires_docker

COLUMNS = ("re74", "re75", "treat")
DOC = (("ev:doc/readme.md", "empty"),)


def requirement(
    requirement_id: str = "design.causal_question",
    *,
    scope_kind: RequirementScopeKind = RequirementScopeKind.DESIGN,
    scope_id: str = "design",
    criticality: Criticality = Criticality.BLOCKING,
    missing_action: MissingAction = MissingAction.ASK_USER,
    attempted: tuple[tuple[str, str], ...] = DOC,
    user_may_know: bool = True,
    answer_schema: str = "free_text",
    decisions: tuple[str, ...] = ("choose_treatment",),
    acceptable: tuple[EvidenceClass, ...] = (EvidenceClass.SOURCE_STATEMENT,),
) -> ContextRequirementV1:
    """One requirement with ask-permitting defaults; every gate condition is overridable."""
    return ContextRequirementV1(
        requirement_id=requirement_id, registry_version="context-requirements.v1",
        scope_kind=scope_kind, scope_id=scope_id, fact_required="what defines the treatment",
        why_required="fixes the causal frame", decisions_blocked=decisions,
        criticality=criticality, acceptable_evidence_types=acceptable,
        required_support=SupportRequirement.DIRECT, methods_required_for=("aipw",),
        attempted_evidence=tuple(AttemptedEvidenceV1(evidence_id=identity, availability_status=s)
                                 for identity, s in attempted),
        user_may_know=user_may_know, expected_answer_schema=answer_schema,
        missing_action=missing_action)


def template(requirement_id: str, acceptable: tuple[EvidenceClass, ...]) -> RequirementTemplateV1:
    """The registry row the gate consults for acceptable evidence classes."""
    return RequirementTemplateV1(
        requirement_id=requirement_id, scope_kind=RequirementScopeKind.DESIGN,
        fact_required="f", why_required="w", criticality=Criticality.BLOCKING,
        acceptable_evidence_types=acceptable, required_support=SupportRequirement.DIRECT,
        methods_required_for=("aipw",), missing_action=MissingAction.ASK_USER,
        expected_answer_schema="free_text", user_may_know=True)


def route(
    req: ContextRequirementV1,
    index: dict[str, str] | None = None,
    round_number: int = 1,
    templates: dict[str, RequirementTemplateV1] | None = None,
) -> tuple[GateRoute, str]:
    """The single requirement's route and reason code."""
    decision = gate((req,), index or {}, round_number, templates or {})[0]
    return decision.route, decision.reason


def packet_for(*reqs: ContextRequirementV1, round_number: int = 1) -> Any:
    """A packet over the given asks, in manifest column order."""
    return build_packet(reqs, 1, round_number, COLUMNS)


def test_freeze_merges_decisions_and_evidence_by_requirement_and_scope() -> None:
    first = requirement(decisions=("b", "a"), attempted=(("ev:doc/a", "empty"),))
    second = requirement(decisions=("c", "a"), attempted=(("ev:doc/b", "withheld"),
                                                          ("ev:doc/a", "empty")))
    other_scope = requirement(scope_kind=RequirementScopeKind.COLUMN, scope_id="re74")
    frozen = freeze_requirements([second, other_scope, first])
    assert [(r.requirement_id, r.scope_id) for r in frozen] == [
        ("design.causal_question", "design"), ("design.causal_question", "re74")]
    merged = frozen[0]
    assert merged.decisions_blocked == ("a", "b", "c")
    assert [(row.evidence_id, row.availability_status) for row in merged.attempted_evidence] == [
        ("ev:doc/a", "empty"), ("ev:doc/b", "withheld")]


def test_gate_resolves_when_an_evidenced_row_matches_the_template_class() -> None:
    req = requirement(attempted=(("ev:doc/protocol.md", "evidenced"),))
    templates = {req.requirement_id: template(req.requirement_id,
                                              (EvidenceClass.SOURCE_STATEMENT,))}
    assert route(req, templates=templates) == (GateRoute.RESOLVED, "evidence_satisfies_template")


def test_gate_asks_when_the_evidenced_class_is_outside_the_template() -> None:
    req = requirement(attempted=(("ev:doc/protocol.md", "evidenced"),))
    templates = {req.requirement_id: template(req.requirement_id,
                                              (EvidenceClass.USER_CONFIRMATION,))}
    assert route(req, templates=templates) == (GateRoute.ASK, "ask_permitted")


def test_gate_asks_when_all_five_conditions_hold() -> None:
    assert route(requirement()) == (GateRoute.ASK, "ask_permitted")


def test_gate_applies_the_fan_in_availability_index_over_the_recorded_status() -> None:
    req = requirement(attempted=(("ev:kaggle/column/nsw.csv/treat/description", "evidenced"),))
    index = {"ev:kaggle/column/nsw.csv/treat/description": "fetch_failed"}
    assert route(req, index) == (GateRoute.TERMINAL_NEEDS_CONTEXT, "sources_unexhausted")


@pytest.mark.parametrize(("req", "expected"), [
    (requirement(criticality=Criticality.SUPPORTING), "supporting_criticality"),
    (requirement(attempted=()), "sources_unexhausted"),
    (requirement(attempted=(("ev:doc/a", "fetch_failed"),)), "sources_unexhausted"),
    (requirement(user_may_know=False), "user_cannot_know"),
    (requirement(answer_schema="mystery-schema.v9"), "schema_undeclared"),
])
def test_gate_refuses_the_question_when_a_condition_is_violated(
    req: ContextRequirementV1, expected: str
) -> None:
    # Condition 5 has no negative case: it holds by construction (see askgate._ask_refusal).
    assert route(req)[1] == expected


def test_gate_records_sensitivity_for_a_blocking_retain_requirement() -> None:
    req = requirement(missing_action=MissingAction.RETAIN_AS_SENSITIVITY)
    assert route(req) == (GateRoute.RECORD_SENSITIVITY, "retain_as_sensitivity")


def test_gate_terminates_a_refuse_requirement_before_asking() -> None:
    req = requirement(missing_action=MissingAction.REFUSE)
    assert route(req) == (GateRoute.TERMINAL_REFUSED, "refuse_by_registered_action")


def test_gate_stops_asking_after_the_second_round() -> None:
    assert route(requirement(), round_number=2)[0] is GateRoute.ASK
    assert route(requirement(), round_number=3) == (
        GateRoute.TERMINAL_NEEDS_CONTEXT, "round_cap_exhausted")


def test_packet_caps_at_five_and_orders_design_then_manifest_columns() -> None:
    asks = [
        requirement("z.design"), requirement("a.design"),
        requirement("col.treat", scope_kind=RequirementScopeKind.COLUMN, scope_id="treat"),
        requirement("col.re74", scope_kind=RequirementScopeKind.COLUMN, scope_id="re74"),
        requirement("col.re75", scope_kind=RequirementScopeKind.COLUMN, scope_id="re75"),
        requirement("ds.grain", scope_kind=RequirementScopeKind.DATASET, scope_id="nsw"),
        requirement("ds.other", scope_kind=RequirementScopeKind.DATASET, scope_id="nsw"),
    ]
    questions = packet_for(*asks).questions
    assert [q.question_id for q in questions] == [
        "q:a.design", "q:z.design", "q:col.re74", "q:col.re75", "q:col.treat"]
    assert questions[0].requirement_ids == ("a.design",)
    assert questions[0].allow_unknown is True


def test_packet_identity_carries_the_revision_and_round() -> None:
    second = build_packet((requirement(),), 3, 2, COLUMNS)
    assert (second.packet_id, second.round_number) == ("qp:3:2", 2)


@pytest.mark.parametrize("round_number", [0, 3])
def test_packet_never_builds_outside_the_two_round_bound(round_number: int) -> None:
    with pytest.raises(AskGateError) as raised:
        packet_for(requirement(), round_number=round_number)
    assert raised.value.code == "round_cap"


def test_packet_rejects_an_empty_ask_set() -> None:
    with pytest.raises(AskGateError) as raised:
        packet_for()
    assert raised.value.code == "empty_packet"


@pytest.mark.parametrize(("answer_schema", "value", "valid"), [
    ("free_text", "randomised in 1976", True), ("free_text", "   ", False),
    ("boolean", "true", True), ("boolean", "yes", False),
    ("iso_date", "1976-04-01", True), ("iso_date", "01/04/1976", False),
    ("column_name", "re74", True), ("column_name", "earnings", False),
    ("choice:itt|att", "att", True), ("choice:itt|att", "late", False),
    ("mapping-list.v1", "anything", False),
])
def test_every_answer_schema_branch(answer_schema: str, value: str, valid: bool) -> None:
    req = requirement(answer_schema=answer_schema)
    packet = packet_for(req)
    submitted = UserContextAnswerV1(packet_id=packet.packet_id, answers=(AnswerItemV1(
        question_id="q:design.causal_question", answer_kind=AnswerKind.VALUE, value=value),))
    requirements = {req.requirement_id: [req]}
    if valid:
        assert validate_answers(packet, submitted, requirements, COLUMNS) == (AnswerOutcome(
            requirement_id=req.requirement_id, scope_ids=(req.scope_id,),
            state=RequirementState.RESOLVED, value=value),)
        return
    with pytest.raises(AskGateError) as raised:
        validate_answers(packet, submitted, requirements, COLUMNS)
    assert raised.value.code == "schema_invalid"


@pytest.mark.parametrize(("missing_action", "state"), [
    (MissingAction.ASK_USER, RequirementState.OPEN),
    (MissingAction.RETAIN_AS_SENSITIVITY, RequirementState.UNKNOWN_ACCEPTED),
    (MissingAction.REFUSE, RequirementState.REFUSED),
])
def test_unknown_answers_route_by_registered_missing_action(
    missing_action: MissingAction, state: RequirementState
) -> None:
    req = requirement(missing_action=missing_action)
    packet = packet_for(req)
    submitted = UserContextAnswerV1(packet_id=packet.packet_id, answers=(AnswerItemV1(
        question_id="q:design.causal_question", answer_kind=AnswerKind.UNKNOWN, value=None),))
    outcome = validate_answers(packet, submitted, {req.requirement_id: [req]}, COLUMNS)[0]
    assert (outcome.state, outcome.value) == (state, None)


def test_answer_set_must_cover_the_packet_exactly() -> None:
    req, other = requirement(), requirement("col.re74")
    packet = packet_for(req, other)
    known = AnswerItemV1(question_id="q:design.causal_question", answer_kind=AnswerKind.UNKNOWN,
                         value=None)
    requirements = {req.requirement_id: [req], other.requirement_id: [other]}
    cases = {
        "missing_answer": (known,),
        "duplicate_answer": (known, known),
        "unknown_question": (known, AnswerItemV1(question_id="q:ghost",
                                                 answer_kind=AnswerKind.UNKNOWN, value=None)),
    }
    for code, answers in cases.items():
        submitted = UserContextAnswerV1(packet_id=packet.packet_id, answers=answers)
        with pytest.raises(AskGateError) as raised:
            validate_answers(packet, submitted, requirements, COLUMNS)
        assert raised.value.code == code


def test_answer_must_target_the_open_packet() -> None:
    packet = packet_for(requirement())
    submitted = UserContextAnswerV1(packet_id="qp:1:2", answers=(AnswerItemV1(
        question_id="q:design.causal_question", answer_kind=AnswerKind.UNKNOWN, value=None),))
    with pytest.raises(AskGateError) as raised:
        validate_answers(packet, submitted, {}, COLUMNS)
    assert raised.value.code == "packet_mismatch"


@requires_docker
def test_requirement_store_round_trip(conn: psycopg.Connection[Any]) -> None:
    store = PsycopgRequirementStore(conn)
    rows = (requirement(attempted=(("ev:doc/a", "empty"),)),
            requirement("col.units", scope_kind=RequirementScopeKind.COLUMN, scope_id="re74",
                        missing_action=MissingAction.REFUSE, attempted=()))
    store.upsert(rows, "an-1", 1)
    store.upsert(rows, "an-1", 1)  # re-freezing the same round is idempotent
    stored = conn.execute(
        "SELECT requirement_id, scope_id, missing_action, state, attempted_evidence"
        " FROM design.context_requirements WHERE analysis_id = 'an-1' ORDER BY requirement_id"
    ).fetchall()
    assert stored == [
        ("col.units", "re74", "refuse", "open", []),
        ("design.causal_question", "design", "ask_user", "open",
         [{"evidence_id": "ev:doc/a", "availability_status": "empty"}]),
    ]
    store.set_state("an-1", 1, "design.causal_question", "design", RequirementState.RESOLVED, None)
    store.upsert(rows, "an-1", 1)  # a later freeze never reopens a settled row
    assert conn.execute(
        "SELECT state, resolving_answer_artifact_id FROM design.context_requirements"
        " WHERE requirement_id = 'design.causal_question'").fetchone() == ("resolved", None)


def column_asks(requirement_id: str, schema: str) -> tuple[ContextRequirementV1, ...]:
    """The same requirement open at three columns, as a live run raises it (D-100)."""
    return tuple(requirement(requirement_id, scope_kind=RequirementScopeKind.COLUMN,
                             scope_id=column, answer_schema=schema) for column in COLUMNS)


def test_one_question_names_every_scope_its_requirement_is_open_at() -> None:
    """The packet schema has no scope field, so an unnamed column is unanswerable (D-100)."""
    questions = packet_for(*column_asks("column.meaning", "free-text.v1")).questions
    assert len(questions) == 1
    assert questions[0].question_text.endswith("re74, re75, treat")


def test_one_answer_settles_every_scope_it_covers() -> None:
    asks = column_asks("column.meaning", "free-text.v1")
    packet = packet_for(*asks)
    submitted = UserContextAnswerV1(packet_id=packet.packet_id, answers=(AnswerItemV1(
        question_id="q:column.meaning", answer_kind=AnswerKind.VALUE,
        value="all three are real earnings in US dollars"),))
    outcome = validate_answers(packet, submitted, {"column.meaning": list(asks)}, COLUMNS)[0]
    assert outcome.state is RequirementState.RESOLVED
    assert outcome.scope_ids == ("re74", "re75", "treat")


def test_a_pair_answer_settles_only_the_scopes_it_names() -> None:
    """A timing differs per column, so the unnamed ones stay open for the next round (D-100)."""
    asks = column_asks("column.measurement_timing", "mapping-list.v1")
    packet = packet_for(*asks)
    submitted = UserContextAnswerV1(packet_id=packet.packet_id, answers=(AnswerItemV1(
        question_id="q:column.measurement_timing", answer_kind=AnswerKind.VALUE,
        value="re74=pre_treatment,treat=concurrent"),))
    outcome = validate_answers(
        packet, submitted, {"column.measurement_timing": list(asks)}, COLUMNS)[0]
    assert outcome.scope_ids == ("re74", "treat")

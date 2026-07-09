"""PostgreSQL claims, ownership fences, resource accounting and atomic publication."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from historical_analysis import tasks
from sqlalchemy import delete, select, text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session
from test_chat import create

from backend.analysis.contracts import CheckpointReceipt
from backend.context.service import fingerprint
from backend.db import engine
from backend.models import (
    AnalysisEvent,
    AnalysisRun,
    AnalysisTask,
    AnalysisTaskAttempt,
    Conversation,
    Dataset,
    DatasetVersion,
    now,
)

PASS = [{"rule": "coverage", "kind": "deterministic", "passed": True, "detail": "All owned columns covered."}]
FAIL = [{**PASS[0], "passed": False}]


@pytest.fixture
def run_id(owner):
    _, _, chat = create(owner)
    with Session(engine()) as db:
        conversation = db.get(Conversation, UUID(chat["id"]))
        version = db.get(DatasetVersion, conversation.version_id)
        dataset = db.get(Dataset, version.dataset_id)
        run = AnalysisRun(
            id=uuid4(),
            conversation_id=conversation.id,
            version_id=version.id,
            author_id=dataset.owner_id,
            submission_id=uuid4(),
            question="Effect of group on score",
            trace_id=uuid4(),
            inputs={"contract": "checkpoint-harness-8", "column_count": 2},
        )
        db.add(run)
        db.commit()
        return run.id


def make(run_id, key="screen:one", column="c1", **kwargs):
    return tasks.ensure(
        run_id,
        key=key,
        checkpoint="target",
        kind="screen",
        column_ids=[column],
        scope={"column": column},
        reason="Screen the supplied column meaning.",
        dependency_receipt_ids=kwargs.pop("dependency_receipt_ids", []),
        packet={"question": "Effect of group on score", "task_columns": [column]},
        **kwargs,
    )


def test_task_identity_is_immutable_and_claim_is_atomic(run_id):
    first = make(run_id)
    assert make(run_id) == first
    with pytest.raises(tasks.TaskConflict, match="immutable"):
        make(run_id, column="different")
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: tasks.claim(UUID(first["id"])), range(2)))
    assert sum(item is not None for item in claims) == 1
    claim = next(item for item in claims if item)
    assert tasks.budgets(run_id)["model_calls"] == 1
    receipt = tasks.finish(claim, {"summary": "Covered"}, PASS)
    CheckpointReceipt.model_validate(receipt)
    assert tasks.finish(claim, {"summary": "Covered"}, PASS) == receipt
    assert tasks.claim(claim.task_id) is None
    assert tasks.coverage(run_id, first["obligation_keys"])["complete"]
    with Session(engine()) as db:
        assert tasks.input_for_receipt(db, run_id, receipt["id"]) == first["packet"]
        sequences = list(
            db.scalars(
                select(AnalysisEvent.sequence)
                .where(AnalysisEvent.run_id == run_id)
                .order_by(AnalysisEvent.sequence)
            )
        )
    assert sequences == list(range(1, len(sequences) + 1))


def test_expired_worker_is_fenced_and_uncertain_attempt_stays_charged(run_id):
    task = make(run_id)
    first = tasks.claim(UUID(task["id"]))
    with Session(engine()) as db:
        row = db.get(AnalysisTask, first.task_id)
        row.lease_until = now() - timedelta(seconds=1)
        db.commit()
    second = tasks.claim(first.task_id)
    assert second.number == 2 and second.generation > first.generation
    with pytest.raises(tasks.TaskConflict, match="superseded"):
        tasks.finish(first, {"summary": "Late"}, PASS)
    tasks.reject(second, {"summary": "Still incomplete"}, FAIL, raw_output="invalid model result")
    assert tasks.claim(first.task_id) is None
    view = tasks.view(first.task_id)
    assert view["status"] == "exhausted"
    assert [item["status"] for item in view["attempt_details"]] == ["uncertain", "rejected"]
    assert tasks.budgets(run_id)["model_calls"] == 2


def test_receipt_and_event_rollback_together_and_failed_gates_cannot_accept(run_id, monkeypatch):
    task = make(run_id)
    claim = tasks.claim(UUID(task["id"]))
    with pytest.raises(ValueError, match="passing checks"):
        tasks.finish(claim, {}, FAIL)
    from backend.analysis import service

    original = service.emit

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("Commit interrupted")

    monkeypatch.setattr(service, "emit", fail)
    with pytest.raises(RuntimeError, match="Commit interrupted"):
        tasks.finish(claim, {"summary": "Result"}, PASS)
    assert tasks.view(claim.task_id)["attempts"] == []
    assert tasks.view(claim.task_id)["status"] == "running"
    monkeypatch.setattr(service, "emit", original)
    tasks.finish(claim, {"summary": "Result"}, PASS)


def test_column_and_obligation_allowances_cannot_reset_by_task_alias(run_id):
    first = make(run_id)
    original = tasks.claim(UUID(first["id"]))
    tasks.reject(original, {"summary": "Missing"}, FAIL)
    alias = make(run_id, key="another-label")
    retry = tasks.claim(UUID(alias["id"]))
    tasks.reject(retry, {"summary": "Still missing"}, FAIL)
    third = make(run_id, key="third-label")
    assert tasks.claim(UUID(third["id"])) is None
    for i in range(14):
        item = make(run_id, key=f"scope:{i}", obligation_keys=[f"distinct:{i}"])
        work = tasks.claim(UUID(item["id"]))
        assert work
        tasks.finish(work, {"summary": "Covered distinct scope"}, PASS)
    over = make(run_id, key="over", obligation_keys=["new-scope"])
    assert tasks.claim(UUID(over["id"])) is None
    assert tasks.budgets(run_id)["columns"]["c1"] == 16
    assert tasks.budgets(run_id)["shared_calls"] == 0


def test_dependencies_and_expansion_are_not_scientific_acceptance(run_id):
    parent = make(run_id)
    work = tasks.claim(UUID(parent["id"]))
    receipt = tasks.finish(work, {"requests": ["relationship:r1"]}, PASS, status="expanded")
    assert not tasks.coverage(run_id, parent["obligation_keys"])["complete"]
    assert tasks.claim(work.task_id) is None
    child = make(run_id, key="dependent", column="c2", dependency_receipt_ids=[receipt["id"]])
    assert tasks.claim(UUID(child["id"])) is None


def test_shared_reservations_are_atomic_separate_and_never_redispatched(run_id):
    def reserve(i):
        with Session(engine()) as db:
            run = db.get(AnalysisRun, run_id)
            tasks.reserve_shared(db, run, f"shared:{i}", "a" * 64)
            tasks.settle_shared(db, run_id, f"shared:{i}", status="accepted", receipt={"value": i})
            db.commit()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(reserve, range(27)))
    assert tasks.budgets(run_id)["model_calls"] == tasks.budgets(run_id)["shared_calls"] == 27
    with Session(engine()) as db:
        run = db.get(AnalysisRun, run_id)
        with pytest.raises(tasks.TaskConflict, match="already reserved"):
            tasks.reserve_shared(db, run, "shared:0", "a" * 64)
    with pytest.raises(tasks.AllowanceExhausted):
        reserve(28)
    column = make(run_id)
    assert tasks.claim(UUID(column["id"]))
    assert tasks.budgets(run_id)["model_calls"] == 28


def test_attempt_failure_and_database_immutability(run_id):
    task = make(run_id)
    work = tasks.claim(UUID(task["id"]))
    receipt = tasks.finish(work, {"summary": "Provider interrupted"}, FAIL, status="failed")
    assert tasks.view(work.task_id)["status"] == "failed"
    assert receipt["status"] == "failed"
    for statement in [
        delete(AnalysisTaskAttempt).where(AnalysisTaskAttempt.id == work.attempt_id),
        update(AnalysisTaskAttempt)
        .where(AnalysisTaskAttempt.id == work.attempt_id)
        .values(raw_output="replacement"),
        update(AnalysisTask).where(AnalysisTask.id == work.task_id).values(input_hash="b" * 64),
    ]:
        with Session(engine()) as db, pytest.raises(DBAPIError):
            db.execute(statement)
            db.commit()
    with Session(engine()) as db:
        run = db.get(AnalysisRun, run_id)
        tasks.reserve_shared(db, run, "shared", "a" * 64)
        db.commit()
    with Session(engine()) as db, pytest.raises(DBAPIError):
        db.execute(update(AnalysisRun).where(AnalysisRun.id == run_id).values(shared_calls=0))
        db.commit()


def test_exact_attempt_request_is_reserved_and_immutable(run_id):
    task = make(run_id)
    request = {
        "system": "Resolve meaning",
        "payload": {"evidence": ["e1"]},
        "schema": {"type": "object"},
        "output_limit": 1000,
        "aliases": {"e1": "columns.c1"},
    }
    work = tasks.claim(UUID(task["id"]), request=request, metrics={"input_tokens": 120})
    assert work.input_hash == task["input_hash"]
    saved_request = dict(request)
    request["system"] = "Changed after reservation"
    receipt = tasks.finish(work, {"summary": "Accepted"}, PASS, metrics={"output_tokens": 18})
    assert receipt["input_hash"] == fingerprint(saved_request)
    with Session(engine()) as db:
        assert tasks.input_for_receipt(db, run_id, receipt["id"]) == saved_request
    details = tasks.view(work.task_id)["attempt_details"][0]
    assert details["metrics"] == {"input_tokens": 120, "output_tokens": 18}
    with Session(engine()) as db, pytest.raises(DBAPIError):
        db.execute(
            update(AnalysisTaskAttempt)
            .where(AnalysisTaskAttempt.id == work.attempt_id)
            .values(request={"changed": True})
        )
        db.commit()


def test_prerequisites_require_complete_coarse_or_accepted_task_receipts_from_same_run(run_id):
    accepted = make(run_id)
    accepted_claim = tasks.claim(UUID(accepted["id"]))
    receipt = tasks.finish(accepted_claim, {"summary": "Scientific obligation accepted"}, PASS)
    rejected = make(run_id, key="rejected", column="c2")
    rejected_claim = tasks.claim(UUID(rejected["id"]))
    rejection = tasks.reject(rejected_claim, {"summary": "Not accepted"}, FAIL)
    with Session(engine()) as db:
        run = db.get(AnalysisRun, run_id)
        run.receipts = {
            "checkpoint:complete": {"id": "complete-checkpoint", "status": "accepted", "complete": True},
            "checkpoint:incomplete": {"id": "incomplete-checkpoint", "status": "accepted", "complete": False},
        }
        foreign = AnalysisRun(
            id=uuid4(),
            conversation_id=run.conversation_id,
            version_id=run.version_id,
            author_id=run.author_id,
            submission_id=uuid4(),
            question=run.question,
            trace_id=uuid4(),
            inputs=dict(run.inputs),
        )
        db.add(foreign)
        db.commit()
        assert receipt["complete"] is False
        tasks.require_dependencies(db, run, {receipt["id"], "complete-checkpoint"})
        for reference in (rejection["id"], "incomplete-checkpoint", "missing"):
            with pytest.raises(tasks.TaskConflict, match="accepted prerequisites"):
                tasks.require_dependencies(db, run, {reference})
        with pytest.raises(tasks.TaskConflict, match="accepted prerequisites"):
            tasks.require_dependencies(db, foreign, {receipt["id"], "complete-checkpoint"})


@contextmanager
def shared_ownership(run_id):
    key = int.from_bytes(run_id.bytes[:8], "big", signed=True)
    with engine().connect().execution_options(isolation_level="AUTOCOMMIT") as guard:
        guard.execute(text("SELECT pg_advisory_lock(:key)"), {"key": key})
        try:
            yield tasks.shared_owner(run_id, guard)
        finally:
            guard.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})


@pytest.mark.parametrize("run_id", [UUID(int=1), UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")])
def test_shared_owner_verifies_both_signed_advisory_key_encodings(run_id):
    with engine().connect().execution_options(isolation_level="AUTOCOMMIT") as guard:
        with pytest.raises(tasks.TaskConflict, match="advisory-lock owner"):
            tasks.shared_owner(run_id, guard)
    with shared_ownership(run_id) as owner:
        owner.verify(run_id)
        assert tasks.shared_owner(run_id, owner.connection) is owner
        with pytest.raises(tasks.TaskConflict, match="advisory-lock owner"):
            tasks.SharedOwner(run_id, owner.connection, uuid4()).verify(run_id)
    with pytest.raises(tasks.TaskConflict, match="advisory-lock owner"):
        owner.verify(run_id)


def test_shared_interruption_requires_new_owner_and_records_uncertainty_once(run_id):
    with shared_ownership(run_id) as first_owner:
        with Session(engine()) as db:
            run = db.get(AnalysisRun, run_id)
            assert tasks.reserve_shared(db, run, "target:0", "a" * 64, owner=first_owner) == "target:0"
            db.commit()
        with Session(engine()) as db:
            run = db.get(AnalysisRun, run_id)
            with pytest.raises(tasks.TaskConflict, match="same invocation"):
                tasks.reserve_shared(db, run, "target:0", "a" * 64, recover=True, owner=first_owner)
        with Session(engine()) as db:
            run = db.get(AnalysisRun, run_id)
            with pytest.raises(tasks.TaskConflict, match="advisory-lock owner"):
                tasks.reserve_shared(db, run, "target:0", "a" * 64, recover=True)
    with shared_ownership(run_id) as resumed_owner, Session(engine()) as db:
        run = db.get(AnalysisRun, run_id)
        replacement = tasks.reserve_shared(db, run, "target:0", "a" * 64, recover=True, owner=resumed_owner)
        assert replacement == "target:0:transport:1"
        tasks.settle_shared(db, run_id, replacement, status="accepted", receipt={"result": "saved"})
        db.commit()
        with pytest.raises(tasks.TaskConflict, match="already reserved"):
            tasks.reserve_shared(db, run, "target:0", "a" * 64, recover=True, owner=resumed_owner)
    with Session(engine()) as db:
        attempts = list(db.scalars(select(AnalysisTaskAttempt).where(AnalysisTaskAttempt.run_id == run_id)
                                   .order_by(AnalysisTaskAttempt.number)))
        assert [item.status for item in attempts] == ["uncertain", "accepted"]
        assert attempts[0].completed_at and attempts[0].receipt is None
        events = list(db.scalars(select(AnalysisEvent).where(AnalysisEvent.run_id == run_id)))
        assert sum(item.kind == "model_interrupted" for item in events) == 1
    assert tasks.budgets(run_id)["shared_calls"] == 2


def test_shared_transport_replacement_consumes_the_scientific_correction_allowance(run_id):
    with shared_ownership(run_id) as owner, Session(engine()) as db:
        run = db.get(AnalysisRun, run_id)
        for _ in range(2):
            reserved = tasks.reserve_shared(db, run, "target:0", "a" * 64, obligation_key="target", recover=True, owner=owner)
            tasks.settle_shared(db, run_id, reserved, status="failed", metrics={"error_type": "OSError"})
            db.commit()
        for key in ("target:0", "target:1"):
            with pytest.raises(tasks.AllowanceExhausted, match="including interrupted calls"):
                tasks.reserve_shared(db, run, key, "a" * 64, obligation_key="target", recover=True, owner=owner)
        attempts = list(db.scalars(select(AnalysisTaskAttempt).where(AnalysisTaskAttempt.run_id == run_id)))
        assert all(item.metrics["obligation_key"] == "target" and item.metrics["error_type"] == "OSError" for item in attempts)
    assert tasks.budgets(run_id)["shared_calls"] == 2


def test_each_diagnostic_action_shares_transport_and_correction_allowance(run_id):
    with shared_ownership(run_id) as owner, Session(engine()) as db:
        run = db.get(AnalysisRun, run_id)
        for action in range(2):
            obligation = f"diagnostic:{action}"
            key = f"{obligation}:0"
            for _ in range(2):
                reserved = tasks.reserve_shared(db, run, key, "a" * 64, obligation_key=obligation, recover=True, owner=owner)
                tasks.settle_shared(db, run_id, reserved, status="failed")
                db.commit()
            with pytest.raises(tasks.AllowanceExhausted):
                tasks.reserve_shared(db, run, f"{obligation}:1", "a" * 64, obligation_key=obligation, owner=owner)
    assert tasks.budgets(run_id)["shared_calls"] == 4


def test_task_publication_rejects_misrouted_run_and_attempt_number(run_id):
    task = make(run_id)
    claim = tasks.claim(UUID(task["id"]))
    with Session(engine()) as db:
        original = db.get(AnalysisRun, run_id)
        foreign = AnalysisRun(
            id=uuid4(), conversation_id=original.conversation_id, version_id=original.version_id,
            author_id=original.author_id, submission_id=uuid4(), question=original.question,
            trace_id=uuid4(), inputs=dict(original.inputs),
        )
        db.add(foreign)
        db.commit()
        foreign_id = foreign.id
    for invalid in (replace(claim, run_id=foreign_id), replace(claim, number=claim.number + 1)):
        with pytest.raises(tasks.TaskConflict, match="claim does not exist"):
            tasks.finish(invalid, {"summary": "Misrouted"}, PASS)
    assert tasks.view(claim.task_id)["status"] == "running"
    with Session(engine()) as db:
        assert not list(db.scalars(select(AnalysisEvent).where(AnalysisEvent.run_id == foreign_id)))
    receipt = tasks.finish(claim, {"summary": "Correct owner"}, PASS)
    assert receipt["status"] == "accepted"
    with pytest.raises(tasks.TaskConflict, match="claim does not exist"):
        tasks.finish(replace(claim, run_id=foreign_id), {"summary": "Correct owner"}, PASS)

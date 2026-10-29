"""Chat authority boundaries on real PostgreSQL and private object storage."""

from uuid import uuid4

import pytest
from sqlalchemy import select

from causal_agent.common.llm import set_llm
from causal_agent.desk.explainer import DiscussionReply
from causal_agent.desk.graph import Interpretation
from causal_agent.memory.context.conversation import ChatCommand, GapQuestion, Option, Reference
from causal_agent.memory.context.review import ColumnDisposition, ContextEdit, RevisionConflict
from causal_agent.memory.context.tests.builders import Scripted
from causal_agent.server.context import service as context_service
from causal_agent.server.context.repository import Repository as Context
from causal_agent.server.conversations import schema as S
from causal_agent.server.conversations import service
from causal_agent.server.conversations.commands import Busy, Repository
from causal_agent.server.conversations.manager import Manager
from causal_agent.server.conversations.runtime import Runtime
from causal_agent.server.ingestion.service import Missing

pytestmark = pytest.mark.ingestion


@pytest.fixture
def chat_setup(prepared, storage, settings):
    owner = str(uuid4())
    data = prepared(owner)
    context_service.generate(storage, owner, data["dataset_id"])
    manager = Manager(settings)

    def create():
        context = Context(storage, owner, data["dataset_id"])
        review = context.start_review("What do the recorded amounts mean?")
        context.save_review(review["id"], 1, [], [ColumnDisposition(column_id=c.column_id, status="unresolved") for c in review["columns"]])
        chat = service.create(storage, owner, data["dataset_id"], manager, review["id"], 2)
        return Repository(storage, owner, chat["id"])

    yield create, manager, storage, owner, data
    manager.pool.shutdown(wait=True)
    manager.connection.close()


def command(repo, kind="message", **kwargs):
    return ChatCommand(submission_id=str(uuid4()), revision_id=repo.state()["revision_id"], kind=kind, **kwargs)


def test_question_first_welcome_needs_no_model(chat_setup):
    create, manager, _, _, _ = chat_setup
    repo = create()
    view = manager.view(repo.conversation)
    assert view.question in view.prompt and "Context study" in view.prompt
    assert view.status == "idle" and view.message_count == 1
    assert view.analysis_review is None
    assert not hasattr(view, "transcript")
    assert not hasattr(view, "claims")


def test_duplicate_submission_and_active_turn_guard(chat_setup):
    create, _, _, _, _ = chat_setup
    repo = create()
    cmd = command(repo, text="Discuss this dataset")
    first, fresh = repo.accept(cmd)
    again, fresh_again = repo.accept(cmd)
    assert fresh and not fresh_again and first["id"] == again["id"]
    assert len(repo.messages()) == 2
    with pytest.raises(ValueError, match="different content"):
        repo.accept(cmd.model_copy(update={"text": "Different request"}))
    with pytest.raises(Busy):
        repo.accept(command(repo, text="Competing request"))


def test_revision_pinning_and_idempotent_correction(chat_setup):
    create, _, _, _, _ = chat_setup
    first, second = create(), create()
    selected = first.state()["revision_id"]
    assert second.state()["revision_id"] == selected
    old = first.context.read(selected)
    edit = ContextEdit(action="review", target_id=old.knowledge.statements[0].id)
    cmd = command(first, "correct", edits=[edit], text="Confirm the displayed statement")
    first.accept(cmd)
    new = first.correct(cmd.submission_id, cmd.edits, cmd.text)
    assert first.correct(cmd.submission_id, cmd.edits, cmd.text).id == new.id
    first.finish(cmd.submission_id, "Updated.")
    first.finish(cmd.submission_id, "Updated.")
    assert len(first.messages()) == 3
    assert second.state()["revision_id"] == selected
    assert first.context.read(selected) == old
    refresh = command(second, "refresh")
    second.accept(refresh)
    second.controls(refresh.submission_id)
    second.finish(refresh.submission_id, "Refreshed.")
    assert second.state()["revision_id"] == new.id


def test_stale_edit_and_wrong_reference_never_write(chat_setup):
    create, _, _, _, _ = chat_setup
    first, second = create(), create()
    old = first.state()["revision_id"]
    edit = ContextEdit(action="review", target_id=first.context.read(old).knowledge.statements[0].id)
    cmd = command(first, "correct", edits=[edit], text="Confirmed")
    first.accept(cmd)
    first.correct(cmd.submission_id, cmd.edits, cmd.text)
    first.finish(cmd.submission_id, "Updated")
    stale = command(second, "correct", edits=[edit], text="Confirmed")
    with pytest.raises(RevisionConflict):
        second.accept(stale)
    assert second.state()["revision_id"] == old
    assert second.state()["status"] == "idle"
    first_state = first.state()
    with pytest.raises(Missing):
        first.accept(command(first, "focus", references=[Reference(kind="column", id=str(uuid4()))]))
    assert first.state() == first_state


def test_unknown_answer_is_structured_and_not_authorization(chat_setup):
    create, manager, storage, _, _ = chat_setup
    repo = create()
    runtime = Runtime(repo, manager.s)
    question = GapQuestion(
        id=str(uuid4()),
        revision_id=repo.state()["revision_id"],
        target="claim:assignment.kind",
        text="How was the change assigned?",
        options=[Option(id="unknown", label="Not sure", unknown=True)],
    )
    runtime.set_preparation([question], None)
    cmd = command(repo, "answer", question_id=question.id, option_id="unknown")
    repo.accept(cmd)
    revision = repo.answer(cmd.submission_id)
    assert repo.answer(cmd.submission_id).id == revision.id
    field = runtime.memory().field("claim:assignment.kind")
    assert field.status == "unknown" and field.value is None
    assert repo.state()["preparation"] is None
    with storage.engine.connect() as db:
        answer = db.scalar(select(S.questions.c.answer).where(S.questions.c.id == question.id))
        assert answer["unknown"]


def test_restart_exposes_interruption_and_preserves_committed_update(chat_setup):
    create, manager, storage, owner, _ = chat_setup
    repo = create()
    old = repo.context.read(repo.state()["revision_id"])
    cmd = command(repo, "correct", edits=[ContextEdit(action="review", target_id=old.knowledge.statements[0].id)], text="Confirmed")
    repo.accept(cmd)
    revision = repo.correct(cmd.submission_id, cmd.edits, cmd.text)
    restarted = Manager(manager.s)
    try:
        restarted.bind(repo.conversation, storage, owner)
        assert restarted.view(repo.conversation).status == "interrupted"
        assert repo.state()["revision_id"] == revision.id
        restarted.resume(repo.conversation)
        restarted.futures[repo.conversation].result(timeout=10)
        assert restarted.view(repo.conversation).status == "idle"
        assert repo.state()["revision_id"] == revision.id
        assert len(repo.messages()) == 3
    finally:
        restarted.pool.shutdown(wait=True)
        restarted.connection.close()


@pytest.mark.parametrize(
    "mode,text", [("discussion", "What does amount represent?"), ("hypothetical", "Suppose the amounts were euros instead. What would that mean?")]
)
def test_discussion_and_hypotheticals_never_write_context(chat_setup, mode, text):
    create, manager, _, _, _ = chat_setup
    repo = create()
    before = repo.context.read(repo.state()["revision_id"])
    set_llm(Scripted([Interpretation(mode=mode, discussion=text), DiscussionReply(text="The recorded unit is dollars [context:s1].", cites=["context:s1"])]))
    manager.send(repo.conversation, command(repo, text=text))
    manager.futures[repo.conversation].result(timeout=10)
    assert manager.view(repo.conversation).status == "idle", manager.view(repo.conversation).error
    assert repo.context.read(repo.state()["revision_id"]) == before


def test_conversational_correction_uses_the_shared_revision_gate(chat_setup):
    create, manager, _, _, _ = chat_setup
    repo = create()
    text = "Please confirm the stated dollar units."
    edits = [ContextEdit(action="review", target_id="s1", reason="Confirmed the units.")]
    set_llm(Scripted([Interpretation(mode="correction", correction_passage=text, edits=edits)]))
    manager.send(repo.conversation, command(repo, text=text))
    manager.futures[repo.conversation].result(timeout=10)
    assert manager.view(repo.conversation).status == "idle", manager.view(repo.conversation).error
    revision = repo.context.read(repo.state()["revision_id"])
    assert next(s for s in revision.knowledge.statements if s.id == "s1").status == "reviewed"
    assert "Updated context revision" in repo.messages()[0]["text"]


def test_ambiguous_reference_does_not_clear_focus_or_mutate(chat_setup):
    create, manager, _, _, _ = chat_setup
    repo = create()
    column = repo.context.read(repo.state()["revision_id"]).knowledge.columns[0]
    focus = command(repo, "focus", references=[Reference(kind="column", id=column.column_id)])
    manager.send(repo.conversation, focus)
    manager.futures[repo.conversation].result(timeout=10)
    before = repo.state()["revision_id"]
    set_llm(Scripted([Interpretation(mode="discussion", clarification="Do you mean the participant code or the amount?")]))
    manager.send(repo.conversation, command(repo, text="Change that column"))
    manager.futures[repo.conversation].result(timeout=10)
    assert repo.state()["revision_id"] == before
    assert repo.state()["focus"]["id"] == column.column_id
    assert "Do you mean" in repo.messages()[0]["text"]


def test_fast_track_pauses_retains_questions_and_never_runs(chat_setup):
    from causal_agent.common.contracts import Candidate, QuestionFrame, Scope

    create, manager, storage, _, _ = chat_setup
    repo = create()
    runtime = Runtime(repo, manager.s)
    runtime.set_question_frame(
        QuestionFrame(
            intent="effect_of_change",
            decision_served="Compare amounts",
            outcome_candidates=[Candidate(column="amount", reason="outcome", cites=[])],
            cause_candidates=[Candidate(column="code", reason="exposure", cites=[])],
            scope=Scope(),
            relevant_columns=[],
            reasons=[],
        )
    )

    def send(kind, **kw):
        manager.send(repo.conversation, command(repo, kind, **kw))
        manager.futures[repo.conversation].result(timeout=10)
        view = manager.view(repo.conversation)
        assert view.status == "idle", view.error
        return view

    first = send("fast_track")
    assert first.questions and first.readiness.status == "needs_context"
    assert first.analysis_review is None
    question = first.questions[0]
    send("pause")
    assert manager.view(repo.conversation).questions[0].id == question.id
    resumed = send("fast_track")
    assert resumed.questions[0].model_dump() == question.model_dump()
    answered = send("answer", question_id=question.id, option_id="unknown")
    assert all(q.target != question.target for q in answered.questions)
    assert runtime.memory().field(question.target).status == "unknown"
    send("pause")
    again = send("fast_track")
    assert all(q.target != question.target for q in again.questions)
    with storage.engine.connect() as db:
        assert not list(
            db.scalars(select(S.commands.c.id).where(S.commands.c.conversation_id == repo.conversation, S.commands.c.body["kind"].as_string() == "authorize"))
        )
    checkpoint = manager.graph.get_state({"configurable": {"thread_id": repo.conversation}})
    assert set(checkpoint.values) == {"operation_id"}


def test_changed_revision_invalidates_preparation_before_run(chat_setup):
    create, manager, storage, _, _ = chat_setup
    repo = create()
    old = repo.state()["revision_id"]
    with storage.engine.begin() as db:
        repo.update(db, preparation={"id": "prepared", "revision_id": old})
    cmd = command(repo, "correct", edits=[ContextEdit(action="review", target_id="s1")], text="Confirm units")
    manager.send(repo.conversation, cmd)
    manager.futures[repo.conversation].result(timeout=10)
    assert repo.state()["preparation"] is None
    with pytest.raises(ValueError, match="review"):
        repo.accept(command(repo, "authorize", snapshot_id="prepared"))
    assert repo.state()["status"] == "idle"


def test_interrupted_external_task_is_never_resubmitted(chat_setup, monkeypatch):
    create, manager, _, _, _ = chat_setup
    repo = create()
    runtime = Runtime(repo, manager.s)
    cmd = command(repo, text="Draw the amounts")
    repo.accept(cmd)
    runtime.save_progress(cmd.submission_id, exploration_started=True)

    def forbidden(**kw):
        raise AssertionError("An interrupted request was resubmitted")

    monkeypatch.setattr("causal_agent.server.conversations.runtime.exploration.explore", forbidden)
    assert "has not been resubmitted" in runtime.explore_once(cmd.submission_id, cmd.text, [])


def test_multi_intent_keeps_correction_tool_and_discussion(chat_setup, monkeypatch):
    from uuid import UUID

    from fastapi import HTTPException

    from causal_agent.server.conversations.routes import evidence

    create, manager, storage, owner, _ = chat_setup
    repo = create()
    calls = []
    monkeypatch.setattr(Runtime, "explore_once", lambda self, op, request, refs: calls.append(request) or "Visualization saved.")
    text = "Confirm dollar units. Draw a chart. Explain the units."
    set_llm(
        Scripted(
            [
                Interpretation(
                    mode="correction",
                    correction_passage="Confirm dollar units.",
                    edits=[ContextEdit(action="review", target_id="s1")],
                    exploration="Draw a chart.",
                    exploration_passage="Draw a chart.",
                    discussion="Explain the units.",
                ),
                DiscussionReply(text="The unit is dollars [context:s1].", cites=["context:s1"]),
            ]
        )
    )
    submitted = command(repo, text=text)
    manager.send(repo.conversation, submitted)
    manager.futures[repo.conversation].result(timeout=10)
    view = manager.view(repo.conversation)
    assert view.status == "idle", view.error
    assert calls == ["Draw a chart."]
    assert "Updated context revision" in view.prompt and "Visualization saved" in view.prompt and "unit is dollars" in view.prompt
    source = evidence(UUID(repo.conversation), owner, storage, repo.chat, "operation:" + submitted.submission_id)
    assert "Visualization saved" in source.text
    other = create()
    with pytest.raises(HTTPException) as rejected:
        evidence(UUID(other.conversation), owner, storage, other.chat, "operation:" + submitted.submission_id)
    assert rejected.value.status_code == 404


def test_causal_publication_is_idempotent_and_evidence_survives_restart(chat_setup, tmp_path):
    import json

    from causal_agent.common.contracts import RunRecord
    from causal_agent.desk.handoff import forced
    from causal_agent.server.conversations.publication import Repository as Products

    create, manager, storage, owner, _ = chat_setup
    repo = create()
    runtime = Runtime(repo, manager.s)
    handoff = forced(repo.conversation, "Did code change amount?", "adjustment", "amount", "code", [], memory=runtime.memory())
    cmd = command(repo, text="Recorded result publication")
    repo.accept(cmd)
    record = RunRecord(
        index=1,
        dataset=repo.conversation,
        question=handoff.question,
        family="adjustment",
        status="context_deficient",
        specialist_result={
            "status": "context_deficient",
            "context_deficiency": {"stage": "load", "addresses": ["claim:assignment.kind"], "reason": "Unresolved assignment"},
            "estimates": [],
        },
    )
    directory = tmp_path / "assembled"
    directory.mkdir()
    (directory / "record.json").write_text(record.model_dump_json())
    prepared = {"handoff": handoff.model_dump()}
    # Simulate a crash after request creation but before saving that progress.
    runtime.artifacts.operation_id = cmd.submission_id
    original_request = runtime.artifacts.begin(repo.conversation, repo.state()["question"], "Authorized causal analysis", repo.context.info["sha256"], [])
    runtime.publish_analysis(cmd, prepared, record, directory)
    runtime.publish_analysis(cmd, prepared, record, directory)
    products = Products(storage, owner, repo.conversation)
    saved = products.list(repo.conversation)
    assert len(saved) == 1 and saved[0].origin == "analysis" and saved[0].request_id == original_request[0]
    artifact = saved[0]
    assert json.loads(products.read_file(artifact.id, "analysis.json"))["context_deficiency"]["reason"] == "Unresolved assignment"
    material = Runtime(Repository(storage, owner, repo.conversation), manager.s).material([Reference(kind="artifact", id=artifact.id)])
    assert "Unresolved assignment" in material.text
    assert not material.numbers  # A context deficiency must never acquire an estimate.


def test_external_attempt_identity_and_registered_elements_are_checked(chat_setup):
    from causal_agent.server.conversations.publication import Repository as Products

    create, _, storage, owner, data = chat_setup
    repo = create()
    products = Products(storage, owner, repo.conversation)
    products.operation_id = str(uuid4())
    first = products.begin(repo.conversation, "Describe amounts", "", data["sha256"], [])
    assert products.begin(repo.conversation, "Describe amounts", "", data["sha256"], []) == first
    with pytest.raises(ValueError, match="different inputs"):
        products.begin(repo.conversation, "Different request", "", data["sha256"], [])
    products.external_task(first[1], "completed-a2a-task")
    products.external_task(first[1], "completed-a2a-task")
    with pytest.raises(ValueError, match="different external task"):
        products.external_task(first[1], "another-task")

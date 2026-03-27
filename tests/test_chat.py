"""Persisted chat journeys; real-model/runner acceptance is separately opt-in."""

from uuid import UUID, uuid4

import pytest
from conftest import PASSWORD
from model_double import model_double
from sqlalchemy import update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from backend import model_access
from backend.chat import delivery, jobs
from backend.chat.contracts import Exploration, Interpretation
from backend.chat.exploration import validate_record
from backend.config import settings
from backend.db import engine
from backend.integration.readers import Analyses
from backend.models import ChatTurn, Conversation
from backend.responses import Reply


@pytest.fixture(autouse=True)
def storage(isolated, tmp_path, monkeypatch):
    monkeypatch.setattr(settings(), "storage_dir", tmp_path)


def create(owner):
    data = owner.post(
        "/api/datasets",
        data={"title": "Chat sample", "submission_id": str(uuid4())},
        files={"file": ("sample.csv", b"group,score\na,2\na,4\nb,8\n", "text/csv")},
    ).json()
    version = data["version"]["id"]
    body = {"title": "Discuss scores", "submission_id": str(uuid4()), "context_revision_id": None}
    base = f"/api/versions/{version}/conversations"
    response = owner.post(base, json=body)
    assert response.status_code == 201, response.text
    return base, body, response.json()


def send(
    owner, identity, text="Remember that my question concerns the score column.", expected=0, submission=None
):
    return owner.post(
        f"/api/conversations/{identity}/turns",
        json={"submission_id": submission or str(uuid4()), "expected_turn": expected, "text": text},
    )


def scripted(schema, system, payload, model, **kwargs):
    if schema is Interpretation:
        return Interpretation()
    assert schema is Reply
    return Reply(
        text="The saved question concerns score. This discussion does not establish a causal effect."
    )


def test_saved_multiturn_and_owner_isolation(owner, monkeypatch):
    base, body, conversation = create(owner)
    cid = conversation["id"]
    assert owner.post(base, json=body).json()["id"] == cid
    assert owner.post(base, json={**body, "title": "Changed"}).status_code == 409
    packets = []

    def infer(*args, **kwargs):
        packets.append(args[2])
        return scripted(*args, **kwargs)

    monkeypatch.setattr(model_access, "structured", model_double(infer))
    first = send(owner, cid).json()
    jobs.process(UUID(first["id"]), Analyses())
    response = send(owner, cid, "Which column did I ask about?", expected=1)
    assert response.status_code == 202, response.text
    jobs.process(UUID(response.json()["id"]), Analyses())
    saved = owner.get(f"/api/conversations/{cid}").json()
    assert len(saved["turns"]) == 2 and all(t["status"] == "succeeded" for t in saved["turns"]), saved
    assert packets[-1]["history"][0]["user"] == first["text"]
    assert packets[-1]["history"][0]["assistant"]["text"] == saved["turns"][0]["reply"]["text"]
    monkeypatch.setattr(model_access, "structured", model_double(lambda *a, **kw: pytest.fail("Reads cannot call a model")))
    assert owner.get(f"/api/conversations/{cid}").json() == saved
    owner.post("/api/auth/login", json={"email": "other@example.com", "password": PASSWORD})
    assert owner.get(f"/api/conversations/{cid}").status_code == 404
    assert owner.get(base).status_code == 404
    assert send(owner, cid, expected=2).status_code == 404


def test_duplicate_conflicting_stale_and_concurrent_sends(owner):
    _, _, chat = create(owner)
    cid = chat["id"]
    submission = str(uuid4())
    first = send(owner, cid, submission=submission)
    assert first.status_code == 202
    assert send(owner, cid, submission=submission).json()["id"] == first.json()["id"]
    assert send(owner, cid, "different", submission=submission).status_code == 409
    assert send(owner, cid, expected=0).status_code == 409
    assert send(owner, cid, expected=1).status_code == 409


def test_queue_failure_rolls_back_turn(owner, monkeypatch):
    _, _, chat = create(owner)

    def fail(*args):
        raise OSError("queue unavailable")

    monkeypatch.setattr(delivery, "enqueue", fail)
    assert send(owner, chat["id"]).status_code == 503
    assert owner.get(f"/api/conversations/{chat['id']}").json()["turns"] == []


def test_failed_turn_resumes_completed_model_work(owner, monkeypatch):
    _, _, chat = create(owner)

    def broken(schema, *args, **kwargs):
        if schema is Reply:
            raise RuntimeError("provider unavailable")
        return Interpretation()

    monkeypatch.setattr(model_access, "structured", model_double(broken))
    first = send(owner, chat["id"]).json()
    identity = UUID(first["id"])
    jobs.process(identity, Analyses())
    saved = owner.get(f"/api/conversations/{chat['id']}").json()["turns"][0]
    assert saved["status"] == "failed" and saved["calls"] == 2

    def recovered(schema, *args, **kwargs):
        assert schema is Reply, "A saved interpretation should not be repeated"
        return Reply(text="Recovered reply.")

    monkeypatch.setattr(model_access, "structured", model_double(recovered))
    assert owner.post(f"/api/conversations/{chat['id']}/turns/{identity}/retry", json={}).status_code == 202
    jobs.process(identity, Analyses())
    result = owner.get(f"/api/conversations/{chat['id']}").json()["turns"][0]
    assert result["status"] == "succeeded" and result["calls"] == 3
    jobs.process(identity, Analyses())
    assert owner.get(f"/api/conversations/{chat['id']}").json()["turns"][0] == result


@pytest.mark.parametrize("repeat_invalid", [False, True])
def test_malformed_reply_uses_one_saved_correction(owner, monkeypatch, repeat_invalid):
    from pydantic import ValidationError

    _, _, chat = create(owner)

    def infer(schema, system, payload, model, **kwargs):
        if schema is Interpretation:
            return Interpretation()
        if "validation_error" not in payload or repeat_invalid:
            try:
                Reply.model_validate({"text": "x", "analysis_references": ["x"] * 101})
            except ValidationError as error:
                raise model_access.InvalidModelOutput('{"malformed": true}', error) from error
        assert payload["rejected_reply"]["raw"] == '{"malformed": true}'
        return Reply(text="The corrected response.")

    monkeypatch.setattr(model_access, "structured", model_double(infer))
    turn = send(owner, chat["id"]).json()
    jobs.process(UUID(turn["id"]), Analyses())
    saved = owner.get(f"/api/conversations/{chat['id']}").json()["turns"][0]
    assert saved["calls"] == 3
    assert saved["status"] == ("failed" if repeat_invalid else "succeeded")
    with Session(engine()) as db:
        assert db.get(ChatTurn, UUID(turn["id"])).receipts["invalid_model:reply"]["raw"]
    if repeat_invalid:
        assert not saved["retryable"] and "after one correction" in saved["error"]


def test_completed_turn_and_conversation_inputs_are_immutable(owner, monkeypatch):
    _, _, chat = create(owner)
    monkeypatch.setattr(model_access, "structured", model_double(scripted))
    turn = send(owner, chat["id"]).json()
    jobs.process(UUID(turn["id"]), Analyses())
    with Session(engine()) as db:
        with pytest.raises(DBAPIError):
            db.execute(update(ChatTurn).where(ChatTurn.id == UUID(turn["id"])).values(text="rewritten"))
            db.commit()
        db.rollback()
        with pytest.raises(DBAPIError):
            db.execute(
                update(Conversation).where(Conversation.id == UUID(chat["id"])).values(title="rewritten")
            )
            db.commit()


def test_forged_context_and_artifact_bindings_rejected(owner):
    base, body, chat = create(owner)
    assert (
        owner.post(
            base, json={**body, "submission_id": str(uuid4()), "context_revision_id": str(uuid4())}
        ).status_code
        == 404
    )
    response = owner.post(
        f"/api/conversations/{chat['id']}/turns",
        json={
            "submission_id": str(uuid4()),
            "expected_turn": 0,
            "text": "Use this chart",
            "artifact_ids": [str(uuid4())],
        },
    )
    assert response.status_code == 404


def test_numeric_and_product_gates():
    steps = [{"ok": True, "facts": {"mean": 3.0}, "files": {}}]
    record = Exploration(
        status="completed",
        summary="The mean is 3.",
        step=1,
        facts=["mean"],
        method="Arithmetic mean.",
        findings=["The mean is 3 [mean]."],
        limitations=["Descriptive only."],
    )
    validate_record(record, steps, set())
    with pytest.raises(ValueError, match="lacks support"):
        validate_record(record.model_copy(update={"summary": "The mean is 999."}), steps, set())
    with pytest.raises(ValueError, match="pixels"):
        validate_record(record.model_copy(update={"figure": True}), steps, set())
    with pytest.raises(ValueError, match="bracket"):
        validate_record(record.model_copy(update={"findings": ["The mean is 3 [invented]."]}), steps, set())


def test_grouped_number_formatting_keeps_exact_source_binding():
    from backend.numerical import numbers, validate
    from backend.validation import ReplyNumber

    assert numbers("42,613 people; -1,234.5; 1.2e3; list [1, 234].") == [42613, -1234.5, 1200, 1, 234]
    reply = Reply(text="The sample contains 42,613 people.", numbers=[ReplyNumber(source="n", value=42613)])
    validate(reply, {"n": {"value": 42613}})
    with pytest.raises(ValueError, match="does not match"):
        validate(reply, {"n": {"value": 42693}})


def test_reply_numbers_bind_to_exact_evidence():
    from backend.numerical import validate
    from backend.validation import ReplyNumber

    evidence = {"mean": {"value": 71.65426356589147}, "count": {"value": 645}}
    validate(
        Reply(
            text="The mean is 71.65 for 645 observations.",
            numbers=[ReplyNumber(source="mean", value=71.65), ReplyNumber(source="count", value=645)],
        ),
        evidence,
    )
    with pytest.raises(ValueError, match="metadata"):
        validate(Reply(text="The mean is 71.65."), evidence)
    with pytest.raises(ValueError, match="does not match"):
        validate(
            Reply(text="The mean is 71.65.", numbers=[ReplyNumber(source="count", value=71.65)]), evidence
        )


def test_worded_duration_is_not_a_numeric_operand_and_reply_packet_says_so(owner, monkeypatch):
    from backend.numerical import validate
    from backend.validation import ReplyNumber

    _, _, chat = create(owner)
    packets = []

    def infer(schema, system, payload, model, **kwargs):
        if schema is Reply:
            packets.append(payload)
            return Reply(text="The follow-up takes place one month after assignment.")
        return Interpretation()

    monkeypatch.setattr(model_access, "structured", model_double(infer))
    turn = send(owner, chat["id"], "Follow-up takes place one month after assignment.").json()
    jobs.process(UUID(turn["id"]), Analyses())
    assert not any(item["reference"] == "user.current" for item in packets[0]["numeric_evidence"].values())
    saved = owner.get(f"/api/conversations/{chat['id']}").json()
    assert saved["turns"][0]["status"] == "succeeded"
    with pytest.raises(ValueError, match="does not match"):
        validate(Reply(text="Follow-up at 1 month.", numbers=[ReplyNumber(source="duration", value=1)]),
                 {"duration": {"value": "Follow-up takes place one month after assignment."}})


def test_program_literals_are_selected_source_evidence_without_execution():
    from backend.numerical import program_literals

    sources = program_literals(
        "selected.py", "alpha = 0.05\ncorrection = -1\nflag = True\nraise RuntimeError('not executed')"
    )
    assert {record["value"] for record in sources.values()} == {0.05, -1}
    assert all(record["basis"] == "program_literal_not_result" for record in sources.values())
    assert program_literals("broken.py", "if (") == {}


def test_pinned_context_does_not_adopt_new_head(owner, monkeypatch):
    from backend.context.contracts import empty_content
    from backend.context.service import append_revision, locked_pack
    from backend.models import ContextRevision, Dataset, DatasetVersion

    base, body, initial = create(owner)
    version_id = UUID(initial["version_id"])
    with Session(engine()) as db:
        version = db.get(DatasetVersion, version_id)
        dataset = db.get(Dataset, version.dataset_id)
        owner_id = dataset.owner_id
        pack = locked_pack(db, version_id)
        content = empty_content(version_id, version.columns)
        first = append_revision(db, pack, None, uuid4(), uuid4(), "a" * 64, owner_id, content, [], "draft")
        db.commit()
        first_id = first.id
    response = owner.post(
        base, json={**body, "submission_id": str(uuid4()), "context_revision_id": str(first_id)}
    )
    assert response.status_code == 201, response.text
    chat = response.json()
    with Session(engine()) as db:
        pack = locked_pack(db, version_id)
        first = db.get(ContextRevision, first_id)
        append_revision(db, pack, first, uuid4(), uuid4(), "b" * 64, owner_id, content, [], "reviewed")
        db.commit()
    seen = []

    def infer(schema, system, payload, model, **kwargs):
        seen.append(payload["context"]["revision_id"])
        return scripted(schema, system, payload, model, **kwargs)

    monkeypatch.setattr(model_access, "structured", model_double(infer))
    turn = send(owner, chat["id"]).json()
    jobs.process(UUID(turn["id"]), Analyses())
    current = owner.get(f"/api/conversations/{chat['id']}").json()
    assert current["newer_context_available"] is True
    assert current["context_revision_id"] == str(first_id)
    assert set(seen) == {str(first_id)}


@pytest.fixture
def context_chat(owner):
    from backend.context.contracts import Participant, Relationship, SourceReference, empty_content
    from backend.context.service import append_revision, locked_pack
    from backend.models import Dataset, DatasetVersion

    base, body, initial = create(owner)
    version_id = UUID(initial["version_id"])
    with Session(engine()) as db:
        version = db.get(DatasetVersion, version_id)
        dataset = db.get(Dataset, version.dataset_id)
        content = empty_content(version_id, version.columns)
        content.relationships["known"] = Relationship(
            value="Group was assigned before the score was recorded.",
            status="reviewed",
            source=SourceReference(kind="message", id=uuid4(), author_id=dataset.owner_id),
            participants=[Participant(kind="column", id=cid) for cid in content.columns],
        )
        revision = append_revision(
            db,
            locked_pack(db, version_id),
            None,
            uuid4(),
            uuid4(),
            "c" * 64,
            dataset.owner_id,
            content,
            [],
            "reviewed",
        )
        db.commit()
        revision_id = str(revision.id)
    response = owner.post(
        base, json={**body, "submission_id": str(uuid4()), "context_revision_id": revision_id}
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.parametrize(
    "first_key",
    [
        "relationships.known",
        "relationships.missing",
        "concepts.missing",
        "columns.missing.definition.missing",
        "sources.missing",
    ],
)
def test_reply_context_citations_verified_and_corrected(owner, context_chat, monkeypatch, first_key):
    replies = []

    def infer(schema, system, payload, model, **kwargs):
        if schema is Interpretation:
            return Interpretation()
        replies.append(payload)
        if len(replies) == 2:
            assert "validation_error" in payload
            assert payload["rejected_reply"]["context_keys"] == [first_key]
        return Reply(
            text="Group was assigned before the score was recorded.",
            context_keys=[first_key if len(replies) == 1 else "relationships.known"],
        )

    monkeypatch.setattr(model_access, "structured", model_double(infer))
    turn = send(owner, context_chat["id"]).json()
    jobs.process(UUID(turn["id"]), Analyses())
    saved = owner.get(f"/api/conversations/{context_chat['id']}").json()
    result = saved["turns"][0]
    assert result["status"] == "succeeded", result
    assert result["reply"]["context_keys"] == ["relationships.known"]
    assert result["calls"] == (2 if first_key == "relationships.known" else 3)
    assert all(p["context"]["revision_id"] == context_chat["context_revision_id"] for p in replies)
    assert saved["artifacts"] == []


def test_context_citation_correction_is_bounded(owner, context_chat, monkeypatch):
    def infer(schema, *args, **kwargs):
        if schema is Interpretation:
            return Interpretation()
        return Reply(text="An unverified reply.", context_keys=["relationships.missing"])

    monkeypatch.setattr(model_access, "structured", model_double(infer))
    turn = send(owner, context_chat["id"]).json()
    identity = UUID(turn["id"])
    jobs.process(identity, Analyses())
    saved = owner.get(f"/api/conversations/{context_chat['id']}").json()
    result = saved["turns"][0]
    assert result["status"] == "failed" and result["calls"] == 3
    assert result["reply"] is None and saved["artifacts"] == []
    assert "after one correction" in result["error"]
    assert not result["retryable"]
    assert owner.post(f"/api/conversations/{context_chat['id']}/turns/{identity}/retry").status_code == 409
    jobs.process(identity, Analyses())
    assert owner.get(f"/api/conversations/{context_chat['id']}").json() == saved


def test_context_correction_recovers_saved_reply_without_repeating_work(owner, context_chat, monkeypatch):
    def broken(schema, system, payload, model, **kwargs):
        if schema is Interpretation:
            return Interpretation()
        if "validation_error" in payload:
            raise RuntimeError("provider unavailable during correction")
        return Reply(text="An unverified reply.", context_keys=["relationships.missing"])

    monkeypatch.setattr(model_access, "structured", model_double(broken))
    turn = send(owner, context_chat["id"]).json()
    identity = UUID(turn["id"])
    jobs.process(identity, Analyses())
    saved = owner.get(f"/api/conversations/{context_chat['id']}").json()["turns"][0]
    assert saved["status"] == "failed" and saved["calls"] == 3 and saved["retryable"]

    def recovered(schema, system, payload, model, **kwargs):
        assert schema is Reply and "validation_error" in payload
        return Reply(text="Group preceded the score.", context_keys=["relationships.known"])

    monkeypatch.setattr(model_access, "structured", model_double(recovered))
    assert owner.post(f"/api/conversations/{context_chat['id']}/turns/{identity}/retry").status_code == 202
    jobs.process(identity, Analyses())
    result = owner.get(f"/api/conversations/{context_chat['id']}").json()["turns"][0]
    assert result["status"] == "succeeded" and result["calls"] == 4
    with Session(engine()) as db:
        receipts = db.get(ChatTurn, identity).receipts
        assert receipts["model:reply"]["context_keys"] == ["relationships.missing"]
        assert receipts["model:reply-correction"]["context_keys"] == ["relationships.known"]


def test_reply_cannot_cite_context_without_a_pinned_revision(owner, monkeypatch):
    _, _, chat = create(owner)

    def infer(schema, system, payload, model, **kwargs):
        if schema is Interpretation:
            return Interpretation()
        if "validation_error" in payload:
            assert "without a selected revision" in payload["validation_error"]
            assert payload["context"] == {}
            return Reply(text="No context is selected for this conversation.")
        return Reply(text="An unverified reply.", context_keys=["relationships.known"])

    monkeypatch.setattr(model_access, "structured", model_double(infer))
    turn = send(owner, chat["id"]).json()
    jobs.process(UUID(turn["id"]), Analyses())
    result = owner.get(f"/api/conversations/{chat['id']}").json()["turns"][0]
    assert result["status"] == "succeeded" and result["calls"] == 3
    assert result["reply"]["context_keys"] == []

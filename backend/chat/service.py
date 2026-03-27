"""Owned conversation operations and atomic turn submission."""

import json
from typing import cast
from uuid import UUID, uuid4, uuid5

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.chat import prompts
from backend.chat.contracts import (
    ArtifactView,
    ConversationDetail,
    ConversationView,
    CreateConversation,
    Exploration,
    ScriptAttempt,
    SendTurn,
    TurnView,
)
from backend.config import settings
from backend.context.service import fingerprint, owned_version, revision_record
from backend.handoffs import AnalysisDiscussionReader, ConversationScope
from backend.models import (
    ChatArtifact,
    ChatContextSelection,
    ChatTurn,
    ContextPack,
    Conversation,
    Dataset,
    DatasetVersion,
    now,
)


def active_revision(db: Session, conversation: Conversation) -> UUID | None:
    selected = db.scalar(
        select(ChatContextSelection.revision_id)
        .where(ChatContextSelection.conversation_id == conversation.id)
        .order_by(ChatContextSelection.created_at.desc())
        .limit(1)
    )
    return selected or conversation.context_revision_id


def owned(db: Session, owner: UUID, identity: UUID, *, lock: bool = False) -> Conversation:
    query = (
        select(Conversation)
        .join(DatasetVersion)
        .join(Dataset)
        .where(Conversation.id == identity, Dataset.owner_id == owner)
    )
    if lock:
        query = query.with_for_update(of=Conversation)
    row = db.scalar(query)
    if row is None:
        raise HTTPException(404, "Conversation not found.")
    return row


def conversation_scope(db: Session, owner: UUID, identity: UUID, *, lock: bool = False) -> ConversationScope:
    conversation = owned(db, owner, identity, lock=lock)
    exposure = db.scalar(select(func.count()).select_from(ChatArtifact).join(ChatTurn)
                         .where(ChatTurn.conversation_id == conversation.id)) or 0
    return ConversationScope(conversation.id, conversation.version_id, active_revision(db, conversation), exposure)


def create(db: Session, owner: UUID, version_id: UUID, body: CreateConversation) -> ConversationView:
    version = owned_version(db, owner, version_id)
    db.execute(select(DatasetVersion).where(DatasetVersion.id == version.id).with_for_update())
    if not body.title.strip():
        raise ValueError("Give the conversation a title.")
    identity = uuid5(version_id, f"conversation:{body.submission_id}")
    row = db.get(Conversation, identity)
    if row:
        if row.title != body.title.strip() or row.context_revision_id != body.context_revision_id:
            raise HTTPException(409, "This conversation submission has different inputs.")
    else:
        if body.context_revision_id:
            revision_record(db, version_id, body.context_revision_id)
        row = Conversation(
            id=identity,
            version_id=version_id,
            context_revision_id=body.context_revision_id,
            title=body.title.strip(),
            submission_id=body.submission_id,
        )
        db.add(row)
    db.commit()
    return ConversationView.model_validate(row)


def list_conversations(db: Session, owner: UUID, version_id: UUID) -> list[ConversationView]:
    owned_version(db, owner, version_id)
    return [
        ConversationView.model_validate(row)
        for row in db.scalars(
            select(Conversation)
            .where(Conversation.version_id == version_id)
            .order_by(Conversation.created_at.desc())
        )
    ]


def turn_view(
    turn: ChatTurn, latest: int, original_revision: UUID | None = None, applied_revision: UUID | None = None
) -> TurnView:
    scripts = [
        value
        for key, value in sorted(
            turn.receipts.items(),
            key=lambda entry: int(entry[0].split(":")[1]) if entry[0].startswith("tool:") else -1,
        )
        if key.startswith("tool:") and isinstance(value, dict) and "ok" in value
    ]
    return TurnView.model_validate(
        {
            **{
                name: getattr(turn, name)
                for name in TurnView.model_fields
                if name
                not in {"retryable", "script_attempts", "mode", "context_revision_id", "applied_revision_id"}
            },
            "retryable": turn.status == "failed"
            and turn.number == latest
            and turn.calls < 24
            and turn.stage != "rejected",
            "script_attempts": [
                ScriptAttempt(step=index + 1, ok=bool(value["ok"]), stderr=str(value["stderr"]))
                for index, value in enumerate(scripts)
            ],
            "mode": turn.inputs.get("mode", "chat"),
            "context_revision_id": turn.inputs.get("context_revision_id", original_revision),
            "applied_revision_id": applied_revision,
        }
    )


def artifact_view(row: ChatArtifact) -> ArtifactView:
    return ArtifactView(
        id=row.id,
        turn_id=row.turn_id,
        record=Exploration.model_validate(row.record["exploration"]),
        facts=cast(dict[str, float], row.record["facts"]),
        filenames=list(row.files),
        created_at=row.created_at,
    )


def detail(db: Session, conversation: Conversation) -> ConversationDetail:
    turns = list(
        db.scalars(
            select(ChatTurn).where(ChatTurn.conversation_id == conversation.id).order_by(ChatTurn.number)
        )
    )
    artifacts = db.scalars(
        select(ChatArtifact)
        .join(ChatTurn)
        .where(ChatTurn.conversation_id == conversation.id)
        .order_by(ChatTurn.number)
    )
    current = db.scalar(
        select(ContextPack.current_revision_id).where(ContextPack.version_id == conversation.version_id)
    )
    selected_revision = active_revision(db, conversation)
    applications = {
        row.turn_id: row.revision_id
        for row in db.scalars(
            select(ChatContextSelection).where(ChatContextSelection.conversation_id == conversation.id)
        )
    }
    return ConversationDetail(
        **{
            **ConversationView.model_validate(conversation).model_dump(),
            "context_revision_id": selected_revision,
        },
        turns=[
            turn_view(t, len(turns), conversation.context_revision_id, applications.get(t.id)) for t in turns
        ],
        artifacts=[artifact_view(a) for a in artifacts],
        newer_context_available=current is not None and current != selected_revision,
    )


def artifact_record(db: Session, conversation_id: UUID, identity: UUID) -> ChatArtifact:
    record = db.scalar(
        select(ChatArtifact)
        .join(ChatTurn)
        .where(ChatArtifact.id == identity, ChatTurn.conversation_id == conversation_id)
    )
    if record is None:
        raise HTTPException(404, "Artifact not found in this conversation.")
    return record


def send(db: Session, conversation: Conversation, body: SendTurn, analysis_reader: AnalysisDiscussionReader) -> TurnView:
    from backend.chat.delivery import enqueue

    identity = uuid5(conversation.id, f"turn:{body.submission_id}")
    digest = fingerprint(body.model_dump(mode="json"))
    latest = (
        db.scalar(select(func.max(ChatTurn.number)).where(ChatTurn.conversation_id == conversation.id)) or 0
    )
    existing = db.get(ChatTurn, identity)
    if existing:
        if existing.request_hash != digest:
            raise HTTPException(409, "This message submission has different inputs.")
        return turn_view(existing, latest)
    if not body.text.strip():
        raise ValueError("Write a message before sending.")
    if latest != body.expected_turn:
        raise HTTPException(409, "This conversation changed. Refresh before sending your message.")
    if latest >= 60:
        raise HTTPException(
            409,
            "This conversation reached its 60-turn limit. Start a new chat; its saved history remains available.",
        )
    if db.scalar(
        select(ChatTurn.id).where(
            ChatTurn.conversation_id == conversation.id, ChatTurn.status.in_(["queued", "running"])
        )
    ):
        raise HTTPException(
            409, "A reply is already in progress. Wait for it before sending another message."
        )
    for aid in body.artifact_ids:
        artifact_record(db, conversation.id, aid)
    analyses = analysis_reader.capture(db, conversation.id)
    # The prompts and model are frozen here; immutable earlier turns supply history.
    inputs: dict[str, object] = {
        "mode": body.mode,
        "context_revision_id": str(active_revision(db, conversation))
        if active_revision(db, conversation)
        else None,
        "analysis_ids": list(analyses),
        "analysis_bindings": analyses,
        "artifact_ids": [str(a) for a in body.artifact_ids],
        "model": settings().gemini_model,
        "contract": prompts.VERSION,
        "interpret": prompts.INTERPRET,
        "discuss": prompts.DISCUSS,
        "explore": prompts.EXPLORE,
        "finish": prompts.FINISH,
    }
    history = list(db.scalars(select(ChatTurn).where(ChatTurn.conversation_id == conversation.id)))
    if sum(len(json.dumps(t.reply)) + len(t.text) for t in history) + len(body.text) > 240_000:
        raise HTTPException(
            409,
            "This conversation has reached its context limit. Start a new chat; no earlier messages were silently dropped.",
        )
    turn = ChatTurn(
        id=identity,
        conversation_id=conversation.id,
        number=latest + 1,
        submission_id=body.submission_id,
        request_hash=digest,
        text=body.text,
        inputs=inputs,
        trace_id=uuid4(),
    )
    db.add(turn)
    db.flush()
    turn.job_id = enqueue(db, identity)
    db.commit()
    return turn_view(turn, turn.number)


def retry(db: Session, conversation: Conversation, identity: UUID) -> TurnView:
    from backend.chat.delivery import enqueue

    turn = db.get(ChatTurn, identity)
    if turn is None or turn.conversation_id != conversation.id:
        raise HTTPException(404, "Message not found.")
    latest = (
        db.scalar(select(func.max(ChatTurn.number)).where(ChatTurn.conversation_id == conversation.id)) or 0
    )
    view = turn_view(turn, latest)
    if turn.status != "failed":
        return view
    if not view.retryable:
        raise HTTPException(
            409,
            "This turn cannot be retried after a later message or an exhausted budget. Send a new message.",
        )
    turn.status, turn.error, turn.updated_at = "queued", None, now()
    turn.job_id = enqueue(db, turn.id)
    db.commit()
    return turn_view(turn, latest)


def apply_context(db: Session, conversation: Conversation, owner: UUID, turn_id: UUID) -> UUID:
    from backend.context.contracts import SaveInput
    from backend.context.service import save_revision
    from backend.responses import Reply
    from backend.storage import files

    turn = db.get(ChatTurn, turn_id)
    if turn is None or turn.conversation_id != conversation.id:
        raise HTTPException(404, "Message not found.")
    existing = db.get(ChatContextSelection, turn_id)
    if existing:
        return existing.revision_id
    if turn.status != "succeeded" or turn.inputs.get("mode") != "context" or not turn.reply:
        raise HTTPException(409, "This turn has no completed context proposal.")
    reply = Reply.model_validate(turn.reply)
    bound = turn.inputs.get("context_revision_id")
    if not bound or (not reply.context_edits and reply.analysis_brief is None) or active_revision(db, conversation) != UUID(str(bound)):
        raise HTTPException(409, "This proposal is missing or its context changed. Discuss a fresh proposal.")
    if db.scalar(
        select(ChatTurn.id).where(
            ChatTurn.conversation_id == conversation.id, ChatTurn.status.in_(["queued", "running"])
        )
    ):
        raise HTTPException(409, "Wait for the active reply before applying context changes.")
    version = db.get(DatasetVersion, conversation.version_id)
    assert version is not None
    changes = [
        edit.model_copy(
            update={
                "reason": f"Applied chat proposal {turn.id}. User message: {turn.text}\n{edit.reason}"[:10000]
            }
        )
        for edit in reply.context_edits
    ]
    revision = save_revision(
        db,
        files(),
        version,
        owner,
        SaveInput(
            submission_id=uuid5(turn.id, "apply-context"),
            expected_revision_id=UUID(str(bound)),
            edits=changes,
            analysis_brief=reply.analysis_brief,
        ),
        commit=False,
        brief_source_text=f"Applied chat proposal {turn.id}. User message: {turn.text}",
    )
    db.add(ChatContextSelection(turn_id=turn.id, conversation_id=conversation.id, revision_id=revision.id))
    db.commit()
    return revision.id

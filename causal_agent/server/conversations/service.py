"""Bind an owned dataset version to an independent desk conversation."""

import hashlib

from sqlalchemy import select

from causal_agent.common import trace
from causal_agent.server.context.repository import Repository as ContextRepository
from causal_agent.server.conversations import schema as S
from causal_agent.server.ingestion.service import Missing, uid


def get(storage, owner, conversation):
    with storage.engine.connect() as db:
        row = db.execute(select(S.conversations).where(S.conversations.c.id == conversation, S.conversations.c.owner_id == owner)).mappings().first()
    if row is None:
        raise Missing("Conversation not found.")
    return dict(row)


def list_owned(storage, owner, dataset):
    storage.get(owner, dataset)
    with storage.engine.connect() as db:
        return [
            dict(row)
            for row in db.execute(
                select(S.conversations)
                .join(S.state, S.state.c.conversation_id == S.conversations.c.id)
                .where(S.conversations.c.owner_id == owner, S.conversations.c.dataset_id == dataset)
                .order_by(S.conversations.c.created_at.desc())
            ).mappings()
        ]


def create(storage, owner, dataset, manager, review_id, generation):
    context_repository = ContextRepository(storage, owner, dataset)
    revision = context_repository.finish_review(review_id, generation)
    review = context_repository.review(review_id)
    with storage.engine.connect() as db:
        existing = db.scalar(select(S.state.c.conversation_id).where(S.state.c.review_id == review_id, S.state.c.owner_id == owner))
    if existing:
        manager.bind(existing, storage, owner)
        return get(storage, owner, existing)
    info, stream = storage.download(owner, dataset)
    # The completed review is the creation idempotency key, including local input bytes.
    conversation = review_id
    root = manager.s.root
    csv = root / "data" / "conversations" / conversation / "input.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        content = stream.read(storage.settings.max_bytes + 1)
    finally:
        stream.close()
    if len(content) != info["size_bytes"] or hashlib.sha256(content).hexdigest() != info["sha256"]:
        raise OSError("Selected dataset checksum mismatch")
    if csv.exists():
        if hashlib.sha256(csv.read_bytes()).hexdigest() != info["sha256"]:
            raise OSError("Cached dataset checksum mismatch")
    else:
        trace.write(csv, content)
        csv.chmod(0o444)
    context = info["initial_context"] or ""
    with storage.engine.begin() as db:
        context_repository.lock(db)
        existing = db.scalar(select(S.state.c.conversation_id).where(S.state.c.review_id == review_id, S.state.c.owner_id == owner))
        if existing:
            manager.bind(existing, storage, owner)
            return get(storage, owner, existing)
        db.execute(
            S.conversations.insert().values(
                id=conversation, owner_id=owner, dataset_id=dataset, version_id=info["version_id"], title=info["title"], context=context
            )
        )
        welcome = (
            f"Welcome. Let’s explore “{review['question']}” using {info['title']}. "
            "Your reviewed context is saved for this chat, including anything you left unresolved. "
            "We can discuss a column or an artifact, create a visualization, or fast-track the remaining analysis requirements."
        )
        db.execute(
            S.state.insert().values(
                conversation_id=conversation,
                owner_id=owner,
                version_id=info["version_id"],
                review_id=review_id,
                revision_id=revision.id,
                question=review["question"],
                prompt=welcome,
            )
        )
        db.execute(
            S.messages.insert().values(
                id=uid(), owner_id=owner, conversation_id=conversation, operation_id=review_id, role="assistant", text=welcome, references=[]
            )
        )
    manager.bind(conversation, storage, owner)
    return get(storage, owner, conversation)

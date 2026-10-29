"""Transactions for chat selection, accepted commands and transcript publication."""

from uuid import uuid4

from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert

from causal_agent.memory.context.conversation import ChatCommand, GapQuestion
from causal_agent.memory.context.projection import validate_bindings
from causal_agent.memory.context.review import ContextEdit, KnowledgeStatement, RequirementBinding, RevisionConflict, SourceReference, apply_edits
from causal_agent.server.context.repository import Repository as ContextRepository
from causal_agent.server.context.repository import fingerprint
from causal_agent.server.conversations import schema as S
from causal_agent.server.conversations.publication import Repository as Artifacts
from causal_agent.server.conversations.service import get
from causal_agent.server.ingestion.service import Missing, uid


class Busy(ValueError):
    pass


class Repository:
    def __init__(self, storage, owner, conversation):
        self.storage, self.owner, self.conversation = storage, owner, conversation
        self.chat = get(storage, owner, conversation)
        self.context = ContextRepository(storage, owner, self.chat["dataset_id"])

    def state(self, db=None, lock=False):
        if db is None:
            with self.storage.engine.connect() as connection:
                return self.state(connection)
        query = select(S.state).where(S.state.c.conversation_id == self.conversation, S.state.c.owner_id == self.owner)
        row = db.execute(query.with_for_update() if lock else query).mappings().first()
        if row is None:
            raise Missing("Reviewed conversation not found.")
        return dict(row)

    def update(self, db, **values):
        db.execute(S.state.update().where(S.state.c.conversation_id == self.conversation, S.state.c.owner_id == self.owner).values(**values))

    def operation_error_is_input(self, operation_id):
        with self.storage.engine.connect() as db:
            return self.operation(db, operation_id)["body"]["kind"] in {"answer", "correct", "focus", "reuse", "refresh"}

    def operation(self, db, operation_id):
        row = (
            db.execute(
                select(S.commands).where(
                    S.commands.c.id == operation_id, S.commands.c.owner_id == self.owner, S.commands.c.conversation_id == self.conversation
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            raise Missing("Submission not found.")
        return dict(row)

    def reference(self, reference, revision_id):
        if reference.kind == "column":
            revision = self.context.read(revision_id)
            if reference.id not in {c.column_id for c in revision.knowledge.columns} or reference.element is not None:
                raise Missing("Column reference not found.")
        else:
            artifact = Artifacts(self.storage, self.owner, self.conversation).load(self.conversation, reference.id)
            if reference.element is not None and reference.element not in {e.id for e in artifact.elements}:
                raise Missing("Artifact element is not registered.")

    def append(self, db, operation_id, role, text, references=()):
        db.execute(
            insert(S.messages)
            .values(
                id=uid(),
                owner_id=self.owner,
                conversation_id=self.conversation,
                operation_id=operation_id,
                role=role,
                text=text,
                references=[r.model_dump() for r in references],
            )
            .on_conflict_do_nothing(index_elements=["conversation_id", "operation_id", "role"])
        )

    def accept(self, command: ChatCommand):
        # Lock ordering is always dataset version, then conversation.
        with self.storage.engine.begin() as db:
            self.context.lock(db)
            state = self.state(db, lock=True)
            existing = db.execute(select(S.commands).where(S.commands.c.id == command.submission_id)).mappings().first()
            digest = fingerprint(command.model_dump())
            if existing:
                if existing["owner_id"] != self.owner or existing["conversation_id"] != self.conversation:
                    raise Missing("Submission not found.")
                if existing["request_hash"] != digest:
                    raise ValueError("Submission ID already used with different content.")
                return dict(existing), False
            if state["status"] != "idle":
                raise Busy("Finish or explicitly resume the active turn first.")
            if command.revision_id != state["revision_id"]:
                raise RevisionConflict(state["revision_id"], [])
            if command.kind in {"correct", "answer"}:
                self.context.check_revision(db, command.revision_id)
            if command.kind == "correct":
                knowledge = self.context.decode(self.context.row(db, command.revision_id)).knowledge
                validate_bindings(apply_edits(knowledge, command.edits, SourceReference(kind="message", id=command.submission_id, text=command.text)))
            if command.kind == "answer":
                row = (
                    db.execute(
                        select(S.questions).where(
                            S.questions.c.id == command.question_id, S.questions.c.owner_id == self.owner, S.questions.c.conversation_id == self.conversation
                        )
                    )
                    .mappings()
                    .first()
                )
                if row is None or row["answer"] is not None or row["revision_id"] != command.revision_id or command.question_id not in state["pending"]:
                    raise ValueError("This context question is no longer awaiting an answer. Review the current questions.")
                question = GapQuestion.model_validate(row["body"])
                if command.option_id and command.option_id not in {o.id for o in question.options}:
                    raise ValueError("This option does not belong to the question.")
            if command.kind == "authorize":
                preparation = state["preparation"]
                if not preparation or preparation["id"] != command.snapshot_id or preparation["revision_id"] != command.revision_id:
                    raise ValueError("The analysis review is stale. Prepare the current context before authorizing a run.")
            for ref in command.references:
                if command.kind == "reuse":
                    artifact = Artifacts(self.storage, self.owner, self.conversation).available(ref.id)
                    if ref.element is not None and ref.element not in {e.id for e in artifact.elements}:
                        raise Missing("Artifact element is not registered.")
                else:
                    self.reference(ref, state["revision_id"])
            db.execute(
                S.commands.insert().values(
                    id=command.submission_id, owner_id=self.owner, conversation_id=self.conversation, body=command.model_dump(), request_hash=digest
                )
            )
            labels = {
                "fast_track": "Prepare analysis with fast-track.",
                "pause": "Pause fast-track.",
                "refresh": "Use the latest reviewed context in this chat.",
                "authorize": "Run the analysis shown in the review.",
                "focus": "Discuss the selected reference." if command.references else "Clear conversation focus.",
                "reuse": "Reuse and discuss this earlier artifact.",
            }
            text = command.text or labels.get(command.kind, "Apply the displayed context corrections.")
            if command.kind == "answer":
                answer = next((o.label for o in question.options if o.id == command.option_id), command.text)
                text = question.text + "\nMy answer: " + answer
            self.append(db, command.submission_id, "user", text, command.references)
            self.update(db, status="running", active_operation=command.submission_id, error=None)
            return self.operation(db, command.submission_id), True

    def correct(self, operation_id, edits, text):
        with self.storage.engine.begin() as db:
            self.context.lock(db)
            state = self.state(db, lock=True)
            command = ChatCommand.model_validate(self.operation(db, operation_id)["body"])
            if state["active_operation"] != operation_id:
                raise Busy("This operation is no longer active.")
            revision = self.context.commit_in(db, command.revision_id, operation_id, edits, SourceReference(kind="message", id=operation_id, text=text))
            self.update(db, revision_id=revision.id, preparation=None, pending=[])
            return revision

    def answer(self, operation_id):
        with self.storage.engine.begin() as db:
            self.context.lock(db)
            state = self.state(db, lock=True)
            command = ChatCommand.model_validate(self.operation(db, operation_id)["body"])
            row = (
                db.execute(
                    select(S.questions).where(
                        S.questions.c.id == command.question_id, S.questions.c.owner_id == self.owner, S.questions.c.conversation_id == self.conversation
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                raise Missing("Question not found.")
            if row["answer"] is not None:
                if row["answer"]["operation_id"] != operation_id:
                    raise ValueError("This question already has an answer.")
                return self.context.decode(self.context.row(db, state["revision_id"]))
            if row["revision_id"] != state["revision_id"] or command.question_id not in state["pending"]:
                raise RevisionConflict(state["revision_id"], [])
            question = GapQuestion.model_validate(row["body"])
            option = next((o for o in question.options if o.id == command.option_id), None)
            if command.option_id and option is None:
                raise ValueError("This option does not belong to the question.")
            value = option.value if option else command.text
            unknown = option is not None and option.unknown
            source = SourceReference(kind="message", id=operation_id, text=option.label if option else command.text)
            knowledge = self.context.decode(self.context.row(db, state["revision_id"])).knowledge
            previous = next((s for s in knowledge.statements if s.binding and s.binding.address == question.target and s.status != "withdrawn"), None)
            statement = KnowledgeStatement(
                id=str(uuid4()),
                scope="requirement",
                field=question.target,
                value=source.text,
                status="unresolved" if unknown else "reviewed",
                source=source,
                binding=RequirementBinding(address=question.target, value=value),
            )
            edits = [
                ContextEdit(
                    action="replace" if previous else "add",
                    target_id=previous.id if previous else None,
                    statement=statement,
                    reason="Answered the context requirement.",
                )
            ]
            revision = self.context.commit_in(db, command.revision_id, operation_id, edits, source)
            db.execute(
                S.questions.update().where(S.questions.c.id == question.id).values(answer={"operation_id": operation_id, "unknown": unknown, "value": value})
            )
            self.update(db, revision_id=revision.id, pending=[], preparation=None)
            return revision

    def controls(self, operation_id):
        with self.storage.engine.begin() as db:
            self.context.lock(db)
            state = self.state(db, lock=True)
            command = ChatCommand.model_validate(self.operation(db, operation_id)["body"])
            if state["active_operation"] != operation_id:
                raise Busy("This operation is no longer active.")
            if command.kind == "focus":
                self.update(db, focus=command.references[0].model_dump() if command.references else None)
                return "Conversation focus updated." if command.references else "Conversation focus cleared."
            if command.kind == "reuse":
                ref = command.references[0]
                Artifacts(self.storage, self.owner, self.conversation).reuse(db, ref.id, operation_id)
                self.update(db, focus=ref.model_dump())
                return "This artifact is now selected for discussion. Its original conversation and dataset version are preserved."
            if command.kind == "refresh":
                head = self.context.head(db)
                if head["id"] != command.revision_id:
                    self.update(db, revision_id=head["id"], preparation=None, pending=[])
                return "This chat now uses the latest reviewed dataset context."
            if command.kind in {"pause", "fast_track"}:
                self.update(db, fast_track=command.kind == "fast_track")
                return "Fast-track paused. Your answers are saved." if command.kind == "pause" else "Preparing the next unresolved requirements."
            raise ValueError("Not a chat control.")

    def finish(self, operation_id, text):
        with self.storage.engine.begin() as db:
            state = self.state(db, lock=True)
            operation = self.operation(db, operation_id)
            if operation["status"] == "completed":
                return
            if state["active_operation"] != operation_id:
                raise Busy("This operation is no longer active.")
            self.append(db, operation_id, "assistant", text)
            db.execute(
                S.commands.update().where(S.commands.c.id == operation_id).values(status="completed", result={**(operation["result"] or {}), "text": text})
            )
            self.update(db, status="idle", active_operation=None, prompt=text, error=None)

    def interrupt(self, operation_id, error):
        with self.storage.engine.begin() as db:
            state = self.state(db, lock=True)
            if state["active_operation"] == operation_id:
                self.update(db, status="interrupted", error=error)

    def messages(self, before=None, limit=50):
        if not 1 <= limit <= 100:
            raise ValueError("Message page size must be between 1 and 100.")
        with self.storage.engine.connect() as db:
            query = select(S.messages).where(S.messages.c.conversation_id == self.conversation, S.messages.c.owner_id == self.owner)
            if before:
                cursor = db.execute(query.where(S.messages.c.id == before)).mappings().first()
                if cursor is None:
                    raise Missing("Message cursor not found.")
                query = query.where(
                    or_(S.messages.c.created_at < cursor["created_at"], and_(S.messages.c.created_at == cursor["created_at"], S.messages.c.id < before))
                )
            return [dict(r) for r in db.execute(query.order_by(S.messages.c.created_at.desc(), S.messages.c.id.desc()).limit(limit)).mappings()]

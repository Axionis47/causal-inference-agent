"""Connect read contracts to the component that owns each record."""

from uuid import UUID

from sqlalchemy.orm import Session

from backend.analysis import discussion
from backend.chat.service import conversation_scope
from backend.handoffs import AnalysisBinding, AnalysisFile, AnalysisMaterial, ConversationScope


class Conversations:
    def read(self, db: Session, owner: UUID, identity: UUID, *, lock: bool = False) -> ConversationScope:
        return conversation_scope(db, owner, identity, lock=lock)


class Analyses:
    def capture(self, db: Session, conversation_id: UUID) -> dict[str, AnalysisBinding]:
        return discussion.capture(db, conversation_id)

    def views(self, db: Session, conversation_id: UUID, identities: list[UUID],
              bindings: dict[str, AnalysisBinding] | None) -> list[AnalysisMaterial]:
        return discussion.views(db, conversation_id, identities, bindings)

    def file(self, db: Session, conversation_id: UUID, material: AnalysisMaterial,
             name: str, *, expected_hash: str | None = None) -> AnalysisFile:
        return discussion.file(db, conversation_id, material, name, expected_hash=expected_hash)

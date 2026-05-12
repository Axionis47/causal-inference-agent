"""Internal read contracts. These values do not own or persist domain state."""

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict
from uuid import UUID

from sqlalchemy.orm import Session


@dataclass(frozen=True)
class ConversationScope:
    id: UUID
    version_id: UUID
    context_revision_id: UUID | None
    exploration_count: int


class ConversationScopeReader(Protocol):
    def read(self, db: Session, owner: UUID, identity: UUID, *, lock: bool = False) -> ConversationScope: ...


class AnalysisBinding(TypedDict):
    snapshot_hash: str | None
    result_hash: str | None
    status: str
    files: dict[str, str]


@dataclass(frozen=True)
class AnalysisMaterial:
    id: UUID
    question: str
    binding: AnalysisBinding
    snapshot: dict[str, Any] | None
    result: dict[str, Any] | None


@dataclass(frozen=True)
class AnalysisFile:
    raw: bytes
    sha256: str
    facts: dict[str, Any]


class AnalysisDiscussionReader(Protocol):
    def capture(self, db: Session, conversation_id: UUID) -> dict[str, AnalysisBinding]: ...
    def views(self, db: Session, conversation_id: UUID, identities: list[UUID],
              bindings: dict[str, AnalysisBinding] | None) -> list[AnalysisMaterial]: ...
    def file(self, db: Session, conversation_id: UUID, material: AnalysisMaterial,
             name: str, *, expected_hash: str | None = None) -> AnalysisFile: ...

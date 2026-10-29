"""Read projections of the same revision used by extraction, review and chat."""

from datetime import datetime
from typing import Literal

from causal_agent.memory.context.contracts import ColumnIdentity, Contract
from causal_agent.memory.context.review import KnowledgeStatement


class ContextColumnSummary(ColumnIdentity):
    statement_count: int
    relationship_count: int


class ContextSummary(Contract):
    revision_id: str
    dataset_id: str
    version_id: str
    revision: int
    created_at: datetime
    source_id: str | None
    source_checksum: str
    csv_checksum: str
    column_index_fingerprint: str
    schema_version: str
    extractor_version: str
    model: str
    source_present: bool
    columns: list[ContextColumnSummary]


class ContextView(Contract):
    status: Literal["not_prepared", "prepared"]
    pack: ContextSummary | None = None


class ContextDatasetView(Contract):
    pack: ContextSummary
    source_text: str
    dataset: list[KnowledgeStatement]
    concepts: list[KnowledgeStatement]
    # Relations between unmeasured concepts/events are visible here too.
    relationships: list[KnowledgeStatement]


class ContextColumnView(Contract):
    revision_id: str
    version_id: str
    column: ColumnIdentity
    statements: list[KnowledgeStatement]
    dataset: list[KnowledgeStatement]
    relationships: list[KnowledgeStatement]
    concepts: list[KnowledgeStatement]

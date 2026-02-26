"""The original context vocabulary, keyed for exact revision retrieval."""

from datetime import datetime
from typing import Annotated, Literal, get_args
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

DatasetField = Literal[
    "observations", "population", "organization", "study_process", "measurement", "question", "qualifications"
]
ColumnField = Literal[
    "definition",
    "concept",
    "entity",
    "units",
    "codes",
    "valid_ranges",
    "missing_codes",
    "method",
    "measurement_source",
    "determined_by",
    "collection_time",
    "reference_period",
    "construction",
    "derivation",
    "missingness",
    "qualifications",
]
RelationshipKind = Literal[
    "measurement",
    "derivation",
    "temporal_order",
    "membership",
    "assignment_rule",
    "selection_rule",
    "association",
    "causal_influence",
    "no_causal_influence",
    "unclassified",
]
DATASET_FIELDS = get_args(DatasetField)
COLUMN_FIELDS = get_args(ColumnField)
Status = Literal["proposed", "reviewed", "unresolved", "withdrawn"]
RevisionState = Literal["draft", "reviewed"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(Contract):
    source_id: UUID
    source_line: int = Field(ge=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1, max_length=100_000)


class SourceReference(Contract):
    kind: Literal["extraction", "review", "message"]
    id: UUID
    author_id: UUID
    text: str = Field(default="", max_length=10_000)


class Statement(Contract):
    value: str = Field(min_length=1, max_length=10_000)
    basis: Literal["source_statement", "interpretation"] = "source_statement"
    citations: list[Citation] = Field(default_factory=list, max_length=20)
    qualifications: str = Field(default="", max_length=10_000)
    status: Status = "proposed"
    source: SourceReference
    review_source: SourceReference | None = None
    replaces: str | None = None
    reason: str = Field(default="", max_length=10_000)
    conflicts_with: list[Annotated[str, Field(min_length=1, max_length=250)]] | None = Field(default=None, max_length=30, exclude_if=lambda value: value is None, description="Explicitly declared conflicting statement keys. This records a reviewed conflict without inferring contradiction from prose.")


class DatasetStatement(Statement):
    column_ids: list[str] = Field(default_factory=list, max_length=200)


class DatasetContext(Contract):
    observations: dict[str, DatasetStatement] = Field(default_factory=dict)
    population: dict[str, DatasetStatement] = Field(default_factory=dict)
    organization: dict[str, DatasetStatement] = Field(default_factory=dict)
    study_process: dict[str, DatasetStatement] = Field(default_factory=dict)
    measurement: dict[str, DatasetStatement] = Field(default_factory=dict)
    question: dict[str, DatasetStatement] = Field(default_factory=dict)
    qualifications: dict[str, DatasetStatement] = Field(default_factory=dict)


class ColumnIdentity(Contract):
    column_id: str
    position: int = Field(ge=0)
    name: str


class ColumnContext(Contract):
    identity: ColumnIdentity
    definition: dict[str, Statement] = Field(default_factory=dict)
    concept: dict[str, Statement] = Field(default_factory=dict)
    entity: dict[str, Statement] = Field(default_factory=dict)
    units: dict[str, Statement] = Field(default_factory=dict)
    codes: dict[str, Statement] = Field(default_factory=dict)
    valid_ranges: dict[str, Statement] = Field(default_factory=dict)
    missing_codes: dict[str, Statement] = Field(default_factory=dict)
    method: dict[str, Statement] = Field(default_factory=dict)
    measurement_source: dict[str, Statement] = Field(default_factory=dict)
    determined_by: dict[str, Statement] = Field(default_factory=dict)
    collection_time: dict[str, Statement] = Field(default_factory=dict)
    reference_period: dict[str, Statement] = Field(default_factory=dict)
    construction: dict[str, Statement] = Field(default_factory=dict)
    derivation: dict[str, Statement] = Field(default_factory=dict)
    missingness: dict[str, Statement] = Field(default_factory=dict)
    qualifications: dict[str, Statement] = Field(default_factory=dict)


class Concept(Statement):
    kind: Literal["concept", "event"]


class Participant(Contract):
    kind: Literal["column", "concept", "event"]
    id: str
    role: Literal["participant", "from", "to", "intermediate", "condition"] = "participant"


class Relationship(Statement):
    # The description carries the claim. Classification is optional metadata.
    kind: RelationshipKind | None = None
    participants: list[Participant] = Field(min_length=2, max_length=200)
    timing: str = Field(default="", max_length=10_000)
    conditions: str = Field(default="", max_length=10_000)
    directness: Literal["direct", "indirect", "unspecified"] = "unspecified"
    absence_scope: Literal["direct_effect", "any_pathway", "unspecified"] = "unspecified"


class BriefClaim(Contract):
    source_keys: list[Annotated[str, Field(min_length=1, max_length=250)]] = Field(default_factory=list, max_length=30, description="Exact statement keys in the applied context. Drafts may use edit:<zero-based index> for an add/replace in this same proposal. No copied or invented factual claim.")


class BriefVariable(Contract):
    column_ids: list[str] = Field(default_factory=list, max_length=30)
    meaning: BriefClaim = Field(default_factory=BriefClaim)


class BriefTreatment(BriefVariable):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$", description="Stable identity for this intervention variable, not an arm or display-name inference.")
    change: BriefClaim = Field(default_factory=BriefClaim)


class BriefDraft(Contract):
    question: str = Field(default="", max_length=8000, description="The finalized user-request text used to launch the analysis. This is the only launch-question source.")
    target: BriefVariable | None = None
    treatments: list[BriefTreatment] = Field(default_factory=list, max_length=20)
    comparison: BriefClaim = Field(default_factory=BriefClaim)
    population: BriefClaim = Field(default_factory=BriefClaim)
    unit: BriefClaim = Field(default_factory=BriefClaim)
    timing: BriefClaim = Field(default_factory=BriefClaim)
    assumptions: list[BriefClaim] = Field(default_factory=list, max_length=30, description="References to existing conditional, qualified or interpreted context statements. These do not supply missing essential meanings.")
    optional_unknowns: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(default_factory=list, max_length=30, description="Focused unresolved optional questions, not asserted facts. Unknown optional scientific roles do not block preparation.")


class AnalysisBrief(BriefDraft):
    source: SourceReference


def brief_source_keys(brief: BriefDraft) -> set[str]:
    claims = [brief.comparison, brief.population, brief.unit, brief.timing, *brief.assumptions]
    if brief.target:
        claims.append(brief.target.meaning)
    for treatment in brief.treatments:
        claims.extend([treatment.meaning, treatment.change])
    return {key for claim in claims for key in claim.source_keys}


class ContextContent(Contract):
    dataset: DatasetContext = Field(default_factory=DatasetContext)
    columns: dict[str, ColumnContext] = Field(default_factory=dict)
    concepts: dict[str, Concept] = Field(default_factory=dict)
    relationships: dict[str, Relationship] = Field(default_factory=dict)
    analysis_brief: AnalysisBrief | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def references(self) -> "ContextContent":
        entries = statement_entries(self)
        ids = [key.rsplit(".", 1)[-1] for key, _ in entries]
        if len(ids) != len(set(ids)) or len(ids) > 2000:
            raise ValueError("Statement identities must be unique; at most 2,000 statements are allowed.")
        if any(not identity or "." in identity for identity in [*ids, *self.columns]):
            raise ValueError("Identities must be nonempty and cannot contain periods.")
        for cid, column in self.columns.items():
            if column.identity.column_id != cid:
                raise ValueError("Column identity does not match its key.")
        for _, statement in entries:
            if (
                isinstance(statement, DatasetStatement)
                and not set(statement.column_ids) <= self.columns.keys()
            ):
                raise ValueError("Dataset statement names an unknown column.")
            if statement.source.kind == "extraction" and not statement.citations:
                raise ValueError("An extracted statement requires source evidence.")
            if statement.replaces and statement.replaces not in ids:
                raise ValueError("Replacement must retain its predecessor.")
        keys = {key for key, _ in entries}
        for key, statement in entries:
            conflicts = statement.conflicts_with or []
            if len(conflicts) != len(set(conflicts)) or key in conflicts or not set(conflicts) <= keys:
                raise ValueError("Declared conflicts require distinct exact other statement keys in this context.")
        for relation in self.relationships.values():
            if (
                relation.kind not in {None, "unclassified", "no_causal_influence"}
                and relation.absence_scope != "unspecified"
            ):
                raise ValueError("Absence scope only applies to no_causal_influence relationships.")
            if len({p.id for p in relation.participants}) < 2:
                raise ValueError("A relationship needs distinct participants.")
            for participant in relation.participants:
                if participant.kind == "column":
                    valid = participant.id in self.columns
                else:
                    concept = self.concepts.get(participant.id)
                    valid = concept is not None and concept.kind == participant.kind
                if not valid:
                    raise ValueError("Relationship names an unknown participant or the wrong kind.")
        return self


def statement_entries(content: ContextContent) -> list[tuple[str, Statement]]:
    result: list[tuple[str, Statement]] = []
    for field in DATASET_FIELDS:
        result.extend((f"dataset.{field}.{sid}", s) for sid, s in getattr(content.dataset, field).items())
    for cid, column in content.columns.items():
        for field in COLUMN_FIELDS:
            result.extend((f"columns.{cid}.{field}.{sid}", s) for sid, s in getattr(column, field).items())
    result.extend((f"concepts.{sid}", s) for sid, s in content.concepts.items())
    result.extend((f"relationships.{sid}", s) for sid, s in content.relationships.items())
    return result


def empty_content(version_id: UUID, names: list[str]) -> ContextContent:
    columns = {}
    for position, name in enumerate(names):
        cid = uuid5(version_id, f"column:{position}").hex
        columns[cid] = ColumnContext(identity=ColumnIdentity(column_id=cid, position=position, name=name))
    return ContextContent(columns=columns)


class SourceView(Contract):
    model_config = ConfigDict(from_attributes=True, extra="forbid")
    id: UUID
    version_id: UUID
    title: str
    filename: str
    kind: Literal["file", "note"]
    sha256: str
    size_bytes: int
    character_count: int
    created_at: datetime


class SourceText(Contract):
    source: SourceView
    text: str


class RevisionSummary(Contract):
    model_config = ConfigDict(from_attributes=True, extra="forbid")
    id: UUID
    pack_id: UUID
    parent_id: UUID | None
    number: int
    state: RevisionState
    author_id: UUID
    created_at: datetime


class RevisionView(RevisionSummary):
    version_id: UUID
    schema_version: str
    content: ContextContent
    sources: dict[str, SourceView]


class PreparationView(Contract):
    model_config = ConfigDict(from_attributes=True, extra="forbid")
    id: UUID
    base_revision_id: UUID | None
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: str
    calls: int
    error: str | None
    result_revision_id: UUID | None
    created_at: datetime
    updated_at: datetime
    trace_id: UUID


class WorkspaceView(Contract):
    version_id: UUID
    current_revision_id: UUID | None
    sources: list[SourceView]
    revisions: list[RevisionSummary]
    preparations: list[PreparationView]
    dataset_fields: list[str] = list(DATASET_FIELDS)
    column_fields: list[str] = list(COLUMN_FIELDS)
    relationship_kinds: list[str] = list(get_args(RelationshipKind))
    columns: list[ColumnIdentity]


class PrepareInput(Contract):
    submission_id: UUID
    expected_revision_id: UUID | None
    source_ids: list[UUID] = Field(min_length=1, max_length=10)


class EditStatement(Contract):
    """Semantic inputs only. Identity, authorship and status are server assigned."""

    value: str = Field(min_length=1, max_length=10_000)
    basis: Literal["source_statement", "interpretation"] = "source_statement"
    qualifications: str = Field(default="", max_length=10_000)
    citations: list[Citation] = Field(default_factory=list, max_length=20)
    column_ids: list[str] = Field(default_factory=list, max_length=200)
    kind: str | None = None
    participants: list[Participant] = Field(default_factory=list, max_length=200)
    timing: str = Field(default="", max_length=10_000)
    conditions: str = Field(default="", max_length=10_000)
    directness: Literal["direct", "indirect", "unspecified"] = "unspecified"
    absence_scope: Literal["direct_effect", "any_pathway", "unspecified"] = "unspecified"
    conflicts_with: list[Annotated[str, Field(min_length=1, max_length=250)]] | None = Field(default=None, max_length=30, exclude_if=lambda value: value is None, description="Exact conflict statement keys or edit:<index> in this save. Omitted on replacement preserves existing conflicts; explicit [] records reviewed resolution.")


class ContextEdit(Contract):
    action: Literal["add", "replace", "review", "unresolved", "withdraw"]
    key: str = Field(
        min_length=1,
        max_length=250,
        description="For add, use an existing COLLECTION key such as dataset.qualifications, columns.<exact-column-id>.definition, concepts or relationships. Never append an invented statement ID; the server assigns it. For every other action, select the exact existing full statement key including its ID.",
    )
    statement: EditStatement | None = None
    reason: str = Field(default="", max_length=10_000)


class SaveInput(Contract):
    submission_id: UUID
    expected_revision_id: UUID
    state: RevisionState = "draft"
    edits: list[ContextEdit] = Field(default_factory=list, max_length=2000)
    leave_remaining_unresolved: bool = False
    analysis_brief: BriefDraft | None = Field(default=None, exclude_if=lambda value: value is None)


class RetrievedContext(Contract):
    revision_id: UUID
    version_id: UUID
    requested_keys: list[str]
    included_keys: list[str]
    # A typed projection preserves empty keyed fields and all statement metadata.
    content: ContextContent
    sources: dict[str, SourceView]
    # Participant -> other participant -> shared relationship IDs in this read.
    # Derived references only; multi-party descriptions remain stored once.
    related_to: dict[str, dict[str, list[str]]] = Field(default_factory=dict)

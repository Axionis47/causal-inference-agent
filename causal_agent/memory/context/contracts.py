"""One wire and persistence contract for source-grounded context."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DatasetField = Literal["observations", "population", "organization", "study_process", "measurement", "question", "qualifications"]
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


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(Contract):
    # Python Unicode code-point offsets, not UTF-16 offsets. The UI uses Array.from.
    source_line: int = Field(ge=1)
    start: int = Field(default=0, ge=0)
    end: int = Field(default=1, gt=0)
    quote: str = Field(min_length=1, max_length=100000)


class Statement(Contract):
    id: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=10000)
    basis: Literal["source_statement", "interpretation"]
    citations: list[Citation] = Field(min_length=1, max_length=20)
    qualifications: str = Field(default="", max_length=10000)


class DatasetStatement(Statement):
    field: DatasetField
    # Empty means study-wide. Otherwise only these columns inherit this statement.
    column_ids: list[str] = Field(default_factory=list, max_length=200)


class ColumnStatement(Statement):
    field: ColumnField


class ColumnIdentity(Contract):
    column_id: str
    position: int = Field(ge=0)
    name: str
    key: str


class ColumnContext(Contract):
    column_id: str
    statements: list[ColumnStatement] = Field(default_factory=list, max_length=100)


class Concept(Statement):
    kind: Literal["concept", "event"]


class Participant(Contract):
    kind: Literal["column", "concept", "event"]
    id: str
    role: Literal["participant", "from", "to", "intermediate", "condition"] = "participant"


class Relationship(Statement):
    kind: RelationshipKind
    participants: list[Participant] = Field(min_length=2, max_length=200)
    timing: str = Field(default="", max_length=10000)
    conditions: str = Field(default="", max_length=10000)
    directness: Literal["direct", "indirect", "unspecified"] = "unspecified"
    absence_scope: Literal["direct_effect", "any_pathway", "unspecified"] = "unspecified"


class ModelContract(Contract):
    """Provider-compatible schema, with full local validation retained."""

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        # Vertex rejects combinatorial bounds on these nested model schemas.
        # Keep the typed shape/enums in the provider schema; Pydantic and publish
        # still enforce every declared bound locally on the returned record.
        schema = super().model_json_schema(*args, **kwargs)

        def simplify(value):
            if isinstance(value, dict):
                return {
                    k: simplify(v)
                    for k, v in value.items()
                    if k not in {"maxItems", "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum", "additionalProperties"}
                }
            if isinstance(value, list):
                return [simplify(v) for v in value]
            return value

        return simplify(schema)


class DraftContext(ModelContract):
    dataset: list[DatasetStatement] = Field(default_factory=list, max_length=200)
    concepts: list[Concept] = Field(default_factory=list, max_length=200)
    columns: list[ColumnContext] = Field(default_factory=list, max_length=200)
    relationships: list[Relationship] = Field(default_factory=list, max_length=1000)


class Correction(Contract):
    # Empty statement_ids means relevant source material was omitted entirely.
    statement_ids: list[str] = Field(default_factory=list, max_length=200)
    source_lines: list[int] = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=10000)


class Assessment(ModelContract):
    # This reviews extraction, never scientific readiness. Unresolved material stays cited.
    corrections: list[Correction] = Field(default_factory=list, max_length=100)
    notes: list[str] = Field(default_factory=list, max_length=100)


class ExtractedContext(Contract):
    content: DraftContext
    assessment: Assessment
    model_calls: int = Field(ge=0, le=4)

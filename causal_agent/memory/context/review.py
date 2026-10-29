"""Reviewed knowledge and explicit edits, independent of HTTP and persistence.

Extracted statements remain proposals. Reviewing a column never confirms a
relationship or a causal requirement. Every replacement retains its predecessor.
"""

from typing import Any, Literal, Protocol, get_args

from pydantic import Field, model_validator

from causal_agent.memory.context.contracts import Citation, ColumnField, ColumnIdentity, Contract, DatasetField, DraftContext, Participant, RelationshipKind


class RequirementBinding(Contract):
    """A catalogue field bound to this statement, never a separately writable claim."""

    address: str
    value: Any = None


class SourceReference(Contract):
    kind: Literal["extraction", "review", "message"]
    id: str = Field(min_length=1, max_length=100)
    text: str = Field(default="", max_length=20000)


class KnowledgeStatement(Contract):
    id: str = Field(min_length=1, max_length=100)
    scope: Literal["dataset", "column", "concept", "relationship", "requirement"]
    field: str = Field(min_length=1, max_length=80)
    column_ids: list[str] = Field(default_factory=list, max_length=200)
    value: str = Field(min_length=1, max_length=10000)
    basis: Literal["source_statement", "interpretation"] = "source_statement"
    qualifications: str = Field(default="", max_length=10000)
    citations: list[Citation] = Field(default_factory=list, max_length=20)
    participants: list[Participant] = Field(default_factory=list, max_length=200)
    timing: str = Field(default="", max_length=10000)
    conditions: str = Field(default="", max_length=10000)
    directness: Literal["direct", "indirect", "unspecified"] = "unspecified"
    absence_scope: Literal["direct_effect", "any_pathway", "unspecified"] = "unspecified"
    status: Literal["proposed", "reviewed", "unresolved", "withdrawn"] = "proposed"
    source: SourceReference
    review_source: SourceReference | None = None
    replaces: str | None = None
    reason: str = Field(default="", max_length=10000)
    binding: RequirementBinding | None = None


class ColumnDisposition(Contract):
    column_id: str
    status: Literal["reviewed", "unresolved"]
    reason: str = Field(default="", max_length=10000)


class Knowledge(Contract):
    columns: list[ColumnIdentity] = Field(max_length=200)
    statements: list[KnowledgeStatement] = Field(default_factory=list, max_length=10000)
    dispositions: list[ColumnDisposition] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def identities(self):
        columns = {c.column_id for c in self.columns}
        ids = {s.id for s in self.statements}
        if len(columns) != len(self.columns) or len(ids) != len(self.statements):
            raise ValueError("Context identities must be unique.")
        if len({d.column_id for d in self.dispositions}) != len(self.dispositions):
            raise ValueError("Each column has one review disposition.")
        if any(d.column_id not in columns for d in self.dispositions):
            raise ValueError("Unknown column disposition.")
        concepts = {s.id for s in self.statements if s.scope == "concept"}
        bindings = set()
        for s in self.statements:
            fields = {
                "dataset": get_args(DatasetField),
                "column": get_args(ColumnField),
                "relationship": get_args(RelationshipKind),
                "concept": ("concept", "event"),
            }
            if s.scope in fields and s.field not in fields[s.scope]:
                raise ValueError("Statement field is outside the fixed context catalogue.")
            if s.scope == "requirement" and s.binding is None:
                raise ValueError("An analysis requirement must bind a catalogue field.")
            if s.binding and s.status != "withdrawn":
                if s.binding.address in bindings:
                    raise ValueError("A requirement has more than one current statement.")
                bindings.add(s.binding.address)
            if not set(s.column_ids) <= columns:
                raise ValueError("Statement refers to an unknown column.")
            if s.scope == "column" and len(s.column_ids) != 1:
                raise ValueError("A column statement must name one column.")
            if s.scope == "relationship" and len(s.participants) < 2:
                raise ValueError("A relationship needs at least two participants.")
            for p in s.participants:
                if p.id not in (columns if p.kind == "column" else concepts):
                    raise ValueError("Relationship refers to an unknown participant.")
        return self

    def for_column(self, column_id: str) -> list[KnowledgeStatement]:
        if column_id not in {c.column_id for c in self.columns}:
            raise ValueError("Unknown column.")
        return [
            s
            for s in self.statements
            if s.status != "withdrawn"
            and (
                column_id in s.column_ids
                or any(p.kind == "column" and p.id == column_id for p in s.participants)
                or (s.scope == "dataset" and not s.column_ids)
            )
        ]


class ContextEdit(Contract):
    action: Literal["add", "replace", "retire", "review", "unresolved"]
    target_id: str | None = None
    statement: KnowledgeStatement | None = None
    reason: str = Field(default="", max_length=10000)


class Revision(Contract):
    id: str
    version_id: str
    source_pack_id: str
    parent_id: str | None
    number: int
    knowledge: Knowledge


class RevisionConflict(ValueError):
    def __init__(self, current_id: str, changed_ids: list[str]):
        super().__init__("Context changed. Review the newer revision before applying these edits.")
        self.current_id = current_id
        self.changed_ids = changed_ids


class ContextRepository(Protocol):
    """The desk supplies intent; the repository checks selection and commits once."""

    def read(self, revision_id: str) -> Revision: ...

    def commit(self, expected_id: str, operation_id: str, edits: list[ContextEdit], source: SourceReference) -> Revision: ...


def extracted(content: DraftContext, columns: list[ColumnIdentity], pack_id: str) -> Knowledge:
    statements = []
    source = SourceReference(kind="extraction", id=pack_id)
    groups: list[tuple[Any, list[Any], str | None]] = [
        ("dataset", content.dataset, None),
        ("concept", content.concepts, None),
        ("relationship", content.relationships, None),
    ]
    groups.extend(("column", c.statements, c.column_id) for c in content.columns)
    for scope, items, column_id in groups:
        for item in items:
            data = item.model_dump(exclude={"kind", "field"})
            data["field"] = getattr(item, "field", getattr(item, "kind", ""))
            if column_id:
                data["column_ids"] = [column_id]
            statements.append(KnowledgeStatement(scope=scope, source=source, **data))
    return Knowledge(columns=columns, statements=statements)


def apply_edits(knowledge: Knowledge, edits: list[ContextEdit], source: SourceReference) -> Knowledge:
    """Atomic pure operation. Only an explicit human action may alter knowledge."""
    if source.kind == "extraction":
        raise ValueError("Extraction cannot review or correct knowledge.")
    result = knowledge.model_copy(deep=True)
    statements = {s.id: s for s in result.statements}
    touched = set()
    for edit in edits:
        if edit.target_id in touched:
            raise ValueError("Conflicting edits name the same statement.")
        if edit.target_id:
            touched.add(edit.target_id)
        previous = statements.get(edit.target_id or "")
        if edit.action != "add" and (previous is None or previous.status == "withdrawn"):
            raise ValueError("The selected statement is not current knowledge.")
        if edit.action in {"retire", "replace"} and not edit.reason.strip():
            raise ValueError("Withdrawing or replacing knowledge needs a reason.")
        if edit.action in {"add", "replace"}:
            if edit.statement is None or edit.statement.id in statements:
                raise ValueError("A new statement needs a fresh identity.")
            new = edit.statement.model_copy(deep=True)
            new.source, new.status = source, "unresolved" if new.status == "unresolved" else "reviewed"
            new.review_source = None
            new.replaces = previous.id if previous else None
            new.reason = edit.reason
            result.statements.append(new)
            statements[new.id] = new
        elif edit.statement is not None:
            raise ValueError("This action does not accept a replacement statement.")
        if previous:
            previous.status = "withdrawn" if edit.action in {"retire", "replace"} else ("reviewed" if edit.action == "review" else "unresolved")
            # The original source and text remain unchanged in this and earlier revisions.
            previous.reason = edit.reason
            previous.review_source = source
    old_kind = next(
        (s.binding.value for s in knowledge.statements if s.binding and s.binding.address == "claim:assignment.kind" and s.status != "withdrawn"), None
    )
    new_kind = next(
        (s.binding.value for s in result.statements if s.binding and s.binding.address == "claim:assignment.kind" and s.status != "withdrawn"), None
    )
    if old_kind is not None and new_kind != old_kind:
        from causal_agent.memory.ops import KIND_DEPENDENTS

        explicit = {
            s.binding.address for s in result.statements if s.binding and (s.id in touched or any(e.statement and e.statement.id == s.id for e in edits))
        }
        for statement in result.statements:
            if (
                statement.binding
                and statement.binding.address in {"claim:assignment." + field for field in KIND_DEPENDENTS}
                and statement.status != "withdrawn"
                and statement.binding.address not in explicit
            ):
                statement.status = "proposed"
                statement.reason = "The assignment kind changed. Confirm this dependent statement for the new assignment before analysis."
    return Knowledge.model_validate(result.model_dump())


def complete_review(knowledge: Knowledge, dispositions: list[ColumnDisposition]) -> Knowledge:
    if {d.column_id for d in dispositions} != {c.column_id for c in knowledge.columns}:
        raise ValueError("Review every column or explicitly leave it unresolved.")
    result = knowledge.model_copy(deep=True)
    result.dispositions = dispositions
    return Knowledge.model_validate(result.model_dump())


def validate_sources(knowledge: Knowledge, source_text: str) -> None:
    """A human correction may add knowledge but cannot forge an original quotation."""
    from bisect import bisect_right

    starts, offset = [], 0
    for line in source_text.splitlines(keepends=True):
        starts.append(offset)
        offset += len(line)
    for statement in knowledge.statements:
        for citation in statement.citations:
            if (
                citation.end <= citation.start
                or citation.end > len(source_text)
                or source_text[citation.start : citation.end] != citation.quote
                or bisect_right(starts, citation.start) != citation.source_line
            ):
                raise ValueError("A context citation does not match the immutable original account.")

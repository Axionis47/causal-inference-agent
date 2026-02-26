"""Pure edit, evidence validation, and dependency-preserving keyed reads."""

from collections.abc import Mapping
from typing import Any
from uuid import UUID, uuid5

from backend.context.contracts import (
    COLUMN_FIELDS,
    DATASET_FIELDS,
    AnalysisBrief,
    BriefDraft,
    BriefVariable,
    Concept,
    ContextContent,
    ContextEdit,
    DatasetStatement,
    Relationship,
    SourceReference,
    Statement,
    Status,
    brief_source_keys,
    statement_entries,
)


def slot(content: ContextContent, key: str) -> dict[str, Any]:
    parts = key.split(".")
    if len(parts) == 2 and parts[0] == "dataset" and parts[1] in DATASET_FIELDS:
        return getattr(content.dataset, parts[1])  # type: ignore[no-any-return]
    if (
        len(parts) == 3
        and parts[0] == "columns"
        and parts[1] in content.columns
        and parts[2] in COLUMN_FIELDS
    ):
        return getattr(content.columns[parts[1]], parts[2])  # type: ignore[no-any-return]
    if key == "concepts":
        return content.concepts
    if key == "relationships":
        return content.relationships
    raise ValueError(f"Unknown context field: {key}")


def validate_evidence(content: ContextContent, sources: Mapping[str, str]) -> None:
    for _, statement in statement_entries(content):
        for cite in statement.citations:
            source = sources.get(str(cite.source_id))
            if source is None:
                raise ValueError("Citation names a source outside this revision.")
            lines = source.splitlines(keepends=True)
            if cite.source_line > len(lines):
                raise ValueError("Citation line does not exist.")
            line_start = sum(len(line) for line in lines[: cite.source_line - 1])
            line_end = line_start + len(lines[cite.source_line - 1])
            if (
                not (line_start <= cite.start < cite.end <= line_end)
                or source[cite.start : cite.end] != cite.quote
            ):
                raise ValueError("Citation does not match the exact original source passage.")


def _applied_key(key: str, edits: list[ContextEdit], revision: UUID) -> str:
    if not key.startswith("edit:"):
        return key
    suffix = key.removeprefix("edit:")
    if not suffix.isdecimal() or str(int(suffix)) != suffix or int(suffix) >= len(edits):
        raise ValueError("Edit references must select an exact zero-based proposal index.")
    index = int(suffix)
    edit = edits[index]
    if edit.action not in {"add", "replace"}:
        raise ValueError("An edit reference must select a newly added or replaced statement.")
    collection = edit.key if edit.action == "add" else edit.key.rsplit(".", 1)[0]
    return collection + "." + uuid5(revision, f"statement:{index}").hex


def apply_edits(
    content: ContextContent, edits: list[ContextEdit], author: UUID, revision: UUID
) -> ContextContent:
    result = content.model_copy(deep=True)
    entries = dict(statement_entries(result))
    touched: set[str] = set()
    for index, edit in enumerate(edits):
        if edit.action != "add":
            previous = entries.get(edit.key)
            if previous is None or previous.status == "withdrawn":
                raise ValueError("The selected statement is not current context.")
            if edit.key in touched:
                raise ValueError("Only one action per statement can be saved at a time.")
            touched.add(edit.key)
        else:
            previous = None
        if edit.action in {"replace", "withdraw"} and not edit.reason.strip():
            raise ValueError("Explain why the statement is being replaced or withdrawn.")
        source = SourceReference(kind="review", id=revision, author_id=author, text=edit.reason)
        if edit.action in {"add", "replace"}:
            if edit.statement is None:
                raise ValueError("Supply a statement to add or replace.")
            target = edit.key if edit.action == "add" else edit.key.rsplit(".", 1)[0]
            try:
                container = slot(result, target)
            except ValueError as error:
                raise ValueError(
                    f"{error}. For action=add select an existing collection such as dataset.qualifications, "
                    "without a statement ID or invented suffix. New statement identities are server assigned."
                ) from error
            sid = uuid5(revision, f"statement:{index}").hex
            values = edit.statement.model_dump()
            common = {key: values[key] for key in ("value", "basis", "citations", "qualifications")}
            common.update(
                source=source,
                status="reviewed",
                reason=edit.reason,
                replaces=edit.key.rsplit(".", 1)[-1] if previous else None,
            )
            conflicts = edit.statement.conflicts_with
            if conflicts is None and previous:
                inherited = set(previous.conflicts_with or [])
                inherited.update(key for key, record in entries.items() if record.status != "withdrawn"
                                 and edit.key in (record.conflicts_with or []))
                conflicts = sorted(inherited) or None
            common["conflicts_with"] = conflicts
            new: Statement
            if target.startswith("dataset."):
                new = DatasetStatement(**common, column_ids=edit.statement.column_ids)
            elif target == "concepts":
                new = Concept.model_validate({**common, "kind": edit.statement.kind})
            elif target == "relationships":
                new = Relationship(
                    **common,
                    **{
                        key: values[key]
                        for key in (
                            "kind",
                            "participants",
                            "timing",
                            "conditions",
                            "directness",
                            "absence_scope",
                        )
                    },
                )
            else:
                new = Statement(**common)
            container[sid] = new
        elif edit.statement is not None:
            raise ValueError("This action does not accept replacement content.")
        if previous:
            status: Status = (
                "withdrawn"
                if edit.action in {"replace", "withdraw"}
                else ("reviewed" if edit.action == "review" else "unresolved")
            )
            if previous.status == status and previous.reason == edit.reason:
                continue
            previous.status = status
            previous.review_source = source
            previous.reason = edit.reason
    for _, statement in statement_entries(result):
        if statement.conflicts_with is not None:
            statement.conflicts_with = [_applied_key(key, edits, revision) for key in statement.conflicts_with]
    return ContextContent.model_validate(result.model_dump())


def related_to(content: ContextContent) -> dict[str, dict[str, list[str]]]:
    """Index co-participation, never assert pairwise causal edges or direction."""
    result: dict[str, dict[str, list[str]]] = {}
    for rid, relationship in content.relationships.items():
        if relationship.status == "withdrawn":
            continue
        participants = sorted({p.id for p in relationship.participants})
        for participant in participants:
            others = result.setdefault(participant, {})
            for other in participants:
                if other != participant:
                    others.setdefault(other, []).append(rid)
    return result


def apply_brief(
    content: ContextContent, draft: BriefDraft | None, edits: list[ContextEdit], author: UUID, revision: UUID,
    *, source_text: str = "Applied analysis brief.",
) -> ContextContent:
    """Resolve only exact same-transaction edit pointers; source records retain authority."""
    if draft is None:
        return content
    entries = dict(statement_entries(content))
    replacements: dict[str, str] = {}
    for key in brief_source_keys(draft):
        selected = _applied_key(key, edits, revision)
        if selected != key:
            replacements[key] = selected
        if selected not in entries or entries[selected].status == "withdrawn":
            raise ValueError("The analysis brief must reference current exact context statements, never missing or withdrawn records.")

    def resolve(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: [replacements.get(item, item) for item in child] if key == "source_keys" else resolve(child)
                    for key, child in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    result = content.model_copy(deep=True)
    result.analysis_brief = AnalysisBrief.model_validate({
        **resolve(draft.model_dump(mode="json")),
        "source": SourceReference(kind="review", id=revision, author_id=author, text=source_text[:10000]),
    })
    return result


def project(content: ContextContent, keys: list[str]) -> tuple[ContextContent, list[str], set[str]]:
    """Select roots, then semantic dependencies, without walking unrelated edges."""
    if not keys:
        keys = ["dataset", "columns", "concepts", "relationships"]
        if content.analysis_brief is not None:
            keys.append("analysis_brief")
    if len(keys) > 100:
        raise ValueError("Read at most 100 keys at a time.")
    entries = dict(statement_entries(content))
    chosen: set[str] = set()
    selected_columns: set[str] = set()
    included: set[str] = set()
    include_brief = False
    for key in keys:
        if key == "analysis_brief":
            include_brief = True
            included.add(key)
            if content.analysis_brief:
                chosen.update(brief_source_keys(content.analysis_brief) & entries.keys())
                bindings: list[BriefVariable] = [*content.analysis_brief.treatments]
                if content.analysis_brief.target:
                    bindings.append(content.analysis_brief.target)
                selected_columns.update(cid for binding in bindings for cid in binding.column_ids if cid in content.columns)
            continue
        if key.startswith("columns.") and key.endswith(".identity") and len(key.split(".")) == 3:
            cid = key.split(".")[1]
            if cid not in content.columns:
                raise ValueError(f"Unknown context key: {key}")
            selected_columns.add(cid)
            included.add(key)
            continue
        if key in {"dataset", "columns", "concepts", "relationships"}:
            chosen.update(k for k in entries if k.startswith(key + "."))
            if key == "columns":
                selected_columns.update(content.columns)
            included.add(key)
        elif key in entries:
            chosen.add(key)
        elif key.startswith("columns.") and len(key.split(".")) == 2:
            cid = key.split(".")[1]
            if cid not in content.columns:
                raise ValueError(f"Unknown context key: {key}")
            selected_columns.add(cid)
            chosen.update(k for k in entries if k.startswith(key + "."))
            for k, statement in entries.items():
                if isinstance(statement, DatasetStatement) and (
                    not statement.column_ids or cid in statement.column_ids
                ):
                    chosen.add(k)
                if isinstance(statement, Relationship) and any(
                    p.kind == "column" and p.id == cid for p in statement.participants
                ):
                    chosen.add(k)
            included.add(key)
        else:
            slot(content, key)  # Reject malformed or absent keys; valid empty fields remain readable.
            chosen.update(k for k in entries if k.startswith(key + "."))
            if key.startswith("columns."):
                selected_columns.add(key.split(".")[1])
            included.add(key)
    # Include reference targets and replacement history. A referenced withdrawn
    # concept remains withdrawn; it must not be revived by a read.
    by_id = {key.rsplit(".", 1)[-1]: key for key in entries}
    while True:
        before = (len(chosen), len(selected_columns))
        for key in list(chosen):
            statement = entries[key]
            chosen.update(statement.conflicts_with or [])
            chosen.update(other for other, record in entries.items() if key in (record.conflicts_with or []))
            if key.startswith("columns."):
                selected_columns.add(key.split(".")[1])
            if isinstance(statement, DatasetStatement):
                selected_columns.update(statement.column_ids)
            if isinstance(statement, Relationship):
                for participant in statement.participants:
                    if participant.kind == "column":
                        selected_columns.add(participant.id)
                        chosen.update(k for k in entries if k.startswith(f"columns.{participant.id}."))
                    else:
                        chosen.add(f"concepts.{participant.id}")
            if statement.replaces:
                chosen.add(by_id[statement.replaces])
        if before == (len(chosen), len(selected_columns)):
            break
    result = ContextContent(
        analysis_brief=content.analysis_brief.model_copy(deep=True) if include_brief and content.analysis_brief else None,
        columns={
            cid: type(content.columns[cid])(identity=content.columns[cid].identity)
            for cid in content.columns
            if cid in selected_columns
        }
    )
    for key in sorted(chosen):
        target, sid = key.rsplit(".", 1)
        slot(result, target)[sid] = entries[key].model_copy(deep=True)
    included.update(chosen)
    included.update(f"columns.{cid}.identity" for cid in selected_columns)
    source_ids = {str(cite.source_id) for key in chosen for cite in entries[key].citations}
    return ContextContent.model_validate(result.model_dump()), sorted(included), source_ids

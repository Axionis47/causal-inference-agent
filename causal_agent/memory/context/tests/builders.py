"""Shared source-grounded inputs and strict scripted model for Phase 4 tests."""

from causal_agent.memory.context.contracts import Assessment, Citation, ColumnContext, ColumnIdentity, ColumnStatement, DraftContext, Participant, Relationship

SOURCE = "😀 Study account.\ncode identifies a participant. amount is measured in dollars.\ncode and amount are recorded together; this does not establish causation."
COLUMNS = [ColumnIdentity(column_id="c1", position=0, name="code", key="code"), ColumnIdentity(column_id="c2", position=1, name="amount", key="amount")]


def citation(text=SOURCE, quote="amount is measured in dollars."):
    start = text.index(quote)
    return Citation(source_line=text[:start].count("\n") + 1, start=start, end=start + len(quote), quote=quote)


def draft(columns=COLUMNS, source=SOURCE):
    result = DraftContext(columns=[ColumnContext(column_id=c.column_id) for c in columns])
    result.columns[1].statements.append(ColumnStatement(id="s1", field="units", value="dollars", basis="source_statement", citations=[citation(source)]))
    result.relationships.append(
        Relationship(
            id="r1",
            kind="association",
            value="Recorded together, without an established causal relationship.",
            basis="source_statement",
            citations=[citation(source, "code and amount are recorded together; this does not establish causation.")],
            participants=[Participant(kind="column", id=c.column_id) for c in columns],
        )
    )
    return result


class Scripted:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        assert self.outputs, "Unexpected model call"
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, self.schema), f"Wrong scripted contract: {type(item)} != {self.schema}"
        return item


def happy(columns=COLUMNS, source=SOURCE):
    return Scripted([draft(columns, source), Assessment()])

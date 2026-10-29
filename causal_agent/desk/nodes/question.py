"""Causal-question validation used only when preparing analysis."""

from causal_agent.common.contracts import QuestionFrame
from causal_agent.memory.records import Memory


def validate(frame: QuestionFrame, memory: Memory) -> list[str]:
    """Whether the question, as read, is one this file can answer. Code over the frame and the file's facts."""
    out: list[str] = []
    from causal_agent.lane.intake import blank, parse_filter, parse_window

    if not blank(frame.scope.population_filter):
        clauses = parse_filter(frame.scope.population_filter or "")
        if clauses is None:
            out.append("the population restriction needs a named column, comparison and value")
        else:
            for name, _, _ in clauses:
                column = memory.column(name)
                if column is None:
                    out.append(f"the population restriction names unknown column {name!r}")
                elif column.name in {frame.outcome, frame.cause}:
                    out.append("the available endpoints cannot identify a population selected by treatment or outcome")
    if not blank(frame.scope.window) and parse_window(frame.scope.window or "") is None:
        out.append("the time restriction needs explicit bounds, such as from A to B")
    if frame.intent != "effect_of_change":
        out.append(
            {
                "driver_search": "it asks what drives an outcome rather than what one change did to it",
                "root_cause": "it asks why something happened rather than what one change did to an outcome",
                "not_causal": "it does not ask what a change did to an outcome",
            }[frame.intent]
        )
    oc = memory.column(frame.outcome) if frame.outcome else None
    if oc is None:
        out.append("I could not match the outcome to a column in this file")
    elif oc.facts.constant:
        out.append(f"'{oc.name}' takes one value in this file, so nothing can have moved it")
    cc = memory.column(frame.cause) if frame.cause else None
    if frame.intent == "effect_of_change":
        if cc is None:
            out.append("I could not find a column that records who got the change, or when; name it, or say which column tells the rows apart")
        elif cc.facts.constant:
            out.append(f"every row has the same value of '{cc.name}', so there is nobody without the change to compare against")
        elif oc is not None and cc.key == oc.key:
            out.append("the change and the outcome are the same column")
    return out

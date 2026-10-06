"""One real ask to the exploration agent, as the desk would make it, from the command line:

    uv run python -m causal_agent.evals.explore students "the mean reading score by lunch, as a picture"
    uv run python -m causal_agent.evals.explore students "how lunch and parental education go together" --outcome "math score" --treatment "test preparation course"

The memory's context and columns are what the desk sends; the outcome and the treatment default to what the memory holds.
One root span, `explore:cli`; the agent's process nests under it. Prints the summary, the facts with their addresses, and
the folder."""

from __future__ import annotations

import argparse

from causal_agent.common import trace
from causal_agent.desk.nodes.interview import draw_context
from causal_agent.desk.nodes.shared import _csv_path
from causal_agent.explore import client as XC
from causal_agent.memory import store


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("ask")
    ap.add_argument("--outcome", default=None, help="the outcome column, by the file's own name; default: the memory's frame, if any")
    ap.add_argument("--treatment", default=None, help="the treatment column, by the file's own name; default: claim:assignment.treatment_column")
    ap.add_argument("--frozen", action="store_true", help="a design has run: nothing is held back")
    ap.add_argument("--design", type=int, default=None)
    args = ap.parse_args(argv)
    memory = store.memory_for(args.dataset)
    context, columns = draw_context(memory)
    treatment = args.treatment
    if treatment is None:
        t = memory.value("claim:assignment.treatment_column")
        col = memory.column(str(t)) if t else None
        treatment = col.name if col else None
    with trace.span("explore:cli", inputs={"dataset": args.dataset, "ask": args.ask}):
        a, d, thoughts = XC.explore(
            dataset=memory.name,
            ask=args.ask,
            context=context,
            columns=columns,
            csv=_csv_path(memory),
            outcome=args.outcome,
            treatment=treatment,
            frozen=args.frozen,
            design=args.design,
            memory_version=memory.version,
            data_version=memory.data_version,
        )
    if a is None:
        assert d is not None
        print("declined:", d.reason)
    else:
        print(a.render())
        print("folder:", XC.VA.folder(a.dataset, a.id))
    for t in thoughts:
        if t.text:
            print(f"-- thought ({t.node}): {t.text[:600]}")


if __name__ == "__main__":
    main()

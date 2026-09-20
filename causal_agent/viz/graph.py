"""The viz subgraph: a Point in, a Figure out.

    START ─ candidates ─ pick ─ render ─ check ─ END

`candidates` is code: the declared figures whose family matches the point and whose needs the memory meets. `pick` is a
judgement only when more than one candidate stands; one candidate is chosen by code, none is a refusal with why. `render`
calls the pre-viz function on the table. `check` is code: the figure has values, and every address it draws on resolves.

    from causal_agent.viz.graph import make
    make(Point(family="adjustment", claim="both arms exist at every lunch level"), dataset="students3", outcome="math score")
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import pandas as pd
import yaml
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from causal_agent.common.addresses import norm_address
from causal_agent.common.contracts import Thought
from causal_agent.common.llm import structured
from causal_agent.memory import store
from causal_agent.memory.catalogue import load_thresholds
from causal_agent.memory.records import Memory
from causal_agent.memory.views import context_text, table_of
from causal_agent.profile.data import column
from causal_agent.viz import prompts as P
from causal_agent.viz.previz import adjustment, diff_in_diff, discontinuity
from causal_agent.viz.spec import Figure, FigureSpec, Point

_HERE = Path(__file__).parent
PICK_ATTEMPTS = 3
_retry = RetryPolicy(max_attempts=3, initial_interval=1.0)


class FigureDecl(BaseModel):
    name: str
    family: str
    shows: str
    makes_the_point_when: str
    needs: list[str]

    def render(self) -> str:
        return f"{self.name}: shows {self.shows}. Makes the point when {self.makes_the_point_when}."


def load_figures(path: str | Path | None = None) -> list[FigureDecl]:
    raw = yaml.safe_load(Path(path or _HERE / "figures.yaml").read_text())
    return [FigureDecl(name=k, **v) for k, v in raw.items()]


class Choice(BaseModel):
    """The pick: one declared figure, or none."""

    function: str = Field(description="a name from the list, or 'none'")
    why: str
    cites: list[str] = Field(default_factory=list)


class VizState(TypedDict, total=False):
    dataset: str
    point: Point
    outcome: str | None
    treatment: str | None
    candidates: list[str]
    choice: Choice | None
    figure: Figure | None
    pick_attempts: int
    debug: Annotated[list[Thought], operator.add]


# ------------------------------------------------------------------ candidates (code)


def _has(memory: Memory, need: str, outcome: str | None) -> bool:
    if need == "outcome":
        return bool(outcome) and outcome is not None and memory.column(outcome) is not None
    v = memory.value(need)
    return v is not None and v != [] and v != ""


def candidates(state: VizState) -> dict:
    memory = store.memory_for(state["dataset"])
    fits = [d.name for d in load_figures() if d.family == state["point"].family and all(_has(memory, n, state.get("outcome")) for n in d.needs)]
    return {"candidates": fits, "choice": None, "figure": None, "pick_attempts": 0, "debug": []}


# ------------------------------------------------------------------ pick (a judgement only among several)


def pick(state: VizState) -> dict:
    cands = state.get("candidates") or []
    point = state["point"]
    if not cands:
        why = f"no figure declared for {point.family} has what it needs settled in the memory"
        return {"choice": Choice(function="none", why=why), "figure": Figure.refused(why)}
    if len(cands) == 1:
        return {"choice": Choice(function=cands[0], why="the only figure that fits the point's family and the memory")}
    memory = store.memory_for(state["dataset"])
    decls = {d.name: d for d in load_figures()}
    errors = ""
    debug = []
    allowed = set(memory.addresses()) | {"dataset.note", "change:1.note"} | set(point.about)
    for attempt in range(PICK_ATTEMPTS):
        user = P.PICK_USER.format(
            point=f"{point.claim}\nabout: {', '.join(f'[{a}]' for a in point.about) or '(no addresses)'}"
            + (f"\ncolumns named: {', '.join(point.columns)}" if point.columns else ""),
            context=context_text(memory),
            figures="\n".join(decls[c].render() for c in cands),
            names=", ".join(cands),
            errors=errors,
        )
        choice, thought = structured(Choice, P.PICK_SYSTEM, user, node="viz.pick")
        debug.append(thought)
        bad = [a for a in choice.cites if norm_address(a) not in {norm_address(x) for x in allowed}]
        if choice.function not in cands + ["none"]:
            errors = f"\nPREVIOUS ANSWER WAS REJECTED\n- {choice.function!r} is not one of the names\n"
        elif bad:
            errors = "\nPREVIOUS ANSWER WAS REJECTED\n- not addresses in the material: " + ", ".join(bad) + "\n"
        else:
            out: dict = {"choice": choice, "pick_attempts": attempt + 1, "debug": debug}
            if choice.function == "none":
                out["figure"] = Figure.refused(choice.why)
            return out
    return {
        "choice": Choice(function="none", why="the pick did not resolve"),
        "figure": Figure.refused("the pick did not resolve: " + errors.strip()),
        "pick_attempts": PICK_ATTEMPTS,
        "debug": debug,
    }


# ------------------------------------------------------------------ render (fact)


def _render_overlap(memory: Memory, df: pd.DataFrame, state: VizState, th: dict) -> Figure:
    a = memory.values_of("claim:assignment")
    t = column(df, a.get("treatment_column"))
    named = [c for n in (state["point"].columns or []) if (c := column(df, n)) is not None]
    deps = named or [c for d in (a.get("depends_on") or []) if (c := column(df, d)) is not None]
    return adjustment.overlap(
        df,
        t,
        str(a.get("treated_level")),
        deps,
        floor=int(th["arms"]["min_rows_cell"]),
        addresses=["claim:assignment.depends_on", "claim:assignment.treatment_column"],
    )


def _render_did(memory: Memory, df: pd.DataFrame, state: VizState, th: dict) -> Figure:
    a, ch = memory.values_of("claim:assignment"), memory.values_of("claim:change")
    return diff_in_diff.by_group_over_time(
        df,
        column(df, state.get("outcome")),
        column(df, ch.get("date_column")),
        column(df, a.get("treatment_column")),
        str(a.get("treated_level")),
        ch.get("period_value"),
        floor=int(th["probe"]["min_pre_periods"]),
        addresses=["claim:change.date_column", "claim:change.period_value"],
    )


def _render_density(memory: Memory, df: pd.DataFrame, state: VizState, th: dict) -> Figure:
    a = memory.values_of("claim:assignment")
    return discontinuity.density(
        df,
        column(df, a.get("score_column")),
        float(a["cutoff"]),
        floor=int(th["cutoff"]["min_rows_side"]),
        addresses=["claim:assignment.score_column", "claim:assignment.cutoff"],
    )


def _render_outcome_by_bin(memory: Memory, df: pd.DataFrame, state: VizState, th: dict) -> Figure:
    a = memory.values_of("claim:assignment")
    return discontinuity.outcome_by_bin(
        df,
        column(df, a.get("score_column")),
        float(a["cutoff"]),
        column(df, state.get("outcome")),
        floor=int(th["cutoff"]["min_rows_side"]),
        addresses=["claim:assignment.score_column", "claim:assignment.cutoff"],
    )


FUNCTIONS: dict[str, Callable[[Memory, pd.DataFrame, VizState, dict], Figure]] = {
    "adjustment.overlap": _render_overlap,
    "diff_in_diff.by_group_over_time": _render_did,
    "discontinuity.density": _render_density,
    "discontinuity.outcome_by_bin": _render_outcome_by_bin,
}


def render(state: VizState) -> dict:
    if state.get("figure") is not None:  # refused at pick
        return {}
    choice = state.get("choice")
    assert choice is not None, "render runs after pick"
    name = choice.function
    fn = FUNCTIONS.get(name)
    if fn is None:
        return {"figure": Figure.refused(f"{name!r} is declared but not built", name)}
    memory = store.memory_for(state["dataset"])
    fig = fn(memory, table_of(memory), state, load_thresholds())
    fig.function = fig.function or name
    return {"figure": fig}


# ------------------------------------------------------------------ check (fact)


def check_spec(spec: FigureSpec, ok: set[str]) -> list[str]:
    """The problems with a figure, by code: a graph needs nodes and arrows between them; any other kind needs values; and
    every address the figure draws on must resolve in `ok`. Empty means the figure stands."""
    problems = []
    if spec.kind == "graph":
        ids = {n.id for n in spec.nodes}
        if not spec.nodes:
            problems.append("a graph with no nodes")
        loose = [f"{e.src} -> {e.dst}" for e in spec.edges if e.src not in ids or e.dst not in ids]
        if loose:
            problems.append("arrows between nodes the figure does not have: " + ", ".join(loose))
    elif not spec.series or all(not s.x for s in spec.series):
        problems.append("no series to draw")
    elif all(y is None for s in spec.series for y in s.y):
        problems.append("every value is empty")
    okn = {norm_address(a) for a in ok}
    bad = [a for a in spec.draws_on if norm_address(a) not in okn]
    if bad:
        problems.append("draws on addresses that do not resolve: " + ", ".join(bad))
    return problems


def check(state: VizState) -> dict:
    fig = state.get("figure")
    if fig is None or not fig.made or fig.spec is None:
        return {}
    memory = store.memory_for(state["dataset"])
    ok = set(memory.addresses()) | ({fig.probe.address} if fig.probe else set())
    problems = check_spec(fig.spec, ok)
    if problems:
        return {"figure": Figure.refused("the figure did not pass its check: " + "; ".join(problems), fig.function)}
    return {}


# ------------------------------------------------------------------ the graph


def build() -> StateGraph:
    b = StateGraph(VizState)
    b.add_node("candidates", candidates)
    b.add_node("pick", pick, retry_policy=_retry)
    b.add_node("render", render)
    b.add_node("check", check)
    b.add_edge(START, "candidates")
    b.add_edge("candidates", "pick")
    b.add_edge("pick", "render")
    b.add_edge("render", "check")
    b.add_edge("check", END)
    return b


graph = build().compile()


def compile_local():
    return build().compile(checkpointer=InMemorySaver())


def make(point: Point, dataset: str, outcome: str | None = None, treatment: str | None = None) -> Figure:
    """The viz tool as one call: a Point in, a Figure out (made, or refused with why)."""
    import uuid

    out = compile_local().invoke(
        {"dataset": dataset, "point": point, "outcome": outcome, "treatment": treatment}, {"configurable": {"thread_id": str(uuid.uuid4())}}
    )
    return out["figure"]

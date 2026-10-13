"""A lane's ladder, declared once: every node of the lane's graph in the order it runs, what decides it,
where it may go, whether it may stop, the budget a judgement has, what it may ask back and what it stops on. The graph is
built from this file and a test holds the two equal, so a node cannot go where the declaration does not say.

The end states of a lane are derived, never listed. From this file with the space, the catalogue, the checks and the beliefs
come the three kinds every lane shares: the cells a design can reach (an answer), the addresses a rung can send back (a
question), and the typed stops each rung can raise, each with the reason and the fix the declaration gives. A new lane gets
its end states by writing its files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from causal_agent.common.contracts import EndpointCatalogue
from causal_agent.lane.ladder import HARD_THREATS
from causal_agent.lane.space import Space, cells

STOP = "feasibility"  # where every typed stop and every ask goes; the builder's node
TAIL = ("figures", "assemble")  # after the last rung and after a stop; the builder's nodes
DECIDERS = ("code", "pack", "judgement", "ask")

# the causes a rung may declare it stops on, with the words a stop of that cause carries
CAUSES: dict[str, tuple[str, str]] = {
    "intake": ("the hand-off names no file, or a column it names is not in the file", "a hand-off whose file and columns exist"),
    "shape": ("the table fails a shape rule the lane declares", "a table the rule holds on"),
    "graph": ("what the rungs built fails verification", "clearer column notes about what fed the decision"),
    "checks": ("a design check fails its hard threshold", "a design that passes the check"),
    "beliefs": ("what the person believes rules the design out, by the lane's belief rules", "a different answer at the belief's address"),
    "threats": ("a threat the pack names is hard", "a file drawn and measured so the threat does not stand"),
    "assessment": ("a hard flag the assessment could not answer", "a design that answers it, or the person's word on it"),
    "engine": ("the estimator failed to fit", "a different estimator entry"),
}
GATE = ("the record was refused on every try", "clearer notes on what the rung reads")


class Rung(BaseModel):
    """One node of the lane's graph, in the order it runs."""

    model_config = ConfigDict(extra="forbid")
    name: str
    level: str = ""  # the ladder level the rung belongs to, in words
    decider: dict[str, str] = Field(description="code | pack | judgement | ask -> when that decides")
    exits: list[str] | None = None  # where it goes on; the next rung when not said, the tail after the last
    stop: bool = True  # may end the run with a typed stop or an ask
    route: str | None = None  # the function in the lane's nodes that picks among the exits, or fans out to a worker
    worker: bool = False  # reached by Send from the rung before, once per task
    budget: int | None = None  # tool calls per episode; a judgement declares one, nothing else has one
    tools: list[str] = Field(default_factory=list)
    asks: list[str] = Field(default_factory=list, description="the addresses the rung itself may send back as a question")
    stops_on: list[str] = Field(default_factory=list, description="the declared causes, among CAUSES")
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    gate: list[str] = Field(default_factory=list)

    @property
    def judgement(self) -> bool:
        return "judgement" in self.decider

    @model_validator(mode="after")
    def _well_formed(self) -> Rung:
        if not self.decider:
            raise ValueError(f"{self.name}: no decider")
        bad = set(self.decider) - set(DECIDERS)
        if bad:
            raise ValueError(f"{self.name}: decider keys {sorted(bad)} are not among {DECIDERS}")
        bad = set(self.stops_on) - set(CAUSES)
        if bad:
            raise ValueError(f"{self.name}: stops on {sorted(bad)}, which are not declared causes; the causes are {sorted(CAUSES)}")
        if self.judgement and self.budget is None:
            raise ValueError(f"{self.name}: a judgement declares its budget")
        if not self.judgement and (self.budget is not None or self.tools):
            raise ValueError(f"{self.name}: only a judgement has a budget and tools")
        if self.asks and "ask" not in self.decider:
            raise ValueError(f"{self.name}: a rung that asks says so in its decider")
        if self.stops_on and not self.stop:
            raise ValueError(f"{self.name}: stops on {self.stops_on} but never stops")
        return self


class Loop(BaseModel):
    """One edge back down the ladder, declared with its bound and its reason."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    src: str = Field(alias="from")
    to: str
    max: int
    why: str


class LadderSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str  # the rung the ladder climbs to; everything after it works the design out
    loops: list[Loop] = Field(default_factory=list)
    rungs: list[Rung]

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.rungs]

    def rung(self, name: str) -> Rung:
        for r in self.rungs:
            if r.name == name:
                return r
        raise KeyError(f"{name!r} is not a rung of this ladder")

    def exits_of(self, r: Rung) -> list[str]:
        """Where the rung goes on: as declared, else the next rung, else the tail."""
        if r.exits is not None:
            return list(r.exits)
        i = self.names.index(r.name)
        return [self.names[i + 1]] if i + 1 < len(self.names) else [TAIL[0]]

    def budget(self, name: str) -> int:
        b = self.rung(name).budget
        if b is None:
            raise KeyError(f"{name!r} is not a judgement; it has no budget")
        return b

    def edges(self) -> set[tuple[str, str]]:
        """Every edge of the graph the declaration describes, the builder's tail included."""
        out = {("__start__", self.names[0]), (STOP, TAIL[0]), (TAIL[0], TAIL[1]), (TAIL[1], "__end__")}
        for r in self.rungs:
            out.update((r.name, x) for x in self.exits_of(r))
            if r.stop:
                out.add((r.name, STOP))
        return out

    def judgements(self) -> list[Rung]:
        return [r for r in self.rungs if r.judgement]

    @model_validator(mode="after")
    def _well_formed(self) -> LadderSpec:
        names = self.names
        if not names:
            raise ValueError("a ladder has at least one rung")
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"rungs declared twice: {dup}")
        reserved = set(names) & {STOP, *TAIL, "__start__", "__end__"}
        if reserved:
            raise ValueError(f"{sorted(reserved)} are the builder's nodes, not rungs")
        if self.goal not in names:
            raise ValueError(f"the goal {self.goal!r} is not a rung")
        for lp in self.loops:
            if lp.src not in names or lp.to not in names:
                raise ValueError(f"the loop {lp.src} -> {lp.to} names a rung that is not declared")
            if names.index(lp.to) >= names.index(lp.src):
                raise ValueError(f"the loop {lp.src} -> {lp.to} does not go back down the ladder")
        declared = {(lp.src, lp.to) for lp in self.loops}
        back: set[tuple[str, str]] = set()
        for i, r in enumerate(self.rungs):
            exits = self.exits_of(r)
            if not exits:
                raise ValueError(f"{r.name} goes nowhere")
            for x in exits:
                if x == STOP or x in TAIL[1:]:
                    raise ValueError(f"{r.name} exits to {x}; a stop is declared with stop: true and the tail is the builder's")
                if x not in names and x != TAIL[0]:
                    raise ValueError(f"{r.name} exits to {x!r}, which is not a rung")
                if x in names and names.index(x) <= i:
                    back.add((r.name, x))
            if r.route is not None and len(exits) + int(r.stop) < 2 and not any(x in names and self.rung(x).worker for x in exits):
                raise ValueError(f"{r.name}: a route picks among several exits or fans out to a worker")
            if r.worker:
                sources = [s.name for s in self.rungs if r.name in self.exits_of(s)]
                if len(sources) != 1:
                    raise ValueError(f"the worker {r.name} is reached from {sources}; a worker is reached from one rung")
        if back != declared:
            raise ValueError(f"the edges back down the ladder are {sorted(back)} and the loops declared are {sorted(declared)}; they must match")
        return self


def load_spec(path: Path) -> LadderSpec:
    return LadderSpec.model_validate(yaml.safe_load(Path(path).read_text()) or {})


# ------------------------------------------------------------------ the end states, derived


class Answer(BaseModel):
    """A cell of the space the design can reach, with what runs it."""

    point: dict[str, Any]
    estimators: list[str]
    refuters: list[str]


class Ask(BaseModel):
    """One question a rung can send back, keyed to the address the desk can settle."""

    rung: str
    address: str
    question: str = ""


class Stop(BaseModel):
    """One typed stop a rung can raise: the cause, the reason and the fix the declaration gives."""

    rung: str
    cause: str
    reason: str
    fix: str


class Ends(BaseModel):
    answers: list[Answer] = Field(default_factory=list)
    asks: list[Ask] = Field(default_factory=list)
    stops: list[Stop] = Field(default_factory=list)


def _check_stops(rung: str, checks: dict[str, Any]) -> list[Stop]:
    """Every check with a hard threshold or a hard rule in checks.yaml, in the file's words."""
    words = checks.get("in_words") or {}
    out: list[Stop] = []
    for group, body in checks.items():
        if not isinstance(body, dict):
            continue
        about = str(words.get(group, group.replace("_", " "))).replace("{column}", "a column")
        for name, spec in body.items():
            if isinstance(spec, dict) and isinstance(spec.get("hard"), int | float):
                out.append(
                    Stop(rung=rung, cause=f"check:{group}.{name}", reason=f"{about}; {name.replace('_', ' ')} past {spec['hard']}", fix=CAUSES["checks"][1])
                )
            elif spec is True and ("hard" in name):
                out.append(Stop(rung=rung, cause=f"check:{group}.{name}", reason=f"{about}; {name.replace('_', ' ')}", fix=CAUSES["checks"][1]))
    return out


def _belief_ends(rung: str, beliefs: dict[str, Any]) -> tuple[list[Ask], list[Stop]]:
    """What beliefs.yaml can do by itself at the rung that reads it: a stop on a status or on a status with a check, and
    a question on a status, on a status with a check, or on an unknown."""
    asks: list[Ask] = []
    stops: list[Stop] = []
    for kind, spec in (beliefs.get("beliefs") or {}).items():
        address = str(spec.get("from") or f"claim:{kind}.{spec.get('value_field', 'exists')}")
        for status, row in (spec.get("by_status") or {}).items():
            if row.get("level") == "stop":
                stops.append(Stop(rung=rung, cause=f"belief:{kind}.{status}", reason=str(row.get("caveat", "")), fix=f"a different answer at [{address}]"))
            if "ask" in row:
                asks.append(Ask(rung=rung, address=address, question=str(row["ask"].get("question", ""))))
        for check, rows in (spec.get("with_check") or {}).items():
            for row in rows:
                if row.get("then") == "stop":
                    stops.append(
                        Stop(rung=rung, cause=f"belief:{kind}+check:{check}", reason=str(row.get("reason", "")), fix=f"a different answer at [{address}]")
                    )
                elif row.get("then") == "ask":
                    asks.append(Ask(rung=rung, address=address, question=str(row.get("question", ""))))
    for address, row in (beliefs.get("unknowns") or {}).items():
        if "ask" in row:
            asks.append(Ask(rung=rung, address=str(address), question=str(row["ask"].get("question", ""))))
    return asks, stops


def _space_stops(spec: LadderSpec, rung: str, space: Space) -> list[Stop]:
    """What the space lets the rung that decides a dimension stop on: no runnable value left, and each value the declaration
    names unsupported; a combination is the stop of the rung that decides its last dimension."""
    fix = "a file inside the space" if rung == spec.names[0] else "a design inside the space"
    out: list[Stop] = []
    for name, d in space.dimensions.items():
        if d.decided_by != rung:
            continue
        if (d.read_from or not d.discrete()) and not d.soft:  # a soft dimension is a decline at the freeze, never a wall
            out.append(Stop(rung=rung, cause=f"space:{name}", reason=f"no value of {name} the lane runs is left", fix=fix))
        for value, reason in d.unsupported.items():
            out.append(Stop(rung=rung, cause=f"space:{name}={value}", reason=reason, fix=fix))
    for u in space.unsupported:
        deciders = [space.dimensions[k].decided_by for k in u.when if k in space.dimensions]
        last = max(deciders, key=lambda n: spec.names.index(n) if n in spec.names else -1, default=None)
        if last == rung:
            out.append(Stop(rung=rung, cause="space:" + "+".join(f"{k}={v}" for k, v in u.when.items()), reason=u.reason, fix=fix))
    return out


def ends(
    spec: LadderSpec,
    *,
    space: Space | None = None,
    estimators: list[Any] | None = None,
    refuters: list[Any] | None = None,
    checks: dict[str, Any] | None = None,
    beliefs: dict[str, Any] | None = None,
    endpoints: EndpointCatalogue | None = None,
) -> Ends:
    """Every end state the declarations allow, in ladder order: the cells of the space that are served, the questions each rung
    can send back, and the typed stops each rung can raise. A lane without a space has no answers to enumerate yet."""
    out = Ends()
    if endpoints is not None:
        out.answers = [Answer(point={"endpoint": e.id, **e.configuration}, estimators=e.estimators, refuters=[]) for e in endpoints.endpoints if e.executable]
    elif space is not None:
        out.answers = [
            Answer(point=c.point, estimators=c.estimators, refuters=c.refuters) for c in cells(space, estimators or [], refuters or []) if c.unsupported is None
        ]
    for r in spec.rungs:
        if space is not None and r.stop:
            out.stops += _space_stops(spec, r.name, space)
        for a in r.asks:
            out.asks.append(Ask(rung=r.name, address=a))
        for c in r.stops_on:
            if c == "checks":
                out.stops += _check_stops(r.name, checks or {})
            elif c == "beliefs":
                asks, stops = _belief_ends(r.name, beliefs or {})
                out.asks += asks
                out.stops += stops
            elif c == "threats":
                out.stops += [Stop(rung=r.name, cause=f"threat:{n}", reason=t, fix=CAUSES["threats"][1]) for n, t in HARD_THREATS.items()]
            else:
                reason, fix = CAUSES[c]
                out.stops.append(Stop(rung=r.name, cause=c, reason=reason, fix=fix))
        if r.judgement and r.stop:
            out.stops.append(Stop(rung=r.name, cause="gate", reason=GATE[0], fix=GATE[1]))
    return out

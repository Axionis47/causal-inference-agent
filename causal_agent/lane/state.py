"""The state keys every lane shares, and the reducers that keep a fan-out honest."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any

from pydantic import BaseModel
from typing_extensions import TypedDict

from causal_agent.common.contracts import ContextDeficiency, Decline, Feasibility, Handoff, Thought


def merge_dicts(a: dict | None, b: dict | None) -> dict:
    """Two workers may each add keys; a later write to the same key wins."""
    return {**(a or {}), **(b or {})}


def merge_episode_visits(a: dict | None, b: dict | None) -> dict:
    """Every visit survives; the key identifies the episode rather than a node's latest output."""
    out = dict(a or {})
    for name, episode in (b or {}).items():
        visit = getattr(episode, "visit", None)
        if not visit:
            raise ValueError(f"{name}: episode has no visit identity")
        out[f"{name}:{visit}"] = episode
    return out


def by_key(keyfn: Callable[[Any], Any]):
    """A list reducer that replaces an item with the same key instead of appending a duplicate: a re-pick that runs
    every contrast again leaves one estimate per (contrast, method)."""

    def reduce(a: list | None, b: list | None) -> list:
        out = list(a or [])
        index = {keyfn(x): i for i, x in enumerate(out)}
        for x in b or []:
            k = keyfn(x)
            if k in index:
                out[index[k]] = x
            else:
                index[k] = len(out)
                out.append(x)
        return out

    return reduce


class LaneInput(BaseModel):
    """What a lane is given: the question, the pack, the dataset. The form Studio shows for a run; the graph writes the input
    as given, so `load` validates the pack from JSON and writes it back as the object every rung reads."""

    question: str = ""
    handoff: Handoff
    dataset: str = ""


class LaneState(TypedDict, total=False):
    """What the desk gives a lane and what every lane keeps. A specialist state inherits it and adds its own."""

    question: str
    handoff: Handoff | None
    dataset: str
    specialist_result: dict | None
    lead_selection: dict
    debug: Annotated[list[Thought], operator.add]

    run_dir: str
    table_path: str
    columns: dict[str, str]  # key -> raw name
    declines: Annotated[list[Decline], operator.add]
    checks: list  # list[CheckResult]
    check_facts: Annotated[dict, merge_dicts]
    context_deficiency: ContextDeficiency | None
    feasibility: Feasibility | None
    figures: list[dict]
    report: str
    ladder: Any  # the lane's own Ladder record: the rungs climbed so far, each line addressed
    episodes: Annotated[dict[str, Any], merge_episode_visits]

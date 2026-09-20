"""The state keys every lane shares, and the reducers that keep a fan-out honest."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Callable

from typing_extensions import TypedDict

from causal_agent.common.contracts import Decline, Feasibility, Handoff, LaneAsk, Thought


def merge_dicts(a: dict | None, b: dict | None) -> dict:
    """Two workers may each add keys; a later write to the same key wins."""
    return {**(a or {}), **(b or {})}


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


class LaneState(TypedDict, total=False):
    """What the desk gives a lane and what every lane keeps. A specialist state inherits it and adds its own."""

    question: str
    handoff: Handoff | None
    dataset: str
    specialist_result: dict | None
    debug: Annotated[list[Thought], operator.add]

    run_dir: str
    table_path: str
    columns: dict[str, str]  # key -> raw name
    declines: Annotated[list[Decline], operator.add]
    case: Any  # lane.case.Case, kept as an object
    checks: list  # list[CheckResult]
    check_facts: Annotated[dict, merge_dicts]
    ask: LaneAsk | dict | None
    feasibility: Feasibility | None
    figures: list[dict]
    report: str

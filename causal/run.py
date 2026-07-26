"""The spine. Nine lines of control flow, and every name maps to a file.

    profile -> intake -> design menu -> [you choose] -> lane -> report

The choose step is a genuine stop, not a formality. Which design a dataset can
support is a question about the data, and code answers it. Which one to use is
a question about the world, and a person answers that.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import lanes as lane_module
from .design import Option, options
from .estimate import Estimate
from .intake import Intake, read_question
from .profile import Profile, profile
from .report import claim_strength, headline, narrate

LANES = {
    "observational": lane_module.observational,
    "matching": lane_module.matching,
    "did": lane_module.did,
    "rdd": lane_module.rdd,
    "iv": lane_module.iv,
    "survival": lane_module.survival,
    "mediation": lane_module.mediation,
    "time_series": lane_module.time_series,
}


@dataclass
class Analysis:
    """Everything the run produced, including where it stopped."""

    profile: Profile
    intake: Intake | None = None
    menu: list[Option] = field(default_factory=list)
    lane: str = ""
    estimate: Estimate | None = None
    strength: str = ""
    headline: str = ""
    narrative: str = ""
    stopped: str = ""  # why it went no further, empty if it completed


def plan(df: pd.DataFrame, question: str, context: str = "") -> Analysis:
    """Everything before the human decision: read the data, read the question,
    and lay out which designs are on the table."""
    p = profile(df)
    intake = read_question(question, context, p)
    if not intake.usable:
        return Analysis(profile=p, intake=intake, stopped=intake.problem)
    menu = options(df, p, treatment=intake.treatment, outcome=intake.outcome)
    return Analysis(profile=p, intake=intake, menu=menu)


def estimate(
    df: pd.DataFrame,
    a: Analysis,
    lane: str,
    *,
    narrate_result: bool = True,
    **lane_kwargs,
) -> Analysis:
    """Everything after it: run the chosen design and say what it means.

    `lane` is passed in rather than picked here. That is the whole point of the
    gate; nothing in this function guesses.
    """
    if lane not in LANES:
        a.stopped = f"no lane called '{lane}'"
        return a

    a.lane = lane
    est = LANES[lane](df, **lane_kwargs)
    a.estimate = est
    a.strength = claim_strength(lane, est)
    a.headline = headline(lane, est, a.intake.treatment, a.intake.outcome)
    if narrate_result:
        a.narrative = narrate(lane, est, a.intake.treatment, a.intake.outcome)
    return a

"""Reason from context to an experiment design, the same way every time.

Two things live here: the shape of context worth having, and a reasoning step
that turns it into a recommended lane.

Why a template. Which design a dataset supports is decided by a handful of
facts, and not one of them is visible in the columns. Were units observed more
than once? Did treatment arrive at a moment, or by crossing a threshold, or by
someone choosing? Was the outcome measured after the treatment? A person knows
these and rarely writes them down, so the template asks for exactly them and
nothing else. Every field maps to a fork in the decision procedure below.

Why a fixed procedure rather than open reasoning. Asked "which design fits?"
the same model gives different answers to the same data depending on phrasing,
because it is weighing considerations in whatever order they occur to it. The
procedure below fixes that order. The model still reasons, and still says when
the answer is unclear, but it reasons through the same forks in the same
sequence, which is what makes the recommendation reproducible.

The recommendation is never binding. It arrives at the gate with its reasoning
attached and a person accepts or overrides it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .profile import Profile

PROJECT = os.environ.get("GCP_PROJECT_ID", "plotpointe")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
MODEL = os.environ.get("MODEL", "gemini-2.5-flash")

LANES = ("observational", "matching", "did", "rdd", "iv", "survival",
         "mediation", "time_series")


# --------------------------------------------------------------------------
# The context template
# --------------------------------------------------------------------------

TEMPLATE = """\
One row is:            e.g. one patient, one store in one month, one customer
Treatment arrived by:  randomisation | a rule or threshold | a policy at a date |
                       people choosing | unknown
Measured before it:    which columns describe the world before treatment
Measured after it:     which columns describe the world after treatment
Units seen more than once: yes (and which column identifies the unit) | no
The outcome is:        a level | a count | a time until something happens |
                       a rate over time
Also worth knowing:    anything that plausibly drives both the treatment and the
                       outcome, and anything you know is NOT a confounder"""

QUESTIONS = [
    ("unit", "What does one row represent?"),
    ("assignment", "How did units come to be treated or untreated?"),
    ("timing", "Which columns were measured before treatment, and which after?"),
    ("repeats", "Is each unit observed more than once? If so, which column identifies it?"),
    ("outcome_kind", "Is the outcome a level, a count, or a time until an event?"),
    ("confounding", "What else plausibly drives both the treatment and the outcome?"),
]


def blank_context() -> str:
    """The template, for a person to fill in."""
    return TEMPLATE


# --------------------------------------------------------------------------
# The decision procedure
# --------------------------------------------------------------------------

PROCEDURE = """Work through these in order and stop at the first that fits.
Say which step decided it.

1. Is the outcome the TIME UNTIL an event, with some units not yet having had
   it?  ->  survival
2. Did treatment arrive at a known DATE, with the same units (or one series)
   observed on both sides, and NO untreated comparison group?  ->  time_series
3. Did treatment arrive at a known date, with an untreated comparison group,
   and units observed in both periods?  ->  did
4. Was treatment decided by crossing a THRESHOLD on some measured quantity?
   ->  rdd
5. Is there something that nudged treatment but has no other route to the
   outcome?  ->  iv
6. Does the question ask HOW or THROUGH WHAT the effect travels?  ->  mediation
7. Is the treatment binary with plenty of units on both sides and covariates to
   balance?  ->  matching
8. Otherwise  ->  observational

Rules that override the order:
- Never recommend a design whose required columns are absent. Say what is
  missing instead.
- If step 5 tempts you, name the instrument and say why it cannot reach the
  outcome except through treatment. If you cannot, do not recommend iv.
- Prefer the simplest design that fits. A weaker design honestly reported beats
  a stronger one resting on an assumption nobody can defend."""

PROMPT = """Choose the experiment design for this question.

The question: {question}

Context from the person who knows this data:
{context}

The columns ({n_rows} rows):
{columns}

Treatment: {treatment}
Outcome: {outcome}

Designs the data can support on structure alone: {available}
(Designs not in that list need a column you must name, so say which.)

{procedure}

Return the recommended lane, the step number that decided it, one sentence of
reasoning, the assumption it rests on, and your confidence. If the context does
not settle it, say so in `missing` and give your best recommendation anyway."""


class Recommendation(BaseModel):
    lane: str = Field(description="one of the eight lane names")
    decided_at_step: int = Field(default=8, description="which step settled it")
    reasoning: str = Field(default="", description="one sentence")
    assumption: str = Field(default="", description="what it rests on")
    confidence: str = Field(default="low", description="high, medium or low")
    missing: str = Field(default="", description="what context would settle it")
    runner_up: str = Field(default="", description="the next best lane, if any")


@dataclass
class Choice:
    lane: str
    step: int
    reasoning: str
    assumption: str
    confidence: str
    missing: str
    runner_up: str
    failed: str = ""
    considered: list[str] = field(default_factory=list)


def recommend(
    question: str,
    context: str,
    p: Profile,
    treatment: str,
    outcome: str,
    available: list[str],
) -> Choice:
    """One call, one recommendation, arrived at through a fixed order."""
    from google import genai

    try:
        client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
        response = client.models.generate_content(
            model=MODEL,
            contents=PROMPT.format(
                question=question,
                context=context or "(nothing given)",
                n_rows=p.n_rows,
                columns=p.as_text(),
                treatment=treatment or "(none named)",
                outcome=outcome or "(none named)",
                available=", ".join(available) or "none on structure alone",
                procedure=PROCEDURE,
            ),
            config={"response_mime_type": "application/json",
                    "response_schema": Recommendation},
        )
        rec: Recommendation = response.parsed
    except Exception as exc:
        return Choice("", 0, "", "", "low", "", "", failed=f"{type(exc).__name__}: {exc}"[:200])

    lane = rec.lane.strip().lower()
    if lane not in LANES:
        return Choice("", 0, "", "", "low", "",
                      "", failed=f"recommended an unknown lane {rec.lane!r}")
    return Choice(
        lane=lane,
        step=rec.decided_at_step,
        reasoning=rec.reasoning,
        assumption=rec.assumption,
        confidence=rec.confidence,
        missing=rec.missing,
        runner_up=rec.runner_up.strip().lower(),
        considered=list(available),
    )

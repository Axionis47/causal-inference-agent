"""Turn a question in English into the columns to analyse.

This is the first of two places a model is used, and its job is narrow: read
the question and the column list, and say what is being acted on and what is
doing the acting. It does not choose a design, it does not touch a number, and
it cannot invent a column name.

Exposure has four shapes, because designs differ in where the effect enters:

    a treatment column          most questions
    group x period              difference in differences
    a date                      interrupted time series
    a threshold on a column     regression discontinuity

The first version of this file only allowed the first shape, and the model
correctly refused three datasets because of it: a minimum-wage rise is not a
column, it is a state crossed with a period. The schema was wrong, not the
reading.

One call, typed output, no agent loop. Everything downstream is code.

Provider note: this is the only file that knows an LLM vendor exists. It uses
google-genai against Vertex because that is what this machine has credentials
for. Swapping it is this file and nothing else.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .profile import Profile

PROJECT = os.environ.get("GCP_PROJECT_ID", "plotpointe")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
MODEL = os.environ.get("MODEL", "gemini-2.5-flash")

PROMPT = """You are reading a causal question against a real dataset.

The question:
{question}

What the person told us about the data:
{context}

The columns ({n_rows} rows):
{columns}

Always name the outcome column: the thing that might be affected.

Then name what does the affecting. Usually that is a single `treatment` column.
But two shapes have no such column, and you should use their fields instead of
forcing a treatment:

- a policy hitting one group at one time: leave `treatment` empty and give
  `group` (which units were exposed) and `period` (before versus after)
- something changing at a moment in a single series: leave `treatment` empty
  and give `time_column` (the dates)
- assignment decided by crossing a threshold: leave `treatment` empty and give
  `running_variable` (the column the threshold is on) and `cutoff` (the value)

Use column names exactly as written above. If the question names something that
is not in the data, say so in `problem` and leave the field empty rather than
guessing a near match.

Also say which family the question belongs to, from: effect_of_a_binary_treatment,
effect_of_a_continuous_treatment, before_and_after, difference_between_groups,
time_to_event, mechanism_or_pathway, threshold_or_cutoff, none_of_these."""


class Reading(BaseModel):
    """What the model is allowed to tell us. Anything else it says is discarded."""

    outcome: str = Field(default="", description="exact column name")
    treatment: str = Field(default="", description="exact column name, or empty")
    group: str = Field(default="", description="for a policy hitting one group")
    period: str = Field(default="", description="before versus after")
    time_column: str = Field(default="", description="dates, for a single series")
    running_variable: str = Field(default="", description="column a threshold sits on")
    cutoff: float | None = Field(default=None, description="the threshold value")
    question_family: str = Field(default="none_of_these")
    confidence: str = Field(default="low", description="high, medium or low")
    reasoning: str = Field(default="", description="one sentence")
    problem: str = Field(default="", description="what is wrong, if anything")


@dataclass
class Intake:
    outcome: str
    treatment: str
    group: str
    period: str
    time_column: str
    running_variable: str
    cutoff: float | None
    question_family: str
    confidence: str
    reasoning: str
    problem: str

    @property
    def usable(self) -> bool:
        """An outcome, plus something that does the affecting.

        Not every design has a treatment column. A difference in differences is
        identified by group crossed with period, and an interrupted series by a
        date. Demanding a treatment column for those rejects a correct reading.
        """
        return bool(self.outcome and not self.problem and self.exposure)

    @property
    def exposure(self) -> str:
        """How the effect enters, in words: 'column', 'group x period', 'time'."""
        if self.treatment:
            return self.treatment
        if self.group and self.period:
            return f"{self.group} x {self.period}"
        if self.time_column:
            return f"change over {self.time_column}"
        if self.running_variable and self.cutoff is not None:
            return f"{self.running_variable} crossing {self.cutoff:g}"
        return ""


def read_question(question: str, context: str, p: Profile) -> Intake:
    """Ask once, validate hard.

    The model's answer is checked against the real column list before it is
    returned, so a hallucinated column becomes a stated problem rather than a
    crash three steps later.
    """
    from google import genai

    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    response = client.models.generate_content(
        model=MODEL,
        contents=PROMPT.format(
            question=question,
            context=context or "(nothing given)",
            n_rows=p.n_rows,
            columns=p.as_text(),
        ),
        config={"response_mime_type": "application/json", "response_schema": Reading},
    )
    read: Reading = response.parsed

    # Structured output is not consistent about emptiness: the same prompt has
    # returned "" and the literal string "null" for the same absent field.
    # Normalise before validating, or "null" is reported as a missing column.
    EMPTY = {"", "null", "none", "n/a", "na", "nan", "-"}
    blanks = {
        role: ""
        for role in ("outcome", "treatment", "group", "period", "time_column",
                     "running_variable", "problem")
        if str(getattr(read, role)).strip().lower() in EMPTY
    }
    if blanks:
        read = read.model_copy(update=blanks)

    names = set(p.names())
    problems = []
    for role in ("outcome", "treatment", "group", "period", "time_column",
                 "running_variable"):
        value = getattr(read, role)
        if value and value not in names:
            problems.append(f"no column named '{value}' for the {role}")
            read = read.model_copy(update={role: ""})

    if not read.outcome:
        problems.append("could not identify an outcome column")
    if not (read.treatment or (read.group and read.period) or read.time_column
            or (read.running_variable and read.cutoff is not None)):
        problems.append(
            "could not identify what does the affecting: no treatment column, "
            "no group and period pair, no date column, and no threshold"
        )
    # The model's own complaint only counts if we could not recover anyway; it
    # often flags "no single treatment column" while correctly supplying group
    # and period, which is a valid reading, not a problem.
    if read.problem and problems:
        problems.insert(0, read.problem)

    return Intake(
        outcome=read.outcome,
        treatment=read.treatment,
        group=read.group,
        period=read.period,
        time_column=read.time_column,
        running_variable=read.running_variable,
        cutoff=read.cutoff,
        question_family=read.question_family,
        confidence=read.confidence,
        reasoning=read.reasoning,
        problem="; ".join(dict.fromkeys(problems)),
    )

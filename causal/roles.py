"""Ask what each column *is* in the causal story. The one real reasoning task.

Everything else in this engine is arithmetic or bookkeeping, and code does it.
This is different. Whether a column is a confounder, a mediator, or a collider
is a claim about how the world is wired, and it cannot be read off the data:
all three look identical to a correlation.

It matters because the three want opposite treatment:

    confounder  adjust for it, or the estimate is biased
    mediator    do NOT adjust; it is part of the effect you are measuring
    collider    do NOT adjust; adjusting *creates* bias where none existed

Measured on the student data: study time raises grades by 0.97 overall, and
adjusting for `failures` (a mediator) drops that to 0.76. A fifth of the effect
disappears, with no warning, because a greedy rule swept in every numeric
column. That is what this file exists to prevent.

The model reasons; code enforces. A role it returns is checked against the real
columns, treatment and outcome can never be covariates, and the rank-correlation
leak check still runs as a backstop.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .profile import Profile

PROJECT = os.environ.get("GCP_PROJECT_ID", "plotpointe")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
MODEL = os.environ.get("MODEL", "gemini-2.5-flash")

ROLES = ("confounder", "mediator", "collider", "instrument", "proxy_for_outcome",
         "irrelevant")

PROMPT = """Work out what each column is in this causal question.

The question: {question}
What we know about the data: {context}

Treatment (the cause being studied): {treatment}
Outcome (the effect being studied): {outcome}

The columns:
{columns}

For every column other than the treatment and outcome, give exactly one role:

- confounder: plausibly causes BOTH the treatment and the outcome. These are
  what we must adjust for.
- mediator: the treatment causes it, and it causes the outcome. It sits on the
  path. Adjusting for it would remove part of the effect we want to measure.
- collider: the treatment and the outcome both cause it. Adjusting for it would
  invent a relationship that is not there.
- instrument: it moves the treatment, but has no route to the outcome except
  through the treatment.
- proxy_for_outcome: a restatement of the outcome, such as the same quantity on
  a different scale, or something measured after it that reflects it.
- irrelevant: an identifier, a constant, or unrelated to this question.

Think about the direction of time and about what causes what. A column recorded
after the treatment is far more likely to be a mediator than a confounder. When
a column could be either a confounder or a mediator, say mediator: wrongly
adjusting costs us part of the real effect, while wrongly omitting an adjustment
is visible as remaining confounding we can report.

Give one short reason for each."""


class Judgement(BaseModel):
    column: str = Field(description="exact column name")
    role: str = Field(description="one of the six roles")
    why: str = Field(default="", description="one short clause")


class Reasoning(BaseModel):
    roles: list[Judgement] = Field(default_factory=list)


@dataclass
class Roles:
    by_column: dict[str, Judgement] = field(default_factory=dict)
    failed: str = ""

    def of(self, name: str) -> str:
        j = self.by_column.get(name)
        return j.role if j else "irrelevant"

    def named(self, role: str) -> list[str]:
        return [n for n, j in self.by_column.items() if j.role == role]

    def confounders(self) -> list[str]:
        return self.named("confounder")

    def why(self, name: str) -> str:
        j = self.by_column.get(name)
        return j.why if j else ""


def read_roles(
    question: str, context: str, p: Profile, treatment: str, outcome: str
) -> Roles:
    """One call. Returns roles keyed by column, or an empty result on failure.

    An empty result is not fatal: the caller falls back to the deterministic
    suggestion, which is worse but still runs. Reasoning is an improvement on
    the heuristic, not a dependency of it.
    """
    from google import genai

    try:
        client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
        response = client.models.generate_content(
            model=MODEL,
            contents=PROMPT.format(
                question=question,
                context=context or "(nothing given)",
                treatment=treatment or "(none named)",
                outcome=outcome or "(none named)",
                columns=p.as_text(),
            ),
            config={"response_mime_type": "application/json",
                    "response_schema": Reasoning},
        )
        parsed: Reasoning = response.parsed
    except Exception as exc:  # reasoning is an upgrade, never a hard dependency
        return Roles(failed=f"{type(exc).__name__}: {exc}"[:200])

    names = set(p.names())
    out: dict[str, Judgement] = {}
    for j in parsed.roles or []:
        if j.column not in names:
            continue  # a column it invented
        if j.column in (treatment, outcome):
            continue  # roles are for the other columns
        role = j.role.strip().lower()
        out[j.column] = Judgement(
            column=j.column,
            role=role if role in ROLES else "irrelevant",
            why=j.why,
        )
    return Roles(by_column=out)

"""Investigate whether an estimate holds up. The one genuinely agentic step.

Everywhere else a model is asked a question and answers it. Here it acts: it
calls a check, reads the result, and decides what to look at next. That is
warranted because what needs checking depends on the design, on what the first
check found, and on which assumptions are testable at all with this data.

Three properties keep the loop safe:

  it cannot change the estimate      the tools only measure; none refit and
                                     return a new answer
  the tools are deterministic        the model chooses what to run, code does
                                     the running, so a finding is reproducible
  it can only lower confidence       findings downgrade claim strength and
                                     never raise it

That last one matters. An investigator rewarded for finding reassurance will
find it. This one has nothing to gain: the best available outcome is that it
reports it found no problem.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .checks import CHECKS, DESCRIPTIONS, Finding
from .checks import run as run_check
from .design import ASSUMPTION

PROJECT = os.environ.get("GCP_PROJECT_ID", "plotpointe")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
MODEL = os.environ.get("MODEL", "gemini-2.5-flash")

MAX_TURNS = 8

BRIEF = """You are checking whether a causal estimate holds up. You cannot change
it; you can only find out how much weight it carries.

The design: {lane}
It rests on: {assumption}
The estimate: {estimand} = {value} on {n} rows
Arguments used: {kwargs}

Checks you can run (call them one at a time, and look at each result before
deciding what to run next):

{catalogue}

Run the checks that bear on THIS design's assumption. Do not run every check;
several do not apply and will say so. When you have enough to judge, stop
calling checks and reply with plain text summarising what you found in two or
three sentences.

If a check says untestable, that is a real finding and worth saying: an
assumption nobody can test is different from one that passed."""


@dataclass
class Diagnosis:
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    failed: str = ""

    @property
    def verdicts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.verdict] = out.get(f.verdict, 0) + 1
        return out

    def downgrades(self) -> bool:
        """Any outright failure means the estimate carries less than it claims."""
        return any(f.verdict == "fail" for f in self.findings)


def _declarations() -> list[dict]:
    return [
        {
            "name": name,
            "description": DESCRIPTIONS[name],
            "parameters": {"type": "object", "properties": {}},
        }
        for name in CHECKS
    ]


def investigate(df, lane: str, kwargs: dict, estimate: dict) -> Diagnosis:
    """Let the model choose checks, run them here, and let it read the results."""
    from google import genai
    from google.genai import types

    catalogue = "\n".join(f"  {n}: {d}" for n, d in DESCRIPTIONS.items())
    brief = BRIEF.format(
        lane=lane,
        assumption=ASSUMPTION.get(lane, "unstated"),
        estimand=estimate.get("estimand", "?"),
        value=f"{estimate.get('value', float('nan')):.6g}",
        n=estimate.get("n", "?"),
        kwargs={k: v for k, v in kwargs.items() if not str(k).startswith("_")},
        catalogue=catalogue,
    )

    try:
        client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    except Exception as exc:
        return Diagnosis(failed=f"{type(exc).__name__}: {exc}"[:200])

    contents: list = [types.Content(role="user", parts=[types.Part(text=brief)])]
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=_declarations())],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    findings: list[Finding] = []
    ran: set[str] = set()
    summary = ""

    # Run the design-agnostic checks first, whatever the model would choose.
    # Left to itself it stops as soon as it finds a problem, which is reasonable
    # behaviour and produces an incomplete report: on LaLonde it found the
    # balance failure and never learned the estimate swings from -674 to +2112
    # across subsamples. Stability is worth knowing on every design, so it is
    # not left to a judgement call. The model then adds what the design needs.
    for name in ("subsample_stability", "leave_one_out"):
        ran.add(name)
        findings.append(run_check(name, df, lane, kwargs, estimate))

    already = "\n".join(f"  {f}" for f in findings)
    contents[0].parts.append(
        types.Part(text=f"\nThese have been run for you already:\n{already}\n\n"
                        "Now add the checks that bear on this design specifically.")
    )

    for _ in range(MAX_TURNS):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=contents, config=config
            )
        except Exception as exc:
            return Diagnosis(findings=findings,
                             failed=f"{type(exc).__name__}: {exc}"[:200])

        calls = [p.function_call for c in (response.candidates or [])
                 for p in (c.content.parts or []) if getattr(p, "function_call", None)]
        if not calls:
            summary = (response.text or "").strip()
            break

        contents.append(response.candidates[0].content)
        replies = []
        for call in calls:
            name = call.name
            if name in ran:  # the same check twice tells us nothing new
                result = "already run this turn; look at the earlier result"
            else:
                ran.add(name)
                finding = run_check(name, df, lane, kwargs, estimate)
                findings.append(finding)
                result = str(finding)
            replies.append(
                types.Part.from_function_response(name=name, response={"result": result})
            )
        contents.append(types.Content(role="user", parts=replies))

    return Diagnosis(findings=findings, summary=summary)

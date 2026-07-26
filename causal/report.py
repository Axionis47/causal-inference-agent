"""Say what the estimate means, without being allowed to change it.

The second and last place a model is used. Every number in the output is
formatted here, in code, before the model sees anything. The model writes the
prose around fixed numbers; it never authors a figure and never decides how
strong the claim is.

Claim strength is a lookup, not a judgement: the design sets the ceiling, and
a interval spanning zero lowers it. A model asked to rate its own confidence
will drift; a table will not.
"""
from __future__ import annotations

import os

from .design import ASSUMPTION
from .estimate import Estimate

MODEL = os.environ.get("MODEL", "gemini-2.5-flash")
PROJECT = os.environ.get("GCP_PROJECT_ID", "plotpointe")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

# What each design can support at best. Randomisation aside, no observational
# design earns "strong" from this code.
CEILING = {
    "did": "moderate",
    "rdd": "moderate",
    "iv": "moderate",
    "observational": "weak",
    "matching": "weak",
    "survival": "weak",
    "time_series": "weak",
    "mediation": "weak",
}

PHRASE = {
    "moderate": "the evidence supports an effect, under this design's assumptions",
    "weak": "this is an adjusted association; confounding remains a live explanation",
    "none": "the data do not show a clear effect",
}

FORBIDDEN = ["proves", "proven", "definitively", "conclusively", "causes outright"]


def claim_strength(lane: str, est: Estimate, diagnostics_failed: bool = False) -> str:
    """The design sets the ceiling; evidence can only lower it.

    Three things bear on this and all three point downwards: the design's own
    ceiling, an interval covering the null, and a diagnostic that failed. There
    is deliberately no route upward. A run whose checks come back clean is not
    thereby stronger; it is merely not weaker.
    """
    if est.ci_low is not None and est.ci_low <= 0 <= est.ci_high:
        return "none"
    if est.estimand == "hazard_ratio" and est.ci_low is not None:
        if est.ci_low <= 1 <= est.ci_high:
            return "none"
    ceiling = CEILING.get(lane, "weak")
    if diagnostics_failed and ceiling == "moderate":
        return "weak"
    return ceiling


def headline(lane: str, est: Estimate, treatment: str, outcome: str,
             strength: str = "") -> str:
    """The load-bearing sentence, built from numbers, not written by a model.

    `strength` is passed in rather than recomputed. Recomputing it here would
    quietly ignore the diagnostics, so the prose could say moderate while the
    interface says weak about the same run.
    """
    interval = ""
    if est.ci_low is not None:
        interval = f" (95% interval {est.ci_low:.4g} to {est.ci_high:.4g})"
    strength = strength or claim_strength(lane, est)
    return (
        f"Estimated {est.estimand} of {treatment} on {outcome}: "
        f"{est.value:.4g}{interval}, from {est.n:,} rows via {est.estimator}. "
        f"{PHRASE[strength].capitalize()}."
    )


def facts(lane: str, est: Estimate, treatment: str, outcome: str,
          strength: str = "", findings: list | None = None) -> str:
    strength = strength or claim_strength(lane, est)
    lines = [
        f"design: {lane}",
        f"assumption it rests on: {ASSUMPTION.get(lane, 'unstated')}",
        f"treatment: {treatment}",
        f"outcome: {outcome}",
        f"estimand: {est.estimand}",
        f"estimate: {est.value:.6g}",
        f"95% interval: "
        + (f"{est.ci_low:.6g} to {est.ci_high:.6g}" if est.ci_low is not None else "not available"),
        f"rows used: {est.n}",
        f"claim strength: {strength}",
    ]
    if est.p_value is not None:
        lines.append(f"p-value: {est.p_value:.4g}")
    lines += [f"note: {n}" for n in est.notes]
    for f in findings or []:
        lines.append(f"check {f.get('check')}: {f.get('verdict')} — {f.get('detail')}")
    return "\n".join(lines)


PROMPT = """Write a short plain-English readout of this causal analysis for
someone who is not a statistician.

{facts}

Rules:
- Do not restate any number differently from how it appears above. If you cite
  a number, copy it.
- The claim strength above is fixed. Do not upgrade it, and do not hedge a
  moderate finding into nothing.
- Name the assumption in your own words and say what would break it.
- Any check listed above that failed is the most important thing on this page.
  Say so plainly in the second paragraph rather than burying it.
- Three short paragraphs: what was found, how much to trust it, what would
  change the answer.
- No headings, no bullet points, no preamble."""


def narrate(lane: str, est: Estimate, treatment: str, outcome: str,
            strength: str = "", findings: list | None = None) -> str:
    """The model's only job: prose around numbers it cannot alter."""
    from google import genai

    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    text = client.models.generate_content(
        model=MODEL,
        contents=PROMPT.format(
            facts=facts(lane, est, treatment, outcome, strength, findings)),
    ).text.strip()

    hit = next((w for w in FORBIDDEN if w in text.lower()), None)
    if hit:
        return (
            f"[narration withheld: it used the word '{hit}', which overstates "
            f"what this design can support]\n\n"
            + headline(lane, est, treatment, outcome, strength)
        )
    return text

"""The report language guard.

A causal report is allowed to discuss the causal DAG, identification, and the
causal effect at the strength the claim critic permits, so this guard does NOT
reuse the EDA agent's blanket "never say cause" ban (which would gut exactly
that narrative). What a report must never do is overclaim, and the claim critic
already encodes that as `claim_critique.forbidden_language` (e.g. "proves",
"definitely causes"), populated from the allowed strength. So the guard keys on
that list. Preferred phrasing (`allowed_language`) is steered by the prompt;
this function enforces the hard floor.
"""
from __future__ import annotations

from src.analysis_v2.spec import ClaimCritique


def forbidden_hit(text: str, critique: ClaimCritique | None) -> str | None:
    """Return the first forbidden phrase present in `text`, or None if clean."""
    if not text or critique is None:
        return None
    lowered = text.lower()
    for phrase in critique.forbidden_language:
        if phrase and phrase.lower() in lowered:
            return phrase
    return None


def passes_report_guard(text: str, critique: ClaimCritique | None) -> bool:
    """True when `text` uses no phrasing the claim critic forbids."""
    return forbidden_hit(text, critique) is None

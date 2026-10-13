"""The terminal run status, derived from deficiencies, stops, and execution records."""

from __future__ import annotations


def status_of(state: dict) -> str:
    if state.get("context_deficiency"):
        return "context_deficient"
    estimates = state.get("estimates") or []
    successful = [e for e in estimates if not (e.get("error") if isinstance(e, dict) else e.error)]
    if state.get("feasibility"):
        return "partial" if successful else "infeasible"
    if estimates and not successful:
        return "execution_failed"
    if any(e.get("error") if isinstance(e, dict) else e.error for e in estimates):
        return "partial"
    return "done"

"""What every lane leaves behind: artifacts.json with the common keys, the specialist_result the desk reads, and the
report's tail (explicit disagreements and essential context deficiencies)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causal_agent.common import trace
from causal_agent.common.contracts import Decline
from causal_agent.lane.terminal import status_of

COMMON = (
    "designs",
    "lead_design_id",
    "lead_selection",
    "ledger",
    "candidates",
    "candidate_history",
    "reasoning_history",
    "checks",
    "check_facts",
    "declines",
    "context_deficiency",
    "estimates",
    "falsifications",
    "sensitivities",
    "comparisons",
    "execution_attempts",
    "interpretations",
    "feasibility",
    "figures",
    "ladder",
    "episodes",
)


def _dump(v: Any) -> Any:
    if hasattr(v, "model_dump"):
        return v.model_dump()
    if isinstance(v, list):
        return [_dump(x) for x in v]
    if isinstance(v, dict):
        return {k: _dump(x) for k, x in v.items()}
    return v


def artifacts(state: dict, extra: dict | None = None) -> dict:
    out = {k: _dump(state.get(k)) for k in COMMON}
    out["checks"] = _dump(state.get("checks") or [])
    out["declines"] = _dump(state.get("declines") or [])
    out.update(_dump(extra or {}))
    return out


def write(run_dir: str | None, arts: dict, report: str) -> None:
    if not run_dir:
        return
    trace.write(Path(run_dir, "report.md"), report)
    trace.write(Path(run_dir, "artifacts.json"), json.dumps(arts, indent=2, default=str))


def result(state: dict, report: str, extra: dict | None = None) -> dict:
    h = state["handoff"]
    out = {
        "status": status_of(state),
        "context_deficiency": _dump(state.get("context_deficiency")),
        "family": h.family,
        "specialist": h.specialist,
        "run_dir": state.get("run_dir"),
        "report": report,
        "designs": _dump(state.get("designs") or {}),
        "lead_design_id": state.get("lead_design_id"),
        "lead_selection": _dump(state.get("lead_selection") or {}),
        "context_id": h.context_id,
        "dataset_identity": h.dataset_identity,
        "context_admission": _dump(h.admission),
        "context_facts": _dump(h.facts),
        "episodes": _dump(state.get("episodes") or {}),
        "ledger": _dump(state.get("ledger")),
        "candidates": _dump(state.get("candidates") or []),
        "candidate_history": _dump(state.get("candidate_history") or []),
        "reasoning_history": _dump(state.get("reasoning_history") or []),
        "comparisons": _dump(state.get("comparisons") or []),
        "execution_attempts": _dump(state.get("execution_attempts") or []),
        "checks": _dump(state.get("checks") or []),
        "declines": _dump(state.get("declines") or []),
        "figures": [f.get("id") for f in state.get("figures") or [] if isinstance(f, dict)],
        "estimates": _dump(state.get("estimates") or []),
        "falsifications": _dump(state.get("falsifications") or []),
        "sensitivities": _dump(state.get("sensitivities") or []),
        "interpretations": _dump(state.get("interpretations") or []),
        "feasibility": _dump(state.get("feasibility")),
        "terminal": {
            "kind": "context_deficiency"
            if state.get("context_deficiency")
            else getattr(state.get("feasibility"), "terminal_kind", None)
            if state.get("feasibility")
            else "execution_failure"
            if status_of(state) == "execution_failed"
            else "valid_design",
            "status": status_of(state),
        },
    }
    out.update(_dump(extra or {}))
    return out


def report_tail(state: dict) -> list[str]:
    lines: list[str] = []
    declines: list[Decline] = [d if isinstance(d, Decline) else Decline.model_validate(d) for d in state.get("declines") or []]
    if declines:
        lines += ["", "DISAGREEMENTS WITH THE PACK"]
        lines += [f"  {d.render()}" for d in declines]
    deficiency = state.get("context_deficiency")
    if deficiency:
        raw = _dump(deficiency)
        lines += ["", "CONTEXT DEFICIENCY", raw["reason"], ", ".join(raw["addresses"])]
    return lines

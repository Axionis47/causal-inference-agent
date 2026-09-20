"""What every lane leaves behind: artifacts.json with the common keys, the specialist_result the desk reads, and the
report's tail (where the lane disagreed with the pack, and what it asks back)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causal_agent.common.contracts import Decline
from causal_agent.lane.asks import ask_dict, status_of

COMMON = ("design", "checks", "check_facts", "declines", "case", "ask", "estimates", "refutations", "interpretations", "feasibility", "figures")


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
    out["ask"] = ask_dict(state.get("ask"))
    out["checks"] = _dump(state.get("checks") or [])
    out["declines"] = _dump(state.get("declines") or [])
    out.update(_dump(extra or {}))
    return out


def write(run_dir: str | None, arts: dict, report: str) -> None:
    if not run_dir:
        return
    Path(run_dir, "report.md").write_text(report)
    Path(run_dir, "artifacts.json").write_text(json.dumps(arts, indent=2, default=str))


def result(state: dict, report: str, extra: dict | None = None) -> dict:
    h = state["handoff"]
    d = state.get("design")
    out = {
        "status": status_of(state),
        "ask": ask_dict(state.get("ask")),
        "family": h.family,
        "specialist": h.specialist,
        "run_dir": state.get("run_dir"),
        "report": report,
        "design": _dump(d) if d is not None else None,
        "checks": _dump(state.get("checks") or []),
        "declines": _dump(state.get("declines") or []),
        "figures": [f.get("id") for f in state.get("figures") or [] if isinstance(f, dict)],
        "estimates": _dump(state.get("estimates") or []),
        "refutations": _dump(state.get("refutations") or []),
        "interpretations": _dump(state.get("interpretations") or []),
        "feasibility": _dump(state.get("feasibility")),
    }
    out.update(_dump(extra or {}))
    return out


def report_tail(state: dict) -> list[str]:
    lines: list[str] = []
    declines: list[Decline] = [d if isinstance(d, Decline) else Decline.model_validate(d) for d in state.get("declines") or []]
    if declines:
        lines += ["", "DISAGREEMENTS WITH THE PACK"]
        lines += [f"  {d.render()}" for d in declines]
    a = ask_dict(state.get("ask"))
    if a:
        lines += ["", f"ASKS BACK    [{a.get('address')}] {a.get('question')}" + (f"  because {a['because']}" if a.get("because") else "")]
    return lines

"""Build the EDA tool registry the ReAct loop offers.

Each tool is a thin, static wrapper around one existing deterministic check.
The set OFFERED is dynamic: the lane/DAG-eligible recipe minus the base checks
(which always run as the floor), so the model only ever sees tools that fit
this design. A handler validates args against the causal model, runs the check,
records the call on the ledger, and returns a one-line observation. Nothing here
mutates run state.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.analysis_v2.agents.base import LoopTool
from src.analysis_v2.spec import EDACheck, EDAToolCall, EDAToolTrace, MethodLane

from . import checks_base
from .common import ArtifactSink, CheckFn, EDAInputs
from .constraints import Verdict, clamp_covariates
from .recipes import build_recipe
from .tool_specs import NONE_SCHEMA, TOOL_DESCRIPTIONS, TOOL_SCHEMAS

# Tools whose column arguments the causal model constrains. Advisory until the
# checks accept tuned arguments (the deferred parameter-pass-through step).
_CONSTRAINTS: dict[str, Callable[[dict[str, Any], EDAInputs], Verdict]] = {
    "covariate_balance": clamp_covariates,
    "propensity_overlap": clamp_covariates,
}


@dataclass
class LedgerEntry:
    call: EDAToolCall
    check: EDACheck | None


def _wrap(
    check_fn: CheckFn, inputs: EDAInputs, sink: ArtifactSink, ledger: list[LedgerEntry]
) -> LoopTool:
    name = check_fn.__name__
    constraint = _CONSTRAINTS.get(name)

    def handler(**kwargs: Any) -> str:
        note = ""
        if constraint is not None:
            verdict = constraint(kwargs, inputs)
            if verdict.rejected:
                ledger.append(LedgerEntry(
                    EDAToolCall(name=name, args=kwargs, status="rejected",
                                rejected=verdict.reason),
                    None,
                ))
                return f"rejected: {verdict.reason}"
            kwargs = verdict.clamped_args
            note = f" ({verdict.reason})" if verdict.reason else ""
        check = check_fn(inputs, sink)
        ledger.append(LedgerEntry(
            EDAToolCall(name=name, args=kwargs, status=check.status.value), check,
        ))
        observation = f"{check.name} [{check.status.value}] {check.detail} {check.metrics or ''}"
        return observation.strip() + note

    return LoopTool(
        name=f"run_{name}",
        description=TOOL_DESCRIPTIONS.get(name, f"Run the {name} check."),
        parameters=TOOL_SCHEMAS.get(name, NONE_SCHEMA),
        handler=handler,
    )


def build_eda_tools(
    inputs: EDAInputs, sink: ArtifactSink, lane: MethodLane | None, has_dag: bool
) -> tuple[list[LedgerEntry], list[LoopTool]]:
    """(ledger, tools). The ledger fills as handlers run; tools are the
    targeted/eligible checks (the base checks run separately as the floor)."""
    ordered, _ = build_recipe(inputs.spec, lane, has_dag=has_dag)
    floor = set(checks_base.DATA_HEALTH_CHECKS)  # the floor runs separately
    offered = [fn for fn in ordered if fn not in floor]
    ledger: list[LedgerEntry] = []
    tools = [_wrap(fn, inputs, sink, ledger) for fn in offered]
    return ledger, tools


def collect_checks(ledger: list[LedgerEntry]) -> list[EDACheck]:
    """The EDAChecks the loop produced, de-duplicated by name (first wins)."""
    seen: set[str] = set()
    checks: list[EDACheck] = []
    for entry in ledger:
        if entry.check is not None and entry.call.name not in seen:
            seen.add(entry.call.name)
            checks.append(entry.check)
    return checks


def collect_trace(ledger: list[LedgerEntry], *, exhausted: bool = False) -> EDAToolTrace:
    return EDAToolTrace(calls=[entry.call for entry in ledger], exhausted=exhausted)

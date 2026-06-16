"""Deterministic assembly proposals and the trace synthesizer.

`propose_single_file` is the fallback the agent returns when there is no tool
support (today's behavior, made explicit). `synthesize_trace` turns any plan
into an ordered tool trace so the deterministic path still has a replay record,
mirroring how targeted_eda's commit synthesizes a trace from the checks that
ran.
"""
from __future__ import annotations

from src.analysis_v2.spec import AssemblyPlan, AssemblyToolCall, AssemblyToolTrace
from src.domain.dataset_manifest import DatasetManifest


def propose_single_file(manifest: DatasetManifest) -> AssemblyPlan:
    """The single winner file, no assembly: exactly today's loader behavior."""
    return AssemblyPlan.single_file(manifest.winner or "")


def synthesize_trace(plan: AssemblyPlan) -> AssemblyToolTrace:
    """An ordered trace derived from the plan, for the deterministic path (or
    an agentic run that produced no trace of its own)."""
    calls: list[AssemblyToolCall] = []
    for name in plan.concat_files:
        calls.append(
            AssemblyToolCall(name="concat", kind="concat",
                             args={"file": name}, status="ok")
        )
    for join in plan.joins:
        calls.append(
            AssemblyToolCall(name="join", kind="join", status="ok",
                             args={"right_file": join.right_file,
                                   "on": join.on, "how": join.how})
        )
    calls.append(
        AssemblyToolCall(name="finish", kind="finish", status="ok",
                         args={"base_file": plan.base_file})
    )
    return AssemblyToolTrace(calls=calls)


def summarize_plan(plan: AssemblyPlan) -> str:
    """One human line for the public summary / SSE headline."""
    if plan.is_trivial:
        return f"Single dataset `{plan.base_file}`; no assembly needed."
    parts = [f"base `{plan.base_file}`"]
    if plan.concat_files:
        parts.append(f"stacked {len(plan.concat_files)} same-schema file(s)")
    for join in plan.joins:
        parts.append(f"{join.how}-joined `{join.right_file}` on {join.on}")
    if plan.drop_columns:
        parts.append(f"dropped {len(plan.drop_columns)} column(s)")
    return "Assembled the analyzable frame: " + "; ".join(parts) + "."

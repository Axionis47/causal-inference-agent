"""One stage's execution: bookkeeping, commit, transition, persist, emit.

The orchestrator-side discipline lives here so agents stay simple: an
AgentRun record brackets the dispatch, the typed output is committed by
the agent's own commit(), the transition event is recorded with the gate
result, the run state is persisted after every stage, and the SSE buffer
gets headline-bearing events the existing terminal UI can render.
"""
from __future__ import annotations

import structlog

from src.analysis_v2.agents.base import AgentCtx, AnalysisAgent
from src.analysis_v2.core import (
    AgentRun,
    AgentRunStatus,
    AnalysisRunState,
    GateStatus,
)
from src.analysis_v2.persistence import save_run

logger = structlog.get_logger(__name__)


def _headline(text: str | None, fallback: str) -> str:
    if not text:
        return fallback
    first = text.strip().splitlines()[0]
    return first[:160] if first else fallback


async def run_stage(agent: AnalysisAgent, ctx: AgentCtx) -> GateStatus:
    """Execute one agent, commit its output, and persist. Returns the gate."""
    run = ctx.run
    record = AgentRun(agent=agent.name, stage=agent.stage)
    run.add_agent_run(record)
    record.start()
    ctx.emit(
        "analysis_stage_started",
        {
            "stage": agent.stage.value,
            "agent": agent.name,
            "headline": f"{agent.name} started",
        },
    )
    await save_run(run)

    try:
        result = await agent.execute(ctx)
    except Exception as exc:
        logger.error("agent_crashed", agent=agent.name, job_id=run.job_id, error=str(exc))
        record.finish(AgentRunStatus.FAILED)
        record.public_summary = f"{agent.name} crashed: {exc}"
        run.mark_failed(f"{agent.name} crashed: {exc}")
        await save_run(run)
        ctx.emit(
            "analysis_failed",
            {"headline": f"{agent.name} crashed", "error": str(exc)},
        )
        return GateStatus.FAIL

    record.public_summary = result.public_summary or None
    record.warnings = list(result.warnings)
    record.artifact_ids = list(result.artifact_ids)
    record.tool_calls = list(result.tool_calls)
    record.tokens = result.tokens
    record.cost_usd = result.cost_usd
    record.gate_result = result.gate
    record.failure = result.failure

    gate = result.gate
    if gate.status == GateStatus.FAIL:
        record.finish(AgentRunStatus.FAILED)
        run.mark_failed("; ".join(gate.hard_failures) or f"{agent.name} failed")
        await save_run(run)
        ctx.emit(
            "analysis_failed",
            {
                "headline": f"{agent.name} failed: "
                + _headline("; ".join(gate.hard_failures), "hard gate failure"),
                "error": "; ".join(gate.hard_failures),
            },
        )
        return gate.status

    if result.output is not None:
        agent.commit(run, result.output)
    record.finish(
        AgentRunStatus.WARNING if result.warnings else AgentRunStatus.PASSED
    )
    run.record_transition(
        to_state=agent.stage,
        agent_name=agent.name,
        gate_result=gate,
        output_artifacts=list(result.artifact_ids),
        warnings=list(result.warnings),
        tokens=result.tokens,
        cost_usd=result.cost_usd,
    )
    await save_run(run)
    ctx.emit(
        "analysis_agent_completed",
        {
            "stage": agent.stage.value,
            "agent": agent.name,
            "status": record.status.value,
            "headline": _headline(result.public_summary, f"{agent.name} done"),
            "warnings": list(result.warnings),
            "artifact_ids": list(result.artifact_ids),
            "tokens": result.tokens.model_dump(),
            "cost_usd": result.cost_usd,
        },
    )
    return gate.status

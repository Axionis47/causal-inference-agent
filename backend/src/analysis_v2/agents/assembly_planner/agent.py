"""AssemblyPlannerAgent (S0A): decide how the bundle becomes one frame.

A Kaggle submission can be several files. When the LLM has tool support and
the bundle holds more than one file, a ReAct loop lets the model inspect the
files and propose joins/concats, producing a typed AssemblyPlan the loader and
the notebook both execute. With no tool support, a single-file bundle, or any
loop failure, the agent returns the single-winner plan, which is exactly
today's behavior. The runner rebuilds ctx.frame from the committed plan before
intake; the agent itself never touches the frame.
"""
from __future__ import annotations

import structlog

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent, react_loop
from src.analysis_v2.core import (
    AnalysisRunState,
    AnalysisStage,
    ArtifactKind,
    GateResult,
    TokenUsage,
)
from src.analysis_v2.spec import AssemblyPlan
from src.domain.dataset_manifest import DatasetManifest
from src.llm import get_llm_client
from src.storage.job_data import read_manifest

from .prompt import MAX_ASSEMBLY_TOOLS, MAX_TURNS, SYSTEM_PROMPT, build_mission
from .proposer import propose_single_file, summarize_plan, synthesize_trace
from .tools import build_assembly_tools, collect_plan, collect_trace

logger = structlog.get_logger(__name__)


class AssemblyPlannerAgent(AnalysisAgent):
    name = "assembly_planner"
    stage = AnalysisStage.S0A_DATA_ASSEMBLED

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        manifest = read_manifest(ctx.job_id)
        if manifest is None or manifest.winner is None:
            # No bundle to reason about; the loader already raised if the
            # dataset is truly missing. Advance without an assembly plan.
            return AgentResult(
                gate=GateResult.advance(),
                public_summary="No dataset manifest to assemble; using the frame as loaded.",
            )

        plan, tokens = await self._plan(ctx, manifest)
        artifact_id = self._write_artifact(ctx, plan)
        return AgentResult(
            gate=GateResult.advance(),
            output=plan,
            public_summary=summarize_plan(plan),
            artifact_ids=[artifact_id],
            tokens=tokens,
        )

    async def _plan(
        self, ctx: AgentCtx, manifest: DatasetManifest
    ) -> tuple[AssemblyPlan, TokenUsage]:
        """Run the agentic loop; fall back to the single-winner plan on a
        single-file bundle, no tool support, a provider without tools, or any
        loop failure. A narrative hiccup must never sink a runnable analysis."""
        ledger, tools = build_assembly_tools(manifest, ctx.job_id)
        if not tools:
            # single-file bundle: today's behavior, no LLM constructed at all.
            return propose_single_file(manifest), TokenUsage()
        llm = get_llm_client()
        if not hasattr(llm, "chat_with_tools"):
            return propose_single_file(manifest), TokenUsage()
        try:
            loop = await react_loop(
                llm, prompt=build_mission(manifest, ctx.run, tools),
                system_instruction=SYSTEM_PROMPT, tools=tools, ctx=ctx,
                max_tool_calls=MAX_ASSEMBLY_TOOLS, max_turns=MAX_TURNS,
            )
        except NotImplementedError:
            return propose_single_file(manifest), TokenUsage()
        except Exception as exc:
            logger.warning("assembly_loop_failed", job_id=ctx.run.job_id, error=str(exc))
            return propose_single_file(manifest), TokenUsage()
        plan = collect_plan(ledger, manifest)
        plan.tool_trace = collect_trace(ledger, exhausted=loop.exhausted)
        return plan, loop.tokens

    def _write_artifact(self, ctx: AgentCtx, plan: AssemblyPlan) -> str:
        ctx.add_artifact(
            agent=self.name,
            stage=self.stage,
            artifact_id="assembly/plan",
            kind=ArtifactKind.JSON,
            title="Data assembly plan",
            relative_path="assembly/assembly_plan.json",
            payload=plan.model_dump(mode="json"),
            summary=summarize_plan(plan),
        )
        return "assembly/plan"

    def commit(self, run: AnalysisRunState, output: AssemblyPlan) -> None:
        run.assembly_plan = output
        # The replayable trace: the agentic path carries its own ordered tool
        # calls; the deterministic path derives one from the plan (mirrors
        # targeted_eda's commit synthesizing a trace from the checks that ran).
        run.assembly_tool_trace = output.tool_trace or synthesize_trace(output)

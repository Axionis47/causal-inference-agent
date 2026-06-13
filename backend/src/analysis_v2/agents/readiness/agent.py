"""ReadinessAgent (S6b): guarantee the frozen plan will run before S7.

Deterministic. Calls the selected lane's check_ready, the exact preconditions
the lane raises on at runtime, against the actual frame and the frozen
MethodPlan. A clean result advances; blocking reasons fail the run here with
every reason at once, instead of a first-error crash inside estimation. The
bounded repair lands in a later step.
"""
from __future__ import annotations

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent
from src.analysis_v2.agents.method_lane.lanes import LANE_CHECKS
from src.analysis_v2.core import AnalysisRunState, AnalysisStage, ArtifactKind, GateResult
from src.analysis_v2.spec import ReadinessResult


class ReadinessAgent(AnalysisAgent):
    name = "readiness"
    stage = AnalysisStage.S6B_ANALYSIS_READY

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        plan = ctx.run.method_plan
        spec = ctx.run.causal_spec
        if plan is None or spec is None:
            return AgentResult(
                gate=GateResult.fail(
                    ["readiness needs a frozen method plan and causal spec"]
                ),
                public_summary="Upstream slots are missing; the spine is out of order.",
            )

        check = LANE_CHECKS.get(plan.lane)
        reasons = list(check(ctx.frame, plan, spec)) if check else []
        result = ReadinessResult(
            lane=plan.lane.value, ready=not reasons, blocking_reasons=reasons
        )
        artifact_id = self._write_artifact(ctx, result)

        if reasons:
            return AgentResult(
                gate=GateResult.fail(
                    [
                        f"the {plan.lane.value} lane cannot run on this data: "
                        + "; ".join(reasons)
                    ]
                ),
                output=result,
                public_summary=(
                    f"The {plan.lane.value} plan is not runnable: " + "; ".join(reasons)
                ),
                artifact_ids=[artifact_id],
            )
        return AgentResult(
            gate=GateResult.advance(),
            output=result,
            public_summary=f"The {plan.lane.value} plan is ready to run.",
            artifact_ids=[artifact_id],
        )

    def _write_artifact(self, ctx: AgentCtx, result: ReadinessResult) -> str:
        ctx.add_artifact(
            agent=self.name,
            stage=self.stage,
            artifact_id="readiness/result",
            kind=ArtifactKind.JSON,
            title="Analysis readiness",
            relative_path="readiness/readiness.json",
            payload=result.model_dump(mode="json"),
        )
        return "readiness/result"

    def commit(self, run: AnalysisRunState, output: ReadinessResult) -> None:
        run.readiness = output

"""MethodLaneAgent (S7): run exactly the frozen plan's lane, nothing else.

The lane comes from the confirmed MethodPlan; switching designs here is
forbidden. Lane input errors are hard gate failures with the lane's own
explanation; an unexpected crash gets one retry, then fails.
"""
from __future__ import annotations

import structlog

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent
from src.analysis_v2.core import (
    AgentFailure,
    AnalysisRunState,
    AnalysisStage,
    ArtifactKind,
    FailureType,
    GateResult,
    NextAction,
)
from src.analysis_v2.spec import EstimateResult

from .lanes import LANES, LaneInputError, LaneOutcome

logger = structlog.get_logger(__name__)

MAX_ATTEMPTS = 2


class MethodLaneAgent(AnalysisAgent):
    name = "method_lane"
    stage = AnalysisStage.S7_METHOD_EXECUTED

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        plan = ctx.run.method_plan
        spec = ctx.run.causal_spec
        if plan is None or spec is None:
            return AgentResult(
                gate=GateResult.fail(["method lane needs the confirmed method plan"]),
                public_summary="No confirmed method plan; the plan gate must run first.",
            )
        runner = LANES.get(plan.lane)
        if runner is None:
            return AgentResult(
                gate=GateResult.fail([f"lane {plan.lane.value} has no runner"]),
                public_summary=f"Lane {plan.lane.value} is not implemented.",
            )

        outcome: LaneOutcome | None = None
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                outcome = runner(ctx.frame, plan, spec)
                break
            except LaneInputError as exc:
                return AgentResult(
                    gate=GateResult.fail([str(exc)]),
                    public_summary=f"The {plan.lane.value} lane refused to run: {exc}",
                )
            except Exception as exc:  # transient crash: one retry, then fail
                last_error = exc
                logger.warning(
                    "lane_crashed", lane=plan.lane.value, attempt=attempt, error=str(exc)
                )
        if outcome is None:
            failure = AgentFailure(
                agent=self.name,
                failure_type=FailureType.EXECUTION_ERROR,
                message=f"{plan.lane.value} lane crashed: {last_error}",
                recoverable=False,
                next_action=NextAction.FAIL_JOB,
            )
            return AgentResult(
                gate=GateResult.fail([failure.message]),
                public_summary=failure.message,
                failure=failure,
            )

        artifact_ids = self._write_artifacts(ctx, outcome)
        primary = outcome.result.primary
        public = (
            f"Ran {outcome.result.estimator} on {outcome.result.n_rows_used:,} rows. "
            f"Primary estimate ({primary.estimand}): {primary.estimate:.4g}"
            + (f" (se {primary.std_error:.3g})" if primary.std_error else "")
            + f". {primary.interpretation}."
        )
        return AgentResult(
            gate=GateResult.advance(soft_warnings=list(outcome.warnings)),
            output=outcome.result,
            public_summary=public,
            warnings=list(outcome.warnings),
            artifact_ids=artifact_ids,
        )

    def _write_artifacts(self, ctx: AgentCtx, outcome: LaneOutcome) -> list[str]:
        ids: list[str] = []
        ctx.add_artifact(
            agent=self.name, stage=self.stage,
            artifact_id="method/estimate_result", kind=ArtifactKind.JSON,
            title="Estimate result", relative_path="method/estimate_result.json",
            payload=outcome.result.model_dump(mode="json"),
        )
        ids.append("method/estimate_result")
        for artifact in outcome.artifacts:
            artifact_id = f"method/{artifact.name}"
            if artifact.kind == "table":
                ctx.add_table_artifact(
                    agent=self.name, stage=self.stage, artifact_id=artifact_id,
                    title=artifact.title,
                    relative_path=f"method/tables/{artifact.name}.csv",
                    frame=artifact.payload,
                )
            elif artifact.kind == "plot":
                ctx.add_artifact(
                    agent=self.name, stage=self.stage, artifact_id=artifact_id,
                    kind=ArtifactKind.PLOT, title=artifact.title,
                    relative_path=f"method/plots/{artifact.name}.png",
                    payload=artifact.payload,
                )
            else:
                ctx.add_artifact(
                    agent=self.name, stage=self.stage, artifact_id=artifact_id,
                    kind=ArtifactKind.MARKDOWN, title=artifact.title,
                    relative_path=f"method/{artifact.name}.md",
                    payload=artifact.payload,
                )
            ids.append(artifact_id)
        return ids

    def commit(self, run: AnalysisRunState, output: EstimateResult) -> None:
        run.estimate_result = output

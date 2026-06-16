"""DiagnosticsSensitivityAgent (S8): stress-test the estimate.

Sensitivity changes confidence and language, never the workflow
direction: weak, fragile, or null results advance with their status.
The one stop is a detected bad control / leakage, which fails the run
with an actionable message (re-submit with the column ignored), because
re-running the same plan would reproduce the same contamination.
"""
from __future__ import annotations

import structlog

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent
from src.analysis_v2.agents.method_lane.lanes import LANES
from src.analysis_v2.core import AnalysisRunState, AnalysisStage, ArtifactKind, GateResult
from src.analysis_v2.spec import (
    CheckStatus,
    DiagnosticCheck,
    DiagnosticsResult,
    SensitivityResult,
)

from . import checks as C
from .compose import overall, rubric, run_lane_checks, summary_line

logger = structlog.get_logger(__name__)


class DiagnosticsSensitivityAgent(AnalysisAgent):
    name = "diagnostics_sensitivity"
    stage = AnalysisStage.S8_DIAGNOSTICS_SENSITIVITY_COMPLETE

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        run = ctx.run
        result = run.estimate_result
        plan = run.method_plan
        spec = run.causal_spec
        if result is None or plan is None or spec is None:
            return AgentResult(
                gate=GateResult.fail(["diagnostics need the estimate and the plan"]),
                public_summary="No estimate to stress-test; the method lane must run first.",
            )
        frame = ctx.frame
        runner = LANES[plan.lane]

        leakage = C.detect_leakage(frame, plan)
        diag: list[DiagnosticCheck] = [leakage]
        if leakage.status == CheckStatus.FAIL:
            self._write_artifacts(ctx, DiagnosticsResult(checks=diag, overall=CheckStatus.FAIL,
                                                         summary=leakage.detail), None)
            return AgentResult(
                gate=GateResult.fail(
                    [
                        f"bad control detected: {leakage.detail}; resubmit the job "
                        "with that column in the ignored list"
                    ]
                ),
                public_summary=f"Stopped: {leakage.detail}. The estimate would be "
                "contaminated; resubmit with the column ignored.",
            )

        lane_diag, sens = run_lane_checks(frame, plan, spec, result, runner)
        diag.extend(lane_diag)

        diagnostics = DiagnosticsResult(
            checks=diag,
            overall=overall(diag),
            summary=summary_line(diag, "diagnostic"),
        )
        robustness, reason = rubric(diag, sens)
        sensitivity = SensitivityResult(
            checks=sens, robustness=robustness, confidence_reason=reason
        )
        self._write_artifacts(ctx, diagnostics, sensitivity)

        warnings = [
            f"{c.name}: {c.detail}"
            for c in diag + sens
            if c.status in (CheckStatus.WARNING, CheckStatus.FAIL)
        ]
        public = (
            f"Diagnostics {diagnostics.overall.value}; sensitivity verdict: "
            f"{robustness.value}. {reason}"
        )
        return AgentResult(
            gate=GateResult.advance(soft_warnings=warnings),
            output=(diagnostics, sensitivity),
            public_summary=public,
            warnings=warnings,
            artifact_ids=["diagnostics/report", "diagnostics/sensitivity"],
        )

    def _write_artifacts(self, ctx, diagnostics, sensitivity) -> None:
        ctx.add_artifact(
            agent=self.name, stage=self.stage, artifact_id="diagnostics/report",
            kind=ArtifactKind.JSON, title="Diagnostics report",
            relative_path="diagnostics/diagnostics_report.json",
            payload=diagnostics.model_dump(mode="json"),
        )
        if sensitivity is not None:
            ctx.add_artifact(
                agent=self.name, stage=self.stage, artifact_id="diagnostics/sensitivity",
                kind=ArtifactKind.JSON, title="Sensitivity report",
                relative_path="diagnostics/sensitivity_report.json",
                payload=sensitivity.model_dump(mode="json"),
            )

    def commit(self, run: AnalysisRunState, output) -> None:
        diagnostics, sensitivity = output
        run.diagnostics_result = diagnostics
        run.sensitivity_result = sensitivity

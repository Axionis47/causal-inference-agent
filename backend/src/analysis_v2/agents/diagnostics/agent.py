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
    MethodLane,
    RobustnessStatus,
    SensitivityResult,
)

from . import checks as C

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
        base = result.primary.estimate
        runner = LANES[plan.lane]

        diag: list[DiagnosticCheck] = []
        sens: list[DiagnosticCheck] = []

        leakage = C.detect_leakage(frame, plan)
        diag.append(leakage)
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

        if plan.lane in (MethodLane.OBSERVATIONAL, MethodLane.MATCHING):
            sens.append(C.evalue(result, frame))
            sens.append(C.trimming_stability(frame, plan, spec, base, runner))
            diag.append(C.estimator_agreement(result))
        elif plan.lane == MethodLane.DID:
            diag.append(C.did_pre_trend(frame, plan))
            sens.append(C.trimming_stability(frame, plan, spec, base, runner))
        elif plan.lane == MethodLane.RDD:
            sens.extend(C.rdd_bandwidth_sensitivity(frame, plan, spec, base, runner))
            sens.append(C.rdd_placebo_cutoff(frame, plan, spec, runner))
        elif plan.lane == MethodLane.IV:
            diag.append(C.iv_first_stage_strength(frame, plan))
            diag.append(
                DiagnosticCheck(
                    name="exclusion_restriction",
                    status=CheckStatus.NOT_APPLICABLE,
                    detail="the exclusion restriction is an assumption; no data "
                    "test exists for it",
                )
            )
        elif plan.lane == MethodLane.TIME_SERIES:
            sens.append(C.ts_window_sensitivity(frame, plan, spec, base, runner))
        elif plan.lane == MethodLane.MEDIATION:
            diag.append(
                DiagnosticCheck(
                    name="mediator_timing",
                    status=CheckStatus.WARNING,
                    detail="treatment -> mediator -> outcome ordering is assumed, "
                    "not observed",
                )
            )
            sens.append(C.trimming_stability(frame, plan, spec, base, runner))
        elif plan.lane == MethodLane.SURVIVAL:
            diag.append(C.survival_km_crossing(frame, plan))

        diagnostics = DiagnosticsResult(
            checks=diag,
            overall=self._overall(diag),
            summary=self._summary_line(diag, "diagnostic"),
        )
        robustness, reason = self._rubric(diag, sens)
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

    @staticmethod
    def _overall(checks: list[DiagnosticCheck]) -> CheckStatus:
        statuses = {c.status for c in checks}
        if CheckStatus.FAIL in statuses:
            return CheckStatus.FAIL
        if CheckStatus.WARNING in statuses:
            return CheckStatus.WARNING
        return CheckStatus.PASS

    @staticmethod
    def _summary_line(checks: list[DiagnosticCheck], kind: str) -> str:
        worst = [c for c in checks if c.status in (CheckStatus.WARNING, CheckStatus.FAIL)]
        if not worst:
            return f"all {kind} checks passed"
        return "; ".join(f"{c.name}: {c.detail}" for c in worst[:4])

    @staticmethod
    def _rubric(
        diag: list[DiagnosticCheck], sens: list[DiagnosticCheck]
    ) -> tuple[RobustnessStatus, str]:
        hard_fails = [c for c in diag + sens if c.status == CheckStatus.FAIL]
        warns = [c for c in diag + sens if c.status == CheckStatus.WARNING]
        if hard_fails:
            return (
                RobustnessStatus.NOT_SUPPORTED,
                "a design assumption failed its check: "
                + "; ".join(c.name for c in hard_fails),
            )
        if warns:
            return (
                RobustnessStatus.FRAGILE,
                "the estimate survives but with caveats: "
                + "; ".join(c.name for c in warns[:4]),
            )
        return (
            RobustnessStatus.ROBUST,
            "the estimate is stable across the perturbations tried",
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

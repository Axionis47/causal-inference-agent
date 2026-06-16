"""FlowAuditAgent (S11a): prove the run is internally consistent before it ends.

Deterministic. Three things must hold once everything has run:
1. the estimate adjusted for exactly what the causal model sanctioned (for the
   adjustment-based estimators; designs that legitimately use other columns,
   multi-factor's factors, the interaction's moderator, DiD/RDD/ITS structure,
   are out of scope for this check);
2. the claim strength is consistent with whether the effect is identified, a
   not-identified effect cannot carry a strong or moderate claim;
3. every signal the investigator raised is accounted for (reported, not failed).

A logical inconsistency (1 or 2) fails the run; unaccounted signals (3) are
surfaced as warnings.
"""
from __future__ import annotations

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent
from src.analysis_v2.core import AnalysisRunState, AnalysisStage, ArtifactKind, GateResult
from src.analysis_v2.spec import (
    BACKDOOR_IDENTIFIED_ESTIMATORS,
    ClaimStrength,
    FlowAuditResult,
    FlowSignal,
    FlowSignalStatus,
)


# A FlowSignal label is capped at 300 chars; investigator open questions are
# free text and can exceed that. Clip at the source so a long question is
# surfaced, never crashes the audit.
def _signal(text: str) -> str:
    return text if len(text) <= 300 else text[:297] + "..."


# Estimators whose covariates ARE the DAG's backdoor adjustment set. Others
# (joint_regression's factors, interaction_regression's moderator, DiD/RDD/ITS)
# legitimately use non-adjustment columns, so the covariate check skips them.
ADJUSTMENT_ESTIMATORS = frozenset({
    "regression_adjustment",
    "propensity_matching",
    "two_stage_least_squares",
    "product_of_coefficients",
    "cox_proportional_hazards",
})


class FlowAuditAgent(AnalysisAgent):
    name = "flow_audit"
    stage = AnalysisStage.S11A_FLOW_AUDITED

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        run = ctx.run
        signals: list[FlowSignal] = []
        inconsistencies: list[str] = []
        dag = run.causal_dag
        estimate = run.estimate_result

        # 1. the estimate only adjusted for what the DAG sanctioned.
        if dag is not None and estimate is not None and estimate.estimator in ADJUSTMENT_ESTIMATORS:
            sanctioned = dag.adjustment_set()
            used = set(estimate.covariates_used)
            extra = used - sanctioned
            if extra:
                inconsistencies.append(
                    f"the estimate adjusted for {sorted(extra)}, which the causal "
                    f"model's adjustment set {sorted(sanctioned)} does not sanction"
                )
            for column in sorted(sanctioned):
                signals.append(FlowSignal(
                    signal=f"adjustment covariate '{column}'",
                    status=FlowSignalStatus.USED if column in used else FlowSignalStatus.WAIVED,
                ))

        # 2. a not-identified effect cannot carry a strong/moderate claim, for
        # the designs whose identification is backdoor adjustment.
        if (
            dag is not None
            and estimate is not None
            and estimate.estimator in BACKDOOR_IDENTIFIED_ESTIMATORS
            and run.claim_critique is not None
        ):
            identifiable, reason = dag.is_identifiable()
            if not identifiable and run.claim_critique.strength in (
                ClaimStrength.STRONG, ClaimStrength.MODERATE
            ):
                inconsistencies.append(
                    f"claim strength '{run.claim_critique.strength.value}' is too "
                    f"strong: the effect is not point-identified ({reason})"
                )

        # 3. every open question the investigator raised, accounted for or not.
        unaccounted = 0
        if run.dataset_dossier is not None:
            for question in run.dataset_dossier.open_questions:
                signals.append(FlowSignal(
                    signal=_signal(f"open question: {question}"),
                    status=FlowSignalStatus.UNACCOUNTED,
                    note="raised by the investigator, not resolved in the final claim",
                ))
                unaccounted += 1

        passed = not inconsistencies
        result = FlowAuditResult(
            passed=passed, signals=signals, inconsistencies=inconsistencies
        )
        artifact_id = self._write_artifact(ctx, result)

        if not passed:
            return AgentResult(
                gate=GateResult.fail(inconsistencies),
                output=result,
                public_summary="Information-flow audit failed: " + "; ".join(inconsistencies),
                artifact_ids=[artifact_id],
            )
        warnings = (
            [f"{unaccounted} investigator open question(s) not reflected in the final claim"]
            if unaccounted
            else []
        )
        return AgentResult(
            gate=GateResult.advance(soft_warnings=warnings),
            output=result,
            public_summary="Information flow is consistent across the run.",
            warnings=warnings,
            artifact_ids=[artifact_id],
        )

    def _write_artifact(self, ctx: AgentCtx, result: FlowAuditResult) -> str:
        ctx.add_artifact(
            agent=self.name,
            stage=self.stage,
            artifact_id="flow_audit/result",
            kind=ArtifactKind.JSON,
            title="Information-flow audit",
            relative_path="flow_audit/flow_audit.json",
            payload=result.model_dump(mode="json"),
        )
        return "flow_audit/result"

    def commit(self, run: AnalysisRunState, output: FlowAuditResult) -> None:
        run.flow_audit = output

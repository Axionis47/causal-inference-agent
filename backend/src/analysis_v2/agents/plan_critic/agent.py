"""PlanCriticAgent (S5): validate the plan before anything executes.

Deterministic critic, no modeling tools. Produces the PlanCritique, the
selected design, and the proposed MethodPlan. Gate mapping:
pass_auto_approved -> advance, needs_user_confirmation -> park with the
card, fail_missing_required_info -> hard fail listing the fields.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent
from src.analysis_v2.core import AnalysisRunState, AnalysisStage, ArtifactKind, GateResult
from src.analysis_v2.spec import (
    DesignCandidate,
    MethodPlan,
    PlanCritique,
    PlanGateStatus,
)

from .plan_builder import build_method_plan, select_runnable
from .rules import build_card, confirmation_items, hard_missing, needs_confirmation_reasons


@dataclass
class PlanCriticOutput:
    critique: PlanCritique
    selected_design: DesignCandidate | None
    method_plan: MethodPlan | None


class PlanCriticAgent(AnalysisAgent):
    name = "plan_critic"
    stage = AnalysisStage.S5_PLAN_CRITIQUED

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        run = ctx.run
        spec = run.causal_spec
        if spec is None or run.dataset_profile is None:
            return AgentResult(
                gate=GateResult.fail(["plan critic needs the causal spec and profile"]),
                public_summary="Upstream slots are missing; the spine is out of order.",
            )
        candidates = run.design_candidates

        missing = hard_missing(spec, candidates)
        if missing:
            critique = PlanCritique(
                status=PlanGateStatus.FAIL_MISSING_REQUIRED_INFO,
                missing_required=missing,
                summary="The plan cannot be assembled: required information is missing.",
            )
            output = PlanCriticOutput(critique=critique, selected_design=None, method_plan=None)
            ids = self._write_artifacts(ctx, output)
            return AgentResult(
                gate=GateResult.fail([f"missing required info: {'; '.join(missing)}"]),
                output=output,
                public_summary=critique.summary + " " + "; ".join(missing),
                artifact_ids=ids,
            )

        items = confirmation_items(spec, candidates, run.tool_eligibility)
        reasons = needs_confirmation_reasons(spec, candidates)

        if items or reasons:
            # Confirmation pending: keep the top design and let the user resolve
            # the open items first; runnability is settled after they confirm.
            # The plan freezes a concrete outcome, so with the outcome still
            # open it is derived at S6 from the user's answer instead.
            chosen = candidates[0]
            plan = (
                build_method_plan(spec, chosen, run.dataset_dossier)
                if spec.outcome.resolved
                else None
            )
            repair_notes: list[str] = []
            critique = PlanCritique(
                status=PlanGateStatus.NEEDS_USER_CONFIRMATION,
                reasons=reasons,
                confirmation_card=build_card(spec, candidates, items, reasons),
                summary=f"The {chosen.lane.value} plan is ready but needs your confirmation.",
            )
            gate = GateResult.needs_user(reasons or ["plan confirmation requested"])
        else:
            # Auto-approve: pick the highest-ranked design that actually runs on
            # the data, falling down the ranking when the primary is too thin.
            chosen, plan, repair_notes = select_runnable(
                spec, candidates, run.dataset_dossier, ctx.frame
            )
            critique = PlanCritique(
                status=PlanGateStatus.PASS_AUTO_APPROVED,
                summary=f"The {chosen.lane.value} plan is fully resolved; auto-approved.",
            )
            gate = GateResult.advance()

        output = PlanCriticOutput(
            critique=critique, selected_design=chosen, method_plan=plan
        )
        ids = self._write_artifacts(ctx, output)
        public = critique.summary + (
            f" Estimator: {plan.estimator} on outcome '{plan.outcome}'."
            if plan is not None
            else " The estimator is finalized once the outcome is confirmed."
        )
        return AgentResult(
            gate=gate,
            output=output,
            public_summary=public,
            warnings=list(chosen.warnings) + repair_notes,
            artifact_ids=ids,
        )

    def _write_artifacts(self, ctx: AgentCtx, output: PlanCriticOutput) -> list[str]:
        ids: list[str] = []

        def _add(artifact_id: str, title: str, path: str, payload) -> None:
            ctx.add_artifact(
                agent=self.name, stage=self.stage, artifact_id=artifact_id,
                kind=ArtifactKind.JSON, title=title, relative_path=path, payload=payload,
            )
            ids.append(artifact_id)

        _add("plan/critique", "Plan critique", "plan/plan_critique.json",
             output.critique.model_dump(mode="json"))
        if output.critique.confirmation_card is not None:
            _add("plan/confirmation_card", "Confirmation card",
                 "plan/confirmation_card.json",
                 output.critique.confirmation_card.model_dump(mode="json"))
        if output.method_plan is not None:
            _add("plan/method_plan", "Proposed method plan", "plan/method_plan.json",
                 output.method_plan.model_dump(mode="json"))
        return ids

    def commit(self, run: AnalysisRunState, output: PlanCriticOutput) -> None:
        run.plan_critique = output.critique
        run.selected_design = output.selected_design
        run.method_plan = output.method_plan

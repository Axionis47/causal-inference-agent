"""ClaimCriticAgent (S9): prevent overclaiming, pin the report's language.

Deterministic critic with no modeling tools. The output binds the report
stage: only allowed phrasings, never the forbidden ones, limitations
stated verbatim.
"""
from __future__ import annotations

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent
from src.analysis_v2.core import AnalysisRunState, AnalysisStage, ArtifactKind, GateResult
from src.analysis_v2.spec import BACKDOOR_IDENTIFIED_ESTIMATORS, ClaimCritique

from .rubric import (
    ALLOWED_LANGUAGE,
    FORBIDDEN_LANGUAGE,
    cap_for_latent_confounding,
    claim_strength,
    limitations,
)


class ClaimCriticAgent(AnalysisAgent):
    name = "claim_critic"
    stage = AnalysisStage.S9_CLAIM_CRITIQUED

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        run = ctx.run
        if run.estimate_result is None or run.causal_spec is None:
            return AgentResult(
                gate=GateResult.fail(["claim critic needs the estimate result"]),
                public_summary="No estimate to critique; the method lane must run first.",
            )
        strength, notes = claim_strength(
            run.causal_spec.question_type, run.estimate_result, run.sensitivity_result
        )
        limits = limitations(
            run.causal_spec.question_type, run.estimate_result, run.sensitivity_result
        )
        # A latent confounder on an open backdoor path means a backdoor-adjusted
        # estimate is an association, not the identified effect: cap the claim.
        # Scoped to the backdoor-identified designs; IV/DiD/RDD identify despite
        # such confounding and keep their strength.
        dag = run.causal_dag
        if (
            dag is not None
            and dag.has_latent_confounding()
            and run.estimate_result.estimator in BACKDOOR_IDENTIFIED_ESTIMATORS
        ):
            latents = (
                run.dataset_dossier.suspected_latent_confounders
                if run.dataset_dossier is not None
                else []
            )
            strength, cap_note, cap_limit = cap_for_latent_confounding(strength, latents)
            if cap_note is not None:
                notes.append(cap_note)
            limits = [*limits, cap_limit]
        critique = ClaimCritique(
            strength=strength,
            allowed_language=list(ALLOWED_LANGUAGE),
            forbidden_language=list(FORBIDDEN_LANGUAGE),
            limitations=limits,
            rationale="; ".join(notes)[:2000],
        )
        for artifact_id, title, path, payload in (
            ("claims/critique", "Claim critique", "claims/claim_critique.json",
             critique.model_dump(mode="json")),
            ("claims/allowed_language", "Allowed language", "claims/allowed_language.json",
             {"allowed": critique.allowed_language, "forbidden": critique.forbidden_language}),
            ("claims/limitations", "Limitations", "claims/limitations.json",
             {"limitations": limits}),
        ):
            ctx.add_artifact(
                agent=self.name, stage=self.stage, artifact_id=artifact_id,
                kind=ArtifactKind.JSON, title=title, relative_path=path, payload=payload,
            )
        public = (
            f"Claim strength: {strength.value}. {notes[0] if notes else ''} "
            f"{len(limits)} limitations recorded for the report."
        )
        return AgentResult(
            gate=GateResult.advance(),
            output=critique,
            public_summary=public,
            artifact_ids=["claims/critique", "claims/allowed_language", "claims/limitations"],
        )

    def commit(self, run: AnalysisRunState, output: ClaimCritique) -> None:
        run.claim_critique = output

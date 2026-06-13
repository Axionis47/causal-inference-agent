"""DesignDetectionAgent (S3): spec + profile -> candidates + eligibility.

Fully deterministic. Refines the spec (sole-candidate promotions), runs
the lane rules, ranks candidates, and publishes the eligibility table
that narrows every later agent's tools. No estimation here.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent
from src.analysis_v2.core import AnalysisRunState, AnalysisStage, ArtifactKind, GateResult
from src.analysis_v2.spec import (
    CausalDAG,
    CausalSpec,
    DesignCandidate,
    EligibilityState,
    ToolEligibility,
)

from .candidates import PRIMARY_LANE, build_candidates
from .dag import apply_dag_adjustment
from .resolve import resolve_spec
from .rules import evaluate_all_lanes


@dataclass
class DesignDetectionOutput:
    spec: CausalSpec  # refined: sole-candidate promotions applied
    candidates: list[DesignCandidate]
    eligibility: ToolEligibility
    dag: CausalDAG


def _public_summary(
    spec: CausalSpec,
    candidates: list[DesignCandidate],
    eligibility: ToolEligibility,
    notes: list[str],
) -> str:
    lines: list[str] = []
    primary = PRIMARY_LANE[spec.question_type]
    if candidates:
        lines.append(
            f"For this {spec.question_type.value} question the leading design is "
            f"{candidates[0].design_label} ({candidates[0].confidence.value} confidence)."
        )
    else:
        lines.append(
            f"No design is currently executable for this {spec.question_type.value} "
            f"question; the natural lane ({primary.value}) is missing requirements."
        )
    enabled = [e.lane.value for e in eligibility.lanes if e.state == EligibilityState.ENABLED]
    conditional = [
        f"{e.lane.value} (if {', '.join(e.missing_requirements) or 'confirmed'})"
        for e in eligibility.lanes
        if e.state == EligibilityState.CONDITIONAL
    ]
    disabled = [
        f"{e.lane.value}: {e.reason}"
        for e in eligibility.lanes
        if e.state == EligibilityState.DISABLED
    ]
    lines.append(f"Enabled lanes: {', '.join(enabled) or 'none'}.")
    if conditional:
        lines.append(f"Conditional: {'; '.join(conditional)}.")
    if disabled:
        lines.append("Disabled: " + "; ".join(disabled) + ".")
    if notes:
        lines.append("Deterministic resolutions: " + "; ".join(notes) + ".")
    return "\n".join(lines)


class DesignDetectionAgent(AnalysisAgent):
    name = "design_detection"
    stage = AnalysisStage.S3_DESIGN_CANDIDATES_CREATED

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        spec = ctx.run.causal_spec
        profile = ctx.run.dataset_profile
        if spec is None or profile is None:
            return AgentResult(
                gate=GateResult.fail(
                    ["design detection needs the causal spec and the dataset profile"]
                ),
                public_summary="Upstream slots are missing; the spine is out of order.",
            )

        refined, notes = resolve_spec(spec, profile)
        dag = apply_dag_adjustment(refined, ctx.run.dataset_dossier)
        identifiable, id_reason = dag.is_identifiable()
        eligibility = evaluate_all_lanes(refined, profile)
        candidates = build_candidates(refined, eligibility)
        output = DesignDetectionOutput(
            spec=refined, candidates=candidates, eligibility=eligibility, dag=dag
        )

        artifact_ids = self._write_artifacts(ctx, output)
        summary = _public_summary(refined, candidates, eligibility, notes)
        warnings = [w for c in candidates for w in c.warnings]
        if not identifiable:
            warnings.append(f"effect not point-identified: {id_reason}")
        if not candidates:
            warnings.append("no executable design candidates; plan gate must resolve")
        return AgentResult(
            gate=GateResult.advance(soft_warnings=warnings),
            output=output,
            public_summary=summary,
            warnings=warnings,
            artifact_ids=artifact_ids,
        )

    def _write_artifacts(self, ctx: AgentCtx, output: DesignDetectionOutput) -> list[str]:
        ids: list[str] = []

        def _add(artifact_id: str, title: str, path: str, payload) -> None:
            ctx.add_artifact(
                agent=self.name, stage=self.stage, artifact_id=artifact_id,
                kind=ArtifactKind.JSON, title=title, relative_path=path, payload=payload,
            )
            ids.append(artifact_id)

        _add(
            "design/candidates", "Design candidates", "design/design_candidates.json",
            [c.model_dump(mode="json") for c in output.candidates],
        )
        _add(
            "design/tool_eligibility", "Tool eligibility", "design/tool_eligibility.json",
            output.eligibility.model_dump(mode="json"),
        )
        _add(
            "design/disabled_methods", "Disabled methods with reasons",
            "design/disabled_methods_with_reasons.json",
            [
                {"lane": e.lane.value, "reason": e.reason,
                 "missing_requirements": e.missing_requirements}
                for e in output.eligibility.lanes
                if e.state == EligibilityState.DISABLED
            ],
        )
        _add(
            "design/required_missing_fields", "Required missing fields",
            "design/required_missing_fields.json",
            {
                "missing_info": output.spec.missing_info,
                "conditional_requirements": {
                    e.lane.value: e.missing_requirements
                    for e in output.eligibility.lanes
                    if e.state == EligibilityState.CONDITIONAL
                },
            },
        )
        _add(
            "design/refined_spec", "Refined causal spec", "design/refined_causal_spec.json",
            output.spec.model_dump(mode="json"),
        )
        _add(
            "design/causal_dag", "Causal DAG", "design/causal_dag.json",
            output.dag.model_dump(mode="json"),
        )
        return ids

    def commit(self, run: AnalysisRunState, output: DesignDetectionOutput) -> None:
        run.causal_spec = output.spec
        run.causal_dag = output.dag
        run.design_candidates = output.candidates
        run.tool_eligibility = output.eligibility

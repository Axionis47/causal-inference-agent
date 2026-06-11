"""TargetedEDAAgent (S4): the data's story for the detected design.

Deterministic check execution (recipe chosen by the leading candidate's
lane), one optional LLM call to narrate the computed metrics, and a
deterministic fallback narrative when the LLM is unavailable. No causal
effect claims anywhere: the prompt forbids them and the fallback never
makes them.
"""
from __future__ import annotations

import structlog

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent
from src.analysis_v2.core import AnalysisRunState, AnalysisStage, ArtifactKind, GateResult
from src.analysis_v2.spec import EDACheck, EDACheckStatus, EDAPlan, EDASummary
from src.llm import get_llm_client

from .common import ArtifactSink, EDAInputs, failed
from .recipes import build_recipe

logger = structlog.get_logger(__name__)

STORY_PROMPT = (
    "You are writing the exploratory-data-analysis story for an analyst. "
    "Below are computed checks (name, status, detail, metrics) for a "
    "{question_type} question heading into a {lane} design.\n\n{checks}\n\n"
    "Write 4-7 plain sentences: what the data looks like, which design-"
    "relevant patterns stand out, and what the warnings mean practically. "
    "STRICT RULES: describe patterns only; never claim anything causes "
    "anything; never call a difference an effect; do not invent numbers "
    "not present above."
)


def _fallback_story(summary: EDASummary) -> str:
    ok = sum(1 for c in summary.checks if c.status == EDACheckStatus.OK)
    warn = [c for c in summary.checks if c.status == EDACheckStatus.WARNING]
    skip = [c for c in summary.checks if c.status == EDACheckStatus.SKIPPED]
    lines = [f"{len(summary.checks)} checks ran: {ok} clean, {len(warn)} with warnings."]
    for c in warn[:5]:
        lines.append(f"{c.name}: {c.detail}.")
    if skip:
        lines.append(
            "Skipped (inputs not available yet): "
            + ", ".join(c.name for c in skip) + "."
        )
    return " ".join(lines)


class TargetedEDAAgent(AnalysisAgent):
    name = "targeted_eda"
    stage = AnalysisStage.S4_TARGETED_EDA_COMPLETE

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        run = ctx.run
        if run.causal_spec is None or run.dataset_profile is None:
            return AgentResult(
                gate=GateResult.fail(["targeted eda needs the causal spec and profile"]),
                public_summary="Upstream slots are missing; the spine is out of order.",
            )
        lane = run.design_candidates[0].lane if run.design_candidates else None
        checks_to_run, targeted_names = build_recipe(run.causal_spec, lane)
        plan = EDAPlan(
            target_lane=lane,
            base_checks=[fn.__name__ for fn in checks_to_run if fn.__name__ not in targeted_names],
            targeted_checks=targeted_names,
            rationale=f"recipe for the {lane.value if lane else 'undecided'} lane",
        )

        inputs = EDAInputs(frame=ctx.frame, spec=run.causal_spec, profile=run.dataset_profile)
        sink = ArtifactSink(ctx=ctx, agent=self.name, stage=self.stage)
        checks: list[EDACheck] = []
        for fn in checks_to_run:
            try:
                checks.append(fn(inputs, sink))
            except Exception as exc:  # a buggy check never sinks the stage
                checks.append(failed(fn.__name__, exc))

        usable = next((c for c in checks if c.name == "usable_sample_size"), None)
        summary = EDASummary(
            plan=plan,
            checks=checks,
            usable_sample_size=(
                int(usable.metrics["complete_rows"])
                if usable and "complete_rows" in usable.metrics else None
            ),
            warnings=[
                f"{c.name}: {c.detail}"
                for c in checks
                if c.status in (EDACheckStatus.WARNING, EDACheckStatus.FAILED)
            ],
        )
        summary.story = await self._story(summary, run)

        artifact_ids = self._write_artifacts(ctx, summary)
        artifact_ids.extend(sink.emitted)
        return AgentResult(
            gate=GateResult.advance(soft_warnings=list(summary.warnings)),
            output=summary,
            public_summary=summary.story,
            warnings=list(summary.warnings),
            artifact_ids=artifact_ids,
        )

    async def _story(self, summary: EDASummary, run: AnalysisRunState) -> str:
        rendered = "\n".join(
            f"- {c.name} [{c.status.value}] {c.detail} {c.metrics or ''}"
            for c in summary.checks
        )
        prompt = STORY_PROMPT.format(
            question_type=run.causal_spec.question_type.value,
            lane=summary.plan.target_lane.value if summary.plan.target_lane else "undecided",
            checks=rendered[:8000],
        )
        try:
            response = await get_llm_client().generate(prompt)
            text = getattr(response, "text", None) or str(response)
            text = text.strip()
            if text and "cause" not in text.lower().replace("because", ""):
                return text[:4000]
            if text:  # the model used causal language; fall back rather than ship it
                logger.warning("eda_story_used_causal_language", job_id=run.job_id)
        except Exception as exc:
            logger.warning("eda_story_generation_failed", error=str(exc))
        return _fallback_story(summary)

    def _write_artifacts(self, ctx: AgentCtx, summary: EDASummary) -> list[str]:
        ids: list[str] = []

        def _add(artifact_id: str, kind: ArtifactKind, title: str, path: str, payload):
            ctx.add_artifact(
                agent=self.name, stage=self.stage, artifact_id=artifact_id,
                kind=kind, title=title, relative_path=path, payload=payload,
            )
            ids.append(artifact_id)

        _add("eda/plan", ArtifactKind.JSON, "EDA plan", "eda/eda_plan.json",
             summary.plan.model_dump(mode="json"))
        _add("eda/summary", ArtifactKind.JSON, "EDA summary", "eda/eda_summary.json",
             summary.model_dump(mode="json"))
        _add("eda/warnings", ArtifactKind.JSON, "EDA warnings", "eda/eda_warnings.json",
             {"warnings": summary.warnings})
        _add("eda/story", ArtifactKind.MARKDOWN, "EDA story", "eda/summary.md",
             summary.story)
        return ids

    def commit(self, run: AnalysisRunState, output: EDASummary) -> None:
        run.eda_plan = output.plan
        run.eda_summary = output

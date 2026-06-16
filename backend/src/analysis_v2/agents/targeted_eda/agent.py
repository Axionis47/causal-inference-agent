"""TargetedEDAAgent (S4): the data's story for the detected design.

The base checks always run as the floor. When the LLM has tool support, a
ReAct loop lets the model choose which design-specific checks to run on top,
guided by the causal DAG; the chosen tool calls are recorded for replay. With
no tool support the agent degrades to the deterministic recipe. Either path
narrates the computed metrics with one guarded LLM call and never makes a
causal effect claim: the prompt forbids them and the fallback never makes them.
"""
from __future__ import annotations

import structlog

from src.analysis_v2.agents.base import (
    AgentCtx,
    AgentResult,
    AnalysisAgent,
    react_loop,
)
from src.analysis_v2.core import (
    AnalysisRunState,
    AnalysisStage,
    ArtifactKind,
    GateResult,
    TokenUsage,
    ToolCallRecord,
)
from src.analysis_v2.spec import (
    EDACheck,
    EDACheckStatus,
    EDAPlan,
    EDASummary,
    EDAToolCall,
    EDAToolTrace,
    MethodLane,
)
from src.llm import get_llm_client

from . import checks_base
from .common import ArtifactSink, CheckFn, EDAInputs, failed
from .prompt import MAX_TARGETED_CHECKS, SYSTEM_PROMPT, build_mission
from .recipes import build_recipe
from .tools import build_eda_tools, collect_checks, collect_trace

logger = structlog.get_logger(__name__)

MAX_TURNS = 6

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


def _no_causal_language(text: str) -> bool:
    return bool(text) and "cause" not in text.lower().replace("because", "")


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
        inputs = EDAInputs(
            frame=ctx.frame, spec=run.causal_spec,
            profile=run.dataset_profile, dag=run.causal_dag,
        )
        sink = ArtifactSink(ctx=ctx, agent=self.name, stage=self.stage)
        # The floor: the data-health checks run for every design, agentic or not.
        # The model curates the design-relevant checks on top, within a budget.
        base_checks = self._run_checks(list(checks_base.DATA_HEALTH_CHECKS), inputs, sink)

        llm = get_llm_client()
        ledger, tools = build_eda_tools(inputs, sink, lane, has_dag=run.causal_dag is not None)
        if not tools or not hasattr(llm, "chat_with_tools"):
            return await self._deterministic_fallback(ctx, inputs, sink, lane, base_checks)

        try:
            loop = await react_loop(
                llm, prompt=build_mission(run, tools),
                system_instruction=SYSTEM_PROMPT, tools=tools, ctx=ctx,
                max_tool_calls=MAX_TARGETED_CHECKS, max_turns=MAX_TURNS,
            )
        except NotImplementedError:
            return await self._deterministic_fallback(ctx, inputs, sink, lane, base_checks)

        targeted = collect_checks(ledger)
        if not targeted:
            # the model ran nothing; never ship a floor-only EDA
            return await self._deterministic_fallback(ctx, inputs, sink, lane, base_checks)
        extra = ["EDA hit its tool budget before finishing"] if loop.exhausted else []
        return await self._finish(
            ctx, sink, lane, base_checks + targeted, [c.name for c in targeted],
            tool_calls=loop.tool_calls,
            trace=collect_trace(ledger, exhausted=loop.exhausted),
            tokens=loop.tokens, story_seed=loop.text, extra_warnings=extra,
            rationale="agentic: base floor plus model-selected checks",
        )

    async def _deterministic_fallback(
        self, ctx: AgentCtx, inputs: EDAInputs, sink: ArtifactSink,
        lane: MethodLane | None, base_checks: list[EDACheck],
    ) -> AgentResult:
        ordered, targeted_names = build_recipe(
            inputs.spec, lane, has_dag=inputs.dag is not None
        )
        floor = set(checks_base.DATA_HEALTH_CHECKS)  # already run; do not repeat
        targeted = self._run_checks([fn for fn in ordered if fn not in floor], inputs, sink)
        return await self._finish(
            ctx, sink, lane, base_checks + targeted, targeted_names,
            tool_calls=[], trace=None, tokens=TokenUsage(), story_seed=None,
            extra_warnings=[],
            rationale=f"recipe for the {lane.value if lane else 'undecided'} lane",
        )

    def _run_checks(
        self, fns: list[CheckFn], inputs: EDAInputs, sink: ArtifactSink
    ) -> list[EDACheck]:
        checks: list[EDACheck] = []
        for fn in fns:
            try:
                checks.append(fn(inputs, sink))
            except Exception as exc:  # a buggy check never sinks the stage
                checks.append(failed(fn.__name__, exc))
        return checks

    async def _finish(
        self, ctx: AgentCtx, sink: ArtifactSink, lane: MethodLane | None,
        checks: list[EDACheck], targeted_names: list[str], *,
        tool_calls: list[ToolCallRecord], trace: EDAToolTrace | None,
        tokens: TokenUsage, story_seed: str | None, extra_warnings: list[str],
        rationale: str,
    ) -> AgentResult:
        run = ctx.run
        targeted_set = set(targeted_names)
        plan = EDAPlan(
            target_lane=lane,
            base_checks=[c.name for c in checks if c.name not in targeted_set],
            targeted_checks=list(targeted_names),
            rationale=rationale,
        )
        usable = next((c for c in checks if c.name == "usable_sample_size"), None)
        warnings = [
            f"{c.name}: {c.detail}" for c in checks
            if c.status in (EDACheckStatus.WARNING, EDACheckStatus.FAILED)
        ]
        warnings.extend(extra_warnings)
        summary = EDASummary(
            plan=plan, checks=checks, warnings=warnings,
            usable_sample_size=(
                int(usable.metrics["complete_rows"])
                if usable and "complete_rows" in usable.metrics else None
            ),
        )
        summary.tool_trace = trace
        summary.story, story_tokens = await self._story(summary, run, story_seed)
        artifact_ids = self._write_artifacts(ctx, summary)
        artifact_ids.extend(sink.emitted)
        return AgentResult(
            gate=GateResult.advance(soft_warnings=list(summary.warnings)),
            output=summary, public_summary=summary.story,
            warnings=list(summary.warnings), artifact_ids=artifact_ids,
            tool_calls=tool_calls, tokens=tokens.add(story_tokens),
        )

    async def _story(
        self, summary: EDASummary, run: AnalysisRunState, seed: str | None = None
    ) -> tuple[str, TokenUsage]:
        tokens = TokenUsage()
        # The agentic loop already wrote a grounded summary; reuse it (guarded)
        # rather than spend a second call, but still fall back if it slips.
        if seed and _no_causal_language(seed.strip()):
            return seed.strip()[:4000], tokens
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
            llm = get_llm_client()
            if hasattr(llm, "generate_text_with_usage"):
                text, usage = await llm.generate_text_with_usage(prompt)
                tokens = TokenUsage(
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)),
                )
            else:  # older stubs expose only generate()
                response = await llm.generate(prompt)
                text = getattr(response, "text", None) or str(response)
            text = text.strip()
            if _no_causal_language(text):
                return text[:4000], tokens
            if text:  # the model used causal language; fall back rather than ship it
                logger.warning("eda_story_used_causal_language", job_id=run.job_id)
        except Exception as exc:
            logger.warning("eda_story_generation_failed", error=str(exc))
        return _fallback_story(summary), tokens

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
        if summary.tool_trace is not None:
            _add("eda/tool_calls", ArtifactKind.JSON, "EDA tool calls",
                 "eda/eda_tool_calls.json", summary.tool_trace.model_dump(mode="json"))
        return ids

    def commit(self, run: AnalysisRunState, output: EDASummary) -> None:
        run.eda_plan = output.plan
        run.eda_summary = output
        # The replayable trace: the agentic path carries the model's tool calls
        # (with args); the deterministic path derives it from the checks that ran.
        run.eda_tool_trace = output.tool_trace or EDAToolTrace(
            calls=[EDAToolCall(name=c.name, status=c.status.value) for c in output.checks]
        )

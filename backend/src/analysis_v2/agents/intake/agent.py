"""IntakeAgent (S1): the free-form question becomes a CausalSpec draft.

One structured generation, deterministic boundary validation, and at most
MAX_ATTEMPTS rounds: generation errors retry, column violations get one
repair round listing exactly what was invented, and whatever remains
after that is quarantined (clue + missing_info), never passed through.
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
    TokenUsage,
)
from src.analysis_v2.spec import CausalSpec
from src.llm import get_llm_client

from .prompt import SYSTEM_PROMPT, build_prompt
from .schema import IntakeDraft
from .validate import to_causal_spec

logger = structlog.get_logger(__name__)

MAX_ATTEMPTS = 3


class IntakeAgent(AnalysisAgent):
    name = "intake"
    stage = AnalysisStage.S1_INTAKE_PARSED

    def _columns(self, ctx: AgentCtx) -> list[tuple[str, str, str | None]]:
        """(name, semantic_type, description) from the frame + gate profile."""
        feature_types: dict[str, str] = {}
        descriptions: dict[str, str] = {}
        if ctx.input_state is not None:
            if ctx.input_state.data_profile is not None:
                feature_types = ctx.input_state.data_profile.feature_types
            descriptions = ctx.input_state.dataset_info.kaggle_column_descriptions
        return [
            (
                str(name),
                feature_types.get(str(name), str(ctx.frame[name].dtype)),
                descriptions.get(str(name)),
            )
            for name in ctx.frame.columns
        ]

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        question = (ctx.run.causal_question or "").strip()
        if not question:
            return AgentResult(
                gate=GateResult.fail(["causal question is empty at pickup"]),
                public_summary="No causal question was provided; intake cannot run.",
            )

        columns = self._columns(ctx)
        schema_columns = [name for name, _, _ in columns]
        info = ctx.input_state.dataset_info if ctx.input_state is not None else None
        prompt = build_prompt(
            causal_question=question,
            columns=columns,
            user_context=info.user_provided_context if info else ctx.run.user_context,
            kaggle_description=info.kaggle_description if info else None,
            dataset_name=info.name if info else ctx.run.dataset_name,
        )

        llm = get_llm_client()
        draft: IntakeDraft | None = None
        spec: CausalSpec | None = None
        violations: list[str] = []
        last_error: str | None = None
        tokens = TokenUsage()

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                if hasattr(llm, "generate_structured_with_usage"):
                    draft, usage = await llm.generate_structured_with_usage(
                        prompt=prompt,
                        response_schema=IntakeDraft,
                        system_instruction=SYSTEM_PROMPT,
                    )
                    tokens = tokens.add(
                        TokenUsage(
                            input_tokens=int(usage.get("input_tokens", 0)),
                            output_tokens=int(usage.get("output_tokens", 0)),
                        )
                    )
                else:  # older stubs expose only generate_structured()
                    draft = await llm.generate_structured(
                        prompt=prompt,
                        response_schema=IntakeDraft,
                        system_instruction=SYSTEM_PROMPT,
                    )
            except Exception as exc:  # generation/parse error: retry
                last_error = f"attempt {attempt}: {exc}"
                logger.warning("intake_generation_failed", attempt=attempt, error=str(exc))
                continue

            spec, violations = to_causal_spec(draft, schema_columns)
            if not violations or attempt >= MAX_ATTEMPTS:
                break
            # One repair round: name the inventions, ask for a corrected draft.
            prompt = (
                f"{prompt}\n\nYour previous draft referenced columns that do not "
                f"exist:\n- " + "\n- ".join(violations) + "\nProduce a corrected "
                "draft using only exact column names from the listing, or leave "
                "the field null with a clue."
            )

        if spec is None or draft is None:
            failure = AgentFailure(
                agent=self.name,
                failure_type=FailureType.TOOL_ERROR,
                message=last_error or "intake generation failed",
                recoverable=False,
                next_action=NextAction.FAIL_JOB,
            )
            return AgentResult(
                gate=GateResult.fail([failure.message]),
                public_summary="The question could not be parsed into a causal spec.",
                failure=failure,
                tokens=tokens,
            )

        warnings = list(violations)
        artifact_ids = self._write_artifacts(ctx, spec, draft)
        return AgentResult(
            gate=GateResult.advance(soft_warnings=warnings),
            output=spec,
            public_summary=draft.reasoning_summary,
            warnings=warnings,
            artifact_ids=artifact_ids,
            tokens=tokens,
        )

    def _write_artifacts(
        self, ctx: AgentCtx, spec: CausalSpec, draft: IntakeDraft
    ) -> list[str]:
        ids: list[str] = []

        def _add(artifact_id: str, kind: ArtifactKind, title: str, path: str, payload):
            ctx.add_artifact(
                agent=self.name,
                stage=self.stage,
                artifact_id=artifact_id,
                kind=kind,
                title=title,
                relative_path=path,
                payload=payload,
            )
            ids.append(artifact_id)

        _add(
            "intake/causal_spec",
            ArtifactKind.JSON,
            "Causal spec draft",
            "intake/causal_spec.json",
            spec.model_dump(mode="json"),
        )
        _add(
            "intake/question_type_candidates",
            ArtifactKind.JSON,
            "Question type candidates",
            "intake/question_type_candidates.json",
            {
                "question_type": spec.question_type.value,
                "candidates": [t.value for t in spec.type_candidates],
                "confidence": spec.confidence.value,
            },
        )
        _add(
            "intake/missing_info",
            ArtifactKind.JSON,
            "Missing information",
            "intake/missing_info.json",
            {"missing_info": spec.missing_info},
        )
        _add(
            "intake/summary",
            ArtifactKind.MARKDOWN,
            "Intake summary",
            "intake/summary.md",
            draft.reasoning_summary,
        )
        return ids

    def commit(self, run: AnalysisRunState, output: CausalSpec) -> None:
        run.causal_spec = output

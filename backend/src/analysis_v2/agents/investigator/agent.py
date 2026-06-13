"""InvestigatorAgent (S2a): understand the dataset before any design work.

A ReAct loop over read-only tools, then one structured synthesis into the
DatasetDossier. Role entries naming invented columns are quarantined as
warnings, mirroring intake's boundary. On an LLM without tool support the
agent degrades to a spec-and-profile dossier and says so; the spine never
stalls on missing agency."""
from __future__ import annotations

from dataclasses import dataclass

import structlog

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent, react_loop
from src.analysis_v2.core import (
    AnalysisRunState,
    AnalysisStage,
    ArtifactKind,
    GateResult,
    TokenUsage,
)
from src.analysis_v2.spec import (
    CausalDAG,
    ColumnRole,
    ContextLedgerItem,
    DatasetDossier,
    LedgerStatus,
    RoleLabel,
)
from src.llm import get_llm_client

from .dag import dag_from_dossier
from .prompt import DOSSIER_PROMPT, SYSTEM_PROMPT, build_mission, render_transcript
from .tools import build_tools

logger = structlog.get_logger(__name__)

MAX_TOOL_CALLS = 18
MAX_TURNS = 8


@dataclass
class InvestigatorOutput:
    """The two slots the investigator produces: the human-facing dossier and
    the causal DAG identification reads from."""

    dossier: DatasetDossier
    dag: CausalDAG


def fallback_dossier(ctx: AgentCtx, note: str) -> DatasetDossier:
    """Honest degraded dossier from the spec and profile alone."""
    run = ctx.run
    spec = run.causal_spec
    roles: list[ColumnRole] = []
    if spec and spec.treatment and spec.treatment.column:
        roles.append(ColumnRole(
            column=spec.treatment.column, role=RoleLabel.TREATMENT,
            reason="named by the parsed question",
        ))
    if spec and spec.outcome and spec.outcome.column:
        roles.append(ColumnRole(
            column=spec.outcome.column, role=RoleLabel.OUTCOME,
            reason="named by the parsed question",
        ))
    profile = run.dataset_profile
    for name in (profile.id_like_columns if profile else []):
        roles.append(ColumnRole(
            column=name, role=RoleLabel.IDENTIFIER,
            reason="id-like per the deterministic profile",
        ))
    info = ctx.input_state.dataset_info if ctx.input_state else None
    description = info.kaggle_description if info else None
    provenance = (description or "")[:1500].strip() or "no dataset description available"
    return DatasetDossier(
        provenance=provenance,
        roles=roles,
        quality_notes=[],
        recommended_exclusions=list(profile.id_like_columns) if profile else [],
        context_ledger=[
            ContextLedgerItem(
                assumption="how units came to be treated (assignment mechanism)",
                status=LedgerStatus.NEEDS_USER,
                note="not investigated",
            ),
            ContextLedgerItem(
                assumption="covariates were measured before treatment",
                status=LedgerStatus.ASSUMED_DEFAULT,
                note="not investigated",
            ),
        ],
        open_questions=["How were units assigned to treatment?"],
        summary=f"Dossier built without investigation: {note}.",
        investigated=False,
    )


class InvestigatorAgent(AnalysisAgent):
    name = "investigator"
    stage = AnalysisStage.S2A_DATASET_INVESTIGATED

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        run = ctx.run
        if run.causal_spec is None or run.dataset_profile is None:
            return AgentResult(
                gate=GateResult.fail(["investigator needs the parsed spec and profile"]),
                public_summary="Intake and profiling must run before the investigator.",
            )

        llm = get_llm_client()
        if not hasattr(llm, "chat_with_tools"):
            dossier = fallback_dossier(ctx, "the configured LLM has no tool support")
            return self._finish(ctx, dossier, [], TokenUsage(),
                                warnings=["investigation skipped: no tool support"])

        tools = build_tools(ctx)
        tokens = TokenUsage()
        try:
            loop = await react_loop(
                llm,
                prompt=build_mission(run),
                system_instruction=SYSTEM_PROMPT,
                tools=tools,
                ctx=ctx,
                max_tool_calls=MAX_TOOL_CALLS,
                max_turns=MAX_TURNS,
            )
            tokens = tokens.add(loop.tokens)
        except NotImplementedError:
            dossier = fallback_dossier(ctx, "the configured LLM has no tool support")
            return self._finish(ctx, dossier, [], tokens,
                                warnings=["investigation skipped: no tool support"])

        columns = list(ctx.frame.columns)
        prompt = DOSSIER_PROMPT.format(
            columns=", ".join(columns[:120]),
            transcript=render_transcript(loop.transcript)
            + (f"\n\nFINAL NOTES: {loop.text}" if loop.text else ""),
        )
        dossier: DatasetDossier | None = None
        warnings: list[str] = []
        for attempt in (1, 2):
            try:
                dossier, usage = await llm.generate_structured_with_usage(
                    prompt, DatasetDossier, SYSTEM_PROMPT
                )
                tokens = tokens.add(TokenUsage(
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)),
                ))
            except Exception as exc:
                logger.warning("dossier_synthesis_failed", attempt=attempt, error=str(exc))
                continue
            invented = [r.column for r in dossier.roles if r.column not in columns]
            if not invented:
                break
            # quarantine the inventions; one repair round, then keep the rest
            if attempt == 1:
                prompt += (
                    "\n\nYour previous dossier referenced columns that do not "
                    "exist: " + ", ".join(invented) + ". Use exact names only."
                )
            else:
                dossier.roles = [r for r in dossier.roles if r.column in columns]
                warnings.append(f"dropped invented role columns: {', '.join(invented)}")

        if dossier is None:
            dossier = fallback_dossier(ctx, "dossier synthesis failed")
            warnings.append("dossier synthesis failed; degraded dossier in place")

        if loop.exhausted:
            warnings.append("investigation hit its budget before finishing")
        return self._finish(ctx, dossier, loop.transcript, tokens,
                            warnings=warnings, tool_calls=loop.tool_calls)

    def _finish(
        self, ctx: AgentCtx, dossier: DatasetDossier, transcript: list[dict],
        tokens: TokenUsage, *, warnings: list[str], tool_calls: list | None = None,
    ) -> AgentResult:
        dag = dag_from_dossier(dossier, ctx.run.causal_spec)
        output = InvestigatorOutput(dossier=dossier, dag=dag)
        artifact_ids = self._write_artifacts(ctx, dossier, dag, transcript)
        open_notes = [f"needs user: {q}" for q in dossier.open_questions[:3]]
        return AgentResult(
            gate=GateResult.advance(soft_warnings=[*warnings, *open_notes]),
            output=output,
            public_summary=dossier.summary,
            warnings=[*warnings, *open_notes],
            artifact_ids=artifact_ids,
            tool_calls=tool_calls or [],
            tokens=tokens,
        )

    def _write_artifacts(
        self, ctx: AgentCtx, dossier: DatasetDossier, dag: CausalDAG,
        transcript: list[dict],
    ) -> list[str]:
        ids: list[str] = []
        ctx.add_artifact(
            agent=self.name, stage=self.stage,
            artifact_id="investigator/dossier", kind=ArtifactKind.JSON,
            title="Dataset dossier", relative_path="investigator/dossier.json",
            payload=dossier.model_dump(mode="json"),
            summary="Provenance, column roles, and the context ledger.",
        )
        ids.append("investigator/dossier")
        ctx.add_artifact(
            agent=self.name, stage=self.stage,
            artifact_id="investigator/causal_dag", kind=ArtifactKind.JSON,
            title="Causal DAG", relative_path="investigator/causal_dag.json",
            payload=dag.model_dump(mode="json"),
            summary="Nodes, edges, and the treatment/outcome under study.",
        )
        ids.append("investigator/causal_dag")
        ctx.add_artifact(
            agent=self.name, stage=self.stage,
            artifact_id="investigator/transcript", kind=ArtifactKind.JSON,
            title="Investigation transcript",
            relative_path="investigator/transcript.json",
            payload={"steps": transcript},
            summary="Every tool call and observation, in order.",
        )
        ids.append("investigator/transcript")
        ctx.add_artifact(
            agent=self.name, stage=self.stage,
            artifact_id="investigator/summary", kind=ArtifactKind.MARKDOWN,
            title="Investigator summary", relative_path="investigator/summary.md",
            payload=dossier.summary,
        )
        ids.append("investigator/summary")
        return ids

    def commit(self, run: AnalysisRunState, output: InvestigatorOutput) -> None:
        run.dataset_dossier = output.dossier
        run.causal_dag = output.dag

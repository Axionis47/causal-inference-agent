"""ProfilingAgent (S2): deterministic dataset profile with displayable artifacts.

No method selection, no conclusions. It states facts: shape, types,
missingness, duplicates, id-like and constant columns, time candidates.
The gate fails only when there is nothing to analyze; quality concerns
ride as soft warnings for the design and plan stages.
"""
from __future__ import annotations

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent
from src.analysis_v2.core import AnalysisRunState, AnalysisStage, ArtifactKind, GateResult
from src.analysis_v2.spec import ProfileSummary

from .plots import missingness_png, numeric_distributions_png
from .tools import build_profile_summary, column_inventory_frame

PREVIEW_ROWS = 50


def _public_summary(summary: ProfileSummary) -> str:
    lines = [
        f"Profiled {summary.n_rows:,} rows x {summary.n_columns} columns.",
    ]
    by_type: dict[str, int] = {}
    for c in summary.columns:
        by_type[c.semantic_type] = by_type.get(c.semantic_type, 0) + 1
    lines.append(
        "Column types: "
        + ", ".join(f"{count} {name}" for name, count in sorted(by_type.items()))
        + "."
    )
    if summary.id_like_columns:
        lines.append(
            f"Identifier-like columns excluded from analysis candidates: "
            f"{', '.join(summary.id_like_columns)}."
        )
    if summary.likely_time_columns:
        lines.append(f"Possible time columns: {', '.join(summary.likely_time_columns)}.")
    if summary.warnings:
        lines.append("Data quality notes: " + "; ".join(summary.warnings) + ".")
    else:
        lines.append("No data quality concerns found.")
    return "\n".join(lines)


class ProfilingAgent(AnalysisAgent):
    name = "profiling"
    stage = AnalysisStage.S2_PROFILE_CREATED

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        df = ctx.frame
        if df.empty or len(df.columns) == 0:
            return AgentResult(
                gate=GateResult.fail(["dataset has no rows or no columns"]),
                public_summary="The loaded dataset is empty; nothing to profile.",
            )

        summary = build_profile_summary(df)
        artifact_ids: list[str] = []

        def _register(artifact_id: str, **kwargs) -> None:
            ctx.add_artifact(
                agent=self.name, stage=self.stage, artifact_id=artifact_id, **kwargs
            )
            artifact_ids.append(artifact_id)

        _register(
            "profiling/dataset_profile",
            kind=ArtifactKind.JSON,
            title="Dataset profile",
            relative_path="profiling/dataset_profile.json",
            payload=summary.model_dump(mode="json"),
            summary=f"{summary.n_rows} rows, {summary.n_columns} columns",
        )
        ctx.add_table_artifact(
            agent=self.name,
            stage=self.stage,
            artifact_id="profiling/column_inventory",
            title="Column inventory",
            relative_path="profiling/column_inventory.csv",
            frame=column_inventory_frame(summary),
        )
        artifact_ids.append("profiling/column_inventory")
        ctx.add_table_artifact(
            agent=self.name,
            stage=self.stage,
            artifact_id="profiling/preview_table",
            title=f"First {min(PREVIEW_ROWS, len(df))} rows",
            relative_path="profiling/preview_table.csv",
            frame=df.head(PREVIEW_ROWS),
        )
        artifact_ids.append("profiling/preview_table")

        miss_png = missingness_png(summary)
        if miss_png is not None:
            _register(
                "profiling/missingness",
                kind=ArtifactKind.PLOT,
                title="Missing data by column",
                relative_path="profiling/missingness.png",
                payload=miss_png,
            )
        hist_png = numeric_distributions_png(df, summary)
        if hist_png is not None:
            _register(
                "profiling/numeric_distributions",
                kind=ArtifactKind.PLOT,
                title="Numeric column distributions",
                relative_path="profiling/numeric_distributions.png",
                payload=hist_png,
            )

        public_summary = _public_summary(summary)
        _register(
            "profiling/summary",
            kind=ArtifactKind.MARKDOWN,
            title="Profiling summary",
            relative_path="profiling/summary.md",
            payload=public_summary,
        )

        return AgentResult(
            gate=GateResult.advance(soft_warnings=list(summary.warnings)),
            output=summary,
            public_summary=public_summary,
            warnings=list(summary.warnings),
            artifact_ids=artifact_ids,
        )

    def commit(self, run: AnalysisRunState, output: ProfileSummary) -> None:
        run.dataset_profile = output

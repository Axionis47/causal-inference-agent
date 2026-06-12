"""Read-only investigation tools over the loaded frame and the pickup
context. Every handler returns a short string observation; nothing here
mutates anything. Output sizes are capped so the loop stays cheap."""
from __future__ import annotations

import pandas as pd

from src.analysis_v2.agents.base import AgentCtx, LoopTool

_MAX_COLUMNS = 80
_MAX_VALUES = 12


def _column_or_error(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return f"no column named '{column}'; use list_columns for exact names"
    return None


def build_tools(ctx: AgentCtx) -> list[LoopTool]:
    frame = ctx.frame
    info = ctx.input_state.dataset_info if ctx.input_state else None

    def dataset_context() -> str:
        parts: list[str] = []
        if info is not None and info.kaggle_description:
            parts.append("KAGGLE DESCRIPTION:\n" + info.kaggle_description[:4000])
        if info is not None and getattr(info, "user_provided_context", None):
            parts.append("USER CONTEXT:\n" + info.user_provided_context[:2000])
        if info is not None and getattr(info, "column_descriptions", None):
            parts.append(f"COLUMN DESCRIPTIONS:\n{info.column_descriptions}"[:3000])
        return "\n\n".join(parts) or "no dataset description or user context available"

    def bundle_files() -> str:
        if info is None or not info.files:
            return "single-file dataset; no sibling files in the bundle"
        lines = [
            f"- {f.name} ({f.format}, {f.size_bytes:,} bytes)"
            + (" [loaded as the analysis frame]" if f.used else "")
            for f in info.files[:20]
        ]
        return "files in the downloaded bundle:\n" + "\n".join(lines)

    def list_columns() -> str:
        lines = []
        for name in list(frame.columns)[:_MAX_COLUMNS]:
            series = frame[name]
            missing = float(series.isna().mean())
            lines.append(
                f"- {name}: dtype={series.dtype}, distinct={series.nunique()}, "
                f"missing={missing:.1%}"
            )
        extra = len(frame.columns) - _MAX_COLUMNS
        if extra > 0:
            lines.append(f"... and {extra} more columns")
        return f"{len(frame):,} rows x {len(frame.columns)} columns\n" + "\n".join(lines)

    def column_stats(column: str) -> str:
        error = _column_or_error(frame, column)
        if error:
            return error
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= max(20, 0.5 * len(series)):
            desc = numeric.describe()
            return (
                f"{column} (numeric view): mean={desc['mean']:.4g}, std={desc['std']:.4g}, "
                f"min={desc['min']:.4g}, 25%={desc['25%']:.4g}, 50%={desc['50%']:.4g}, "
                f"75%={desc['75%']:.4g}, max={desc['max']:.4g}, "
                f"missing={series.isna().mean():.1%}"
            )
        top = series.astype(str).value_counts().head(8)
        rendered = ", ".join(f"{v}: {c}" for v, c in top.items())
        return f"{column} (categorical view): top values {rendered}"

    def value_samples(column: str) -> str:
        error = _column_or_error(frame, column)
        if error:
            return error
        values = frame[column].dropna().unique()[:_MAX_VALUES]
        return f"{column}: " + ", ".join(str(v)[:48] for v in values)

    none_schema = {"type": "object", "properties": {}}
    column_schema = {
        "type": "object",
        "properties": {"column": {"type": "string", "description": "exact column name"}},
        "required": ["column"],
    }
    return [
        LoopTool(
            "dataset_context",
            "The Kaggle description, user-provided context, and column notes.",
            none_schema, dataset_context,
        ),
        LoopTool(
            "bundle_files",
            "Every file in the downloaded bundle and which one is loaded.",
            none_schema, bundle_files,
        ),
        LoopTool(
            "list_columns",
            "All columns with dtype, distinct count, and missing share.",
            none_schema, list_columns,
        ),
        LoopTool(
            "column_stats",
            "Distribution summary of one column (numeric or categorical).",
            column_schema, column_stats,
        ),
        LoopTool(
            "value_samples",
            "Up to 12 distinct raw values of one column.",
            column_schema, value_samples,
        ),
    ]

"""Base EDA checks that run for every design."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis_v2.spec import EDACheck, EDACheckStatus

from .common import ArtifactSink, EDAInputs, failed, resolved_col, skipped


def shape_and_missingness(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    profile = inputs.profile
    high = profile.high_missingness_columns
    status = EDACheckStatus.WARNING if high else EDACheckStatus.OK
    detail = f"{profile.n_rows:,} rows x {profile.n_columns} columns"
    if high:
        detail += f"; high missingness in {', '.join(high)}"
    return EDACheck(
        name="shape_and_missingness", status=status, detail=detail,
        metrics={"n_rows": float(profile.n_rows), "n_columns": float(profile.n_columns)},
    )


def duplicates(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    count = inputs.profile.duplicate_row_count
    share = count / inputs.profile.n_rows if inputs.profile.n_rows else 0.0
    status = EDACheckStatus.WARNING if share > 0.01 else EDACheckStatus.OK
    return EDACheck(
        name="duplicates", status=status,
        detail=f"{count} duplicate rows ({share:.1%})",
        metrics={"duplicate_rows": float(count), "duplicate_share": round(share, 4)},
    )


def outcome_distribution(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    col = resolved_col(inputs.spec.outcome, inputs.frame)
    if col is None:
        return skipped("outcome_distribution", "outcome unresolved")
    try:
        series = pd.to_numeric(inputs.frame[col], errors="coerce").dropna()
        if series.empty:
            return skipped("outcome_distribution", f"'{col}' has no numeric values")
        fig, ax = plt.subplots(figsize=(6, 3))
        series.hist(ax=ax, bins=40, color="#5f7fb4")
        ax.set_title(f"Outcome distribution: {col}")
        artifact = sink.add_plot("outcome_distribution", fig, f"Outcome distribution ({col})")
        skew = float(series.skew())
        status = EDACheckStatus.WARNING if abs(skew) > 3 else EDACheckStatus.OK
        return EDACheck(
            name="outcome_distribution", status=status,
            detail=f"{col}: mean {series.mean():.3g}, sd {series.std():.3g}, skew {skew:.2f}",
            metrics={
                "mean": round(float(series.mean()), 6),
                "std": round(float(series.std()), 6),
                "skew": round(skew, 4),
            },
            artifact_ids=[artifact],
        )
    except Exception as exc:
        return failed("outcome_distribution", exc)


def exposure_distribution(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    spec = inputs.spec
    col = resolved_col(spec.treatment, inputs.frame)
    if col is None:
        return skipped(
            "exposure_distribution",
            "treatment unresolved or derived" if not spec.candidate_factors
            else "multi-factor question; see factor checks",
        )
    try:
        series = inputs.frame[col]
        counts = series.value_counts(dropna=True)
        if spec.treatment_is_continuous:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            fig, ax = plt.subplots(figsize=(6, 3))
            numeric.hist(ax=ax, bins=40, color="#8a6fb4")
            ax.set_title(f"Exposure distribution: {col}")
            artifact = sink.add_plot("exposure_distribution", fig, f"Exposure distribution ({col})")
            return EDACheck(
                name="exposure_distribution", status=EDACheckStatus.OK,
                detail=f"{col}: continuous, range {numeric.min():.3g}..{numeric.max():.3g}",
                metrics={"min": float(numeric.min()), "max": float(numeric.max())},
                artifact_ids=[artifact],
            )
        smallest = int(counts.min()) if len(counts) else 0
        status = EDACheckStatus.WARNING if smallest < 20 else EDACheckStatus.OK
        groups = ", ".join(f"{k}: {v}" for k, v in counts.head(6).items())
        return EDACheck(
            name="exposure_distribution", status=status,
            detail=f"{col} group sizes: {groups}",
            metrics={"n_groups": float(len(counts)), "smallest_group": float(smallest)},
        )
    except Exception as exc:
        return failed("exposure_distribution", exc)


def usable_sample_size(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    spec, frame = inputs.spec, inputs.frame
    involved: list[str] = []
    for ref in (spec.outcome, spec.treatment, spec.mediator, spec.instrument,
                spec.running_variable, spec.duration_column, spec.event_column):
        col = resolved_col(ref, frame)
        if col:
            involved.append(col)
    involved += [c for c in spec.candidate_confounders if c in frame.columns]
    involved += [c for c in spec.candidate_factors if c in frame.columns]
    if not involved:
        return skipped("usable_sample_size", "no resolved analysis columns yet")
    complete = int(frame[sorted(set(involved))].dropna().shape[0])
    share = complete / len(frame) if len(frame) else 0.0
    status = EDACheckStatus.WARNING if share < 0.7 else EDACheckStatus.OK
    return EDACheck(
        name="usable_sample_size", status=status,
        detail=f"{complete:,} of {len(frame):,} rows complete on {len(set(involved))} columns",
        metrics={"complete_rows": float(complete), "complete_share": round(share, 4)},
    )


BASE_CHECKS = [
    shape_and_missingness,
    duplicates,
    outcome_distribution,
    exposure_distribution,
    usable_sample_size,
]

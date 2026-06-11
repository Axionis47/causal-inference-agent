"""Design-specific checks: dose-response, DID, RDD, IV, time series,
mediation, survival. All descriptive."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis_v2.spec import EDACheck, EDACheckStatus

from .common import ArtifactSink, EDAInputs, failed, resolved_col, skipped


def dose_response_scatter(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    x_col = resolved_col(inputs.spec.treatment, inputs.frame)
    y_col = resolved_col(inputs.spec.outcome, inputs.frame)
    if x_col is None or y_col is None:
        return skipped("dose_response_scatter", "exposure or outcome unresolved")
    try:
        x = pd.to_numeric(inputs.frame[x_col], errors="coerce")
        y = pd.to_numeric(inputs.frame[y_col], errors="coerce")
        ok = x.notna() & y.notna()
        x, y = x[ok], y[ok]
        bins = pd.qcut(x, min(12, max(3, x.nunique() // 5)), duplicates="drop")
        binned = y.groupby(bins, observed=True).mean()
        centers = [interval.mid for interval in binned.index]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(x, y, s=6, alpha=0.3, color="#5f7fb4")
        ax.plot(centers, binned.values, color="#b45f5f", marker="o", linewidth=1.5)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col} with binned means")
        artifact = sink.add_plot("dose_response_scatter", fig, f"{y_col} vs {x_col}")
        pearson = float(x.corr(y))
        spearman = float(x.corr(y, method="spearman"))
        nonlinear = abs(pearson - spearman) > 0.1
        return EDACheck(
            name="dose_response_scatter",
            status=EDACheckStatus.WARNING if nonlinear else EDACheckStatus.OK,
            detail=f"pearson {pearson:.3f}, spearman {spearman:.3f}"
            + ("; gap suggests nonlinearity" if nonlinear else ""),
            metrics={"pearson": round(pearson, 4), "spearman": round(spearman, 4)},
            artifact_ids=[artifact],
        )
    except Exception as exc:
        return failed("dose_response_scatter", exc)


def did_trends(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    time_col = resolved_col(inputs.spec.time_column, inputs.frame)
    group_col = resolved_col(inputs.spec.group_column, inputs.frame) or resolved_col(
        inputs.spec.treatment, inputs.frame
    )
    outcome = resolved_col(inputs.spec.outcome, inputs.frame)
    if time_col is None or group_col is None or outcome is None:
        return skipped("did_trends", "needs time, group, and outcome columns (long format)")
    try:
        frame = inputs.frame[[time_col, group_col, outcome]].dropna()
        trends = frame.groupby([time_col, group_col])[outcome].mean().unstack()
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        trends.plot(ax=ax, marker="o")
        ax.set_title(f"Mean {outcome} over {time_col} by {group_col}")
        plot_id = sink.add_plot("did_trends", fig, "Outcome trends by group")
        sizes = frame.groupby([time_col, group_col]).size().unstack().fillna(0)
        table_id = sink.add_table(
            "did_group_sizes", sizes.reset_index(), "Group sizes by period"
        )
        empty_cells = int((sizes == 0).sum().sum())
        status = EDACheckStatus.WARNING if empty_cells else EDACheckStatus.OK
        return EDACheck(
            name="did_trends", status=status,
            detail=f"{trends.shape[0]} periods x {trends.shape[1]} groups"
            + (f"; {empty_cells} empty panel cells" if empty_cells else ""),
            metrics={"n_periods": float(trends.shape[0]), "empty_cells": float(empty_cells)},
            artifact_ids=[plot_id, table_id],
        )
    except Exception as exc:
        return failed("did_trends", exc)


def did_pre_trends(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    time_col = resolved_col(inputs.spec.time_column, inputs.frame)
    group_col = resolved_col(inputs.spec.group_column, inputs.frame) or resolved_col(
        inputs.spec.treatment, inputs.frame
    )
    outcome = resolved_col(inputs.spec.outcome, inputs.frame)
    post_col = "post" if "post" in inputs.frame.columns else None
    if not all((time_col, group_col, outcome, post_col)):
        return skipped(
            "did_pre_trends", "needs long-format time, group, outcome, and a post indicator"
        )
    try:
        pre = inputs.frame[inputs.frame[post_col] == 0]
        slopes = {}
        for g, chunk in pre.groupby(group_col):
            means = chunk.groupby(time_col)[outcome].mean()
            if len(means) >= 2:
                slopes[str(g)] = float(np.polyfit(range(len(means)), means.values, 1)[0])
        if len(slopes) < 2:
            return skipped("did_pre_trends", "fewer than two pre-period points per group")
        values = list(slopes.values())
        gap = float(abs(values[0] - values[1]))
        scale = max(abs(v) for v in values) or 1.0
        status = EDACheckStatus.WARNING if gap > 0.5 * scale else EDACheckStatus.OK
        return EDACheck(
            name="did_pre_trends", status=status,
            detail=f"pre-period slopes {slopes}; gap {gap:.4g} (descriptive parallel-trends look)",
            metrics={"pre_trend_gap": round(gap, 6)},
        )
    except Exception as exc:
        return failed("did_pre_trends", exc)


def rdd_around_cutoff(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    run_col = resolved_col(inputs.spec.running_variable, inputs.frame)
    outcome = resolved_col(inputs.spec.outcome, inputs.frame)
    cutoff = inputs.spec.cutoff_value
    if run_col is None or outcome is None:
        return skipped("rdd_around_cutoff", "running variable or outcome unresolved")
    if cutoff is None:
        return skipped("rdd_around_cutoff", "cutoff unconfirmed; plan gate must collect it")
    try:
        x = pd.to_numeric(inputs.frame[run_col], errors="coerce")
        y = pd.to_numeric(inputs.frame[outcome], errors="coerce")
        ok = x.notna() & y.notna()
        x, y = x[ok], y[ok]
        bandwidth = float(x.std()) or 1.0
        near = (x - cutoff).abs() <= bandwidth
        n_near = int(near.sum())
        bins = pd.cut(x[near], 24)
        binned = y[near].groupby(bins, observed=True).mean()
        centers = [b.mid for b in binned.index]
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.scatter(x[near], y[near], s=6, alpha=0.25, color="#5f7fb4")
        ax.plot(centers, binned.values, color="#b45f5f", marker="o", linewidth=1.2)
        ax.axvline(cutoff, color="#333", linestyle="--")
        ax.set_title(f"{outcome} vs {run_col} near the cutoff {cutoff:g}")
        plot_id = sink.add_plot("rdd_outcome_vs_running", fig, "Outcome vs running variable")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        x.hist(ax=ax2, bins=50, color="#8a6fb4")
        ax2.axvline(cutoff, color="#333", linestyle="--")
        ax2.set_title(f"Running variable distribution: {run_col}")
        hist_id = sink.add_plot("rdd_running_distribution", fig2, "Running variable distribution")
        below = int(((x >= cutoff - bandwidth) & (x < cutoff)).sum())
        above = int(((x >= cutoff) & (x <= cutoff + bandwidth)).sum())
        ratio = (above / below) if below else float("inf")
        status = (
            EDACheckStatus.WARNING
            if n_near < 100 or ratio > 1.5 or ratio < 0.67
            else EDACheckStatus.OK
        )
        return EDACheck(
            name="rdd_around_cutoff", status=status,
            detail=f"{n_near} rows within one sd of the cutoff; above/below ratio "
            f"{ratio:.2f} (a skewed ratio can hint at bunching)",
            metrics={
                "n_near_cutoff": float(n_near),
                "above_below_ratio": round(float(ratio), 4) if below else 99.0,
            },
            artifact_ids=[plot_id, hist_id],
        )
    except Exception as exc:
        return failed("rdd_around_cutoff", exc)


def rdd_treatment_by_side(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    run_col = resolved_col(inputs.spec.running_variable, inputs.frame)
    treat = resolved_col(inputs.spec.treatment, inputs.frame)
    cutoff = inputs.spec.cutoff_value
    if run_col is None or treat is None or cutoff is None:
        return skipped(
            "rdd_treatment_by_side", "needs running variable, treatment column, and cutoff"
        )
    try:
        x = pd.to_numeric(inputs.frame[run_col], errors="coerce")
        d = pd.to_numeric(inputs.frame[treat], errors="coerce")
        ok = x.notna() & d.notna()
        take_above = float(d[ok & (x >= cutoff)].mean())
        take_below = float(d[ok & (x < cutoff)].mean())
        jump = take_above - take_below
        sharp = take_below <= 0.02 and take_above >= 0.98
        return EDACheck(
            name="rdd_treatment_by_side", status=EDACheckStatus.OK,
            detail=f"treatment rate {take_below:.2f} below vs {take_above:.2f} above the "
            f"cutoff ({'sharp' if sharp else 'fuzzy'} pattern, first-stage jump {jump:.2f})",
            metrics={
                "take_below": round(take_below, 4),
                "take_above": round(take_above, 4),
                "first_stage_jump": round(jump, 4),
            },
        )
    except Exception as exc:
        return failed("rdd_treatment_by_side", exc)


def iv_first_stage(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    z_col = resolved_col(inputs.spec.instrument, inputs.frame)
    x_col = resolved_col(inputs.spec.treatment, inputs.frame)
    y_col = resolved_col(inputs.spec.outcome, inputs.frame)
    if z_col is None or x_col is None:
        return skipped("iv_first_stage", "instrument or treatment unresolved")
    try:
        data = inputs.frame[[z_col, x_col] + ([y_col] if y_col else [])].apply(
            lambda s: pd.to_numeric(s, errors="coerce")
        ).dropna()
        corr_zx = float(data[z_col].corr(data[x_col]))
        detail = f"corr({z_col}, {x_col}) = {corr_zx:.3f}"
        metrics = {"corr_z_x": round(corr_zx, 4)}
        if y_col:
            corr_zy = float(data[z_col].corr(data[y_col]))
            detail += f"; corr({z_col}, {y_col}) = {corr_zy:.3f} (reduced form, descriptive)"
            metrics["corr_z_y"] = round(corr_zy, 4)
        status = EDACheckStatus.WARNING if abs(corr_zx) < 0.05 else EDACheckStatus.OK
        if abs(corr_zx) < 0.05:
            detail += "; very weak instrument-treatment relationship"
        return EDACheck(
            name="iv_first_stage", status=status, detail=detail, metrics=metrics
        )
    except Exception as exc:
        return failed("iv_first_stage", exc)


def series_over_time(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    time_col = resolved_col(inputs.spec.time_column, inputs.frame)
    outcome = resolved_col(inputs.spec.outcome, inputs.frame)
    if time_col is None or outcome is None:
        return skipped("series_over_time", "time column or outcome unresolved")
    try:
        frame = inputs.frame[[time_col, outcome]].copy()
        frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
        frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
        frame = frame.dropna().sort_values(time_col)
        series = frame.groupby(time_col)[outcome].mean()
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(series.index, series.values, linewidth=0.9, color="#5f7fb4")
        date = inputs.spec.intervention_date
        if date:
            ax.axvline(pd.to_datetime(date), color="#b45f5f", linestyle="--")
        ax.set_title(f"{outcome} over time" + (f" (intervention {date})" if date else ""))
        artifact = sink.add_plot("series_over_time", fig, f"{outcome} over time")
        gaps = int(series.index.to_series().diff().dt.days.gt(1).sum()) if len(series) > 1 else 0
        status = EDACheckStatus.WARNING if gaps > 0 else EDACheckStatus.OK
        return EDACheck(
            name="series_over_time", status=status,
            detail=f"{len(series)} time points"
            + (f"; {gaps} gaps over one day" if gaps else "")
            + ("" if date else "; no intervention date confirmed yet"),
            metrics={"n_points": float(len(series)), "n_gaps": float(gaps)},
            artifact_ids=[artifact],
        )
    except Exception as exc:
        return failed("series_over_time", exc)


def mediation_pathways(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    t = resolved_col(inputs.spec.treatment, inputs.frame)
    m = resolved_col(inputs.spec.mediator, inputs.frame)
    y = resolved_col(inputs.spec.outcome, inputs.frame)
    if not all((t, m, y)):
        return skipped("mediation_pathways", "needs treatment, mediator, and outcome")
    try:
        data = inputs.frame[[t, m, y]].apply(
            lambda s: pd.to_numeric(s, errors="coerce")
        ).dropna()
        c_tm = float(data[t].corr(data[m]))
        c_my = float(data[m].corr(data[y]))
        c_ty = float(data[t].corr(data[y]))
        return EDACheck(
            name="mediation_pathways", status=EDACheckStatus.WARNING,
            detail=(
                f"corr(T,M) {c_tm:.3f}, corr(M,Y) {c_my:.3f}, corr(T,Y) {c_ty:.3f}; "
                "mediator timing is assumed, not observed in cross-sectional data"
            ),
            metrics={"corr_t_m": round(c_tm, 4), "corr_m_y": round(c_my, 4),
                     "corr_t_y": round(c_ty, 4)},
        )
    except Exception as exc:
        return failed("mediation_pathways", exc)


def survival_overview(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    duration = resolved_col(inputs.spec.duration_column, inputs.frame)
    event = resolved_col(inputs.spec.event_column, inputs.frame)
    if duration is None or event is None:
        return skipped("survival_overview", "duration or event column unresolved")
    try:
        d = pd.to_numeric(inputs.frame[duration], errors="coerce")
        e = pd.to_numeric(inputs.frame[event], errors="coerce")
        ok = d.notna() & e.notna()
        event_rate = float(e[ok].mean())
        fig, ax = plt.subplots(figsize=(6, 3))
        d[ok].hist(ax=ax, bins=40, color="#5f7fb4")
        ax.set_title(f"Time-to-event distribution: {duration}")
        artifact = sink.add_plot("survival_duration", fig, "Time-to-event distribution")
        status = (
            EDACheckStatus.WARNING if event_rate < 0.05 or event_rate > 0.95
            else EDACheckStatus.OK
        )
        return EDACheck(
            name="survival_overview", status=status,
            detail=f"event rate {event_rate:.1%}, censoring {1 - event_rate:.1%}, "
            f"median follow-up {float(d[ok].median()):.4g}",
            metrics={"event_rate": round(event_rate, 4),
                     "censoring_rate": round(1 - event_rate, 4)},
            artifact_ids=[artifact],
        )
    except Exception as exc:
        return failed("survival_overview", exc)

"""Deterministic checks on whether an estimate holds up.

Each returns a Finding: a verdict, a number, and a sentence. None of them can
change the estimate. They exist to answer the question the estimate cannot
answer about itself, which is whether the assumption it rests on looks
survivable.

Several assumptions cannot be tested at all, and saying so plainly is part of
the job. A two-period difference in differences has no pre-period to inspect,
so there is nothing to check and the honest output is "untestable here", not
silence and not a reassuring green tick.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .estimate import LaneError
from .prep import numeric_frame


@dataclass
class Finding:
    check: str
    verdict: str  # pass | warn | fail | untestable
    detail: str
    value: float | None = None

    def __str__(self) -> str:
        num = f" ({self.value:.4g})" if self.value is not None else ""
        return f"[{self.verdict}] {self.check}{num}: {self.detail}"


def _lane_call(lane: str, df: pd.DataFrame, kwargs: dict):
    from . import lanes as lane_module

    fn = getattr(lane_module, lane)
    kw = {k: (tuple(v) if isinstance(v, list) else v)
          for k, v in kwargs.items() if not str(k).startswith("_")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(df, **kw)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def balance(df: pd.DataFrame, lane: str, kwargs: dict, estimate: dict) -> Finding:
    """Standardised differences between arms. Above 0.1 is the usual alarm."""
    treatment = kwargs.get("treatment")
    covs = list(kwargs.get("covariates") or [])
    if not treatment or not covs:
        return Finding("balance", "untestable",
                       "this design has no binary treatment with covariates")
    data = numeric_frame(df, [treatment, *covs], "balance")
    levels = sorted(data[treatment].unique())
    if len(levels) != 2:
        return Finding("balance", "untestable",
                       f"treatment has {len(levels)} levels, not two")
    a, b = data[data[treatment] == levels[0]], data[data[treatment] == levels[1]]
    worst, where = 0.0, ""
    for c in covs:
        pooled = np.sqrt((a[c].var() + b[c].var()) / 2)
        if pooled == 0 or np.isnan(pooled):
            continue
        smd = abs(a[c].mean() - b[c].mean()) / pooled
        if smd > worst:
            worst, where = float(smd), c
    verdict = "pass" if worst < 0.1 else "warn" if worst < 0.25 else "fail"
    return Finding("balance", verdict,
                   f"largest standardised difference is on '{where}'; "
                   f"under 0.1 is comparable, over 0.25 is not", worst)


def overlap(df: pd.DataFrame, lane: str, kwargs: dict, estimate: dict) -> Finding:
    """Do the two arms occupy the same propensity range? No overlap, no comparison."""
    treatment = kwargs.get("treatment")
    covs = list(kwargs.get("covariates") or [])
    if not treatment or not covs:
        return Finding("overlap", "untestable", "needs a binary treatment and covariates")
    data = numeric_frame(df, [treatment, *covs], "overlap")
    levels = sorted(data[treatment].unique())
    if len(levels) != 2:
        return Finding("overlap", "untestable", f"treatment has {len(levels)} levels")
    from sklearn.linear_model import LogisticRegression

    x = data[covs].to_numpy(float)
    x = (x - x.mean(0)) / np.where(x.std(0) == 0, 1, x.std(0))
    t = (data[treatment] == levels[1]).astype(int)
    ps = LogisticRegression(max_iter=2000).fit(x, t).predict_proba(x)[:, 1]
    lo = max(ps[t == 1].min(), ps[t == 0].min())
    hi = min(ps[t == 1].max(), ps[t == 0].max())
    inside = float(((ps >= lo) & (ps <= hi)).mean())
    verdict = "pass" if inside > 0.9 else "warn" if inside > 0.75 else "fail"
    return Finding("overlap", verdict,
                   f"{inside:.0%} of units sit where both arms are present; "
                   f"the rest have no counterpart to compare against", inside)


def subsample_stability(df: pd.DataFrame, lane: str, kwargs: dict, estimate: dict) -> Finding:
    """Refit on random 70% subsamples. A stable design should barely move."""
    point = estimate.get("value")
    if point in (None, 0):
        return Finding("subsample_stability", "untestable", "no point estimate to compare")
    rng = np.random.default_rng(0)
    got = []
    for _ in range(10):
        take = df.sample(frac=0.7, random_state=int(rng.integers(1 << 30)))
        try:
            got.append(_lane_call(lane, take, kwargs).value)
        except (LaneError, Exception):
            continue
    if len(got) < 5:
        return Finding("subsample_stability", "untestable",
                       f"only {len(got)} of 10 subsamples could be fitted")
    spread = float(np.std(got) / abs(point))
    verdict = "pass" if spread < 0.2 else "warn" if spread < 0.5 else "fail"
    return Finding("subsample_stability", verdict,
                   f"across 10 subsamples the estimate ranges "
                   f"{min(got):.4g} to {max(got):.4g} against {point:.4g}", spread)


def leave_one_out(df: pd.DataFrame, lane: str, kwargs: dict, estimate: dict) -> Finding:
    """Drop each covariate in turn. A result resting on one control is fragile."""
    covs = list(kwargs.get("covariates") or [])
    point = estimate.get("value")
    if len(covs) < 2 or point in (None, 0):
        return Finding("leave_one_out", "untestable", "needs two or more covariates")
    worst, where = 0.0, ""
    for c in covs:
        trimmed = dict(kwargs, covariates=[x for x in covs if x != c])
        try:
            moved = abs(_lane_call(lane, df, trimmed).value - point) / abs(point)
        except Exception:
            continue
        if moved > worst:
            worst, where = float(moved), c
    verdict = "pass" if worst < 0.15 else "warn" if worst < 0.4 else "fail"
    return Finding("leave_one_out", verdict,
                   f"dropping '{where}' moves the estimate most; a result that "
                   f"hangs on one control is resting on that control", worst)


def placebo_cutoff(df: pd.DataFrame, lane: str, kwargs: dict, estimate: dict) -> Finding:
    """RDD only: look for a jump where no rule exists. Finding one is bad news."""
    if lane != "rdd":
        return Finding("placebo_cutoff", "untestable", "only applies to a discontinuity design")
    running, real = kwargs.get("running"), kwargs.get("cutoff")
    point = estimate.get("value")
    if not running or real is None or not point:
        return Finding("placebo_cutoff", "untestable", "no running variable or cutoff")
    col = pd.to_numeric(df[running], errors="coerce").dropna()
    fakes = [float(col.quantile(q)) for q in (0.25, 0.4, 0.6, 0.75)]
    fakes = [f for f in fakes if abs(f - real) > 0.05 * (col.max() - col.min())]
    jumps = []
    for f in fakes:
        try:
            jumps.append(abs(_lane_call(lane, df, dict(kwargs, cutoff=f)).value))
        except Exception:
            continue
    if not jumps:
        return Finding("placebo_cutoff", "untestable", "no placebo cutoff could be fitted")
    ratio = float(max(jumps) / abs(point))
    verdict = "pass" if ratio < 0.5 else "warn" if ratio < 1.0 else "fail"
    return Finding("placebo_cutoff", verdict,
                   f"the largest jump at a fake cutoff is {max(jumps):.4g} against "
                   f"{abs(point):.4g} at the real one", ratio)


def placebo_date(df: pd.DataFrame, lane: str, kwargs: dict, estimate: dict) -> Finding:
    """Time series only: a step at a date when nothing happened is a warning."""
    if lane != "time_series":
        return Finding("placebo_date", "untestable", "only applies to an interrupted series")
    time_col, real = kwargs.get("time"), kwargs.get("intervention")
    point = estimate.get("value")
    if not time_col or not real or not point:
        return Finding("placebo_date", "untestable", "no time column or intervention date")
    stamps = pd.to_datetime(df[time_col], errors="coerce", format="mixed").dropna().sort_values()
    real_ts = pd.to_datetime(real)
    steps = []
    for q in (0.3, 0.45, 0.7):
        fake = stamps.quantile(q)
        if abs((fake - real_ts).days) < 60:
            continue
        try:
            steps.append(abs(_lane_call(lane, df, dict(kwargs, intervention=str(fake.date()))).value))
        except Exception:
            continue
    if not steps:
        return Finding("placebo_date", "untestable", "no placebo date could be fitted")
    ratio = float(max(steps) / abs(point))
    verdict = "pass" if ratio < 0.5 else "warn" if ratio < 1.0 else "fail"
    return Finding("placebo_date", verdict,
                   f"the largest step at a date when nothing happened is "
                   f"{max(steps):.4g} against {abs(point):.4g} at the real one", ratio)


def proportional_hazards(df: pd.DataFrame, lane: str, kwargs: dict, estimate: dict) -> Finding:
    """Survival only: does the hazard ratio hold steady, or drift with time?

    Split the follow-up at its median and refit each half. A constant ratio is
    the model's central assumption, so two very different halves mean the single
    number reported is an average over something that changed.
    """
    if lane != "survival":
        return Finding("proportional_hazards", "untestable", "only applies to a survival model")
    duration, point = kwargs.get("duration"), estimate.get("value")
    if not duration or not point:
        return Finding("proportional_hazards", "untestable", "no duration column")
    times = pd.to_numeric(df[duration], errors="coerce")
    mid = float(times.median())
    halves = []
    for label, part in (("early", df[times <= mid]), ("late", df[times > mid])):
        try:
            halves.append((label, _lane_call(lane, part, kwargs).value))
        except Exception:
            continue
    if len(halves) < 2:
        return Finding("proportional_hazards", "untestable",
                       "one half of the follow-up could not be fitted")
    (_, early), (_, late) = halves
    gap = float(abs(early - late) / abs(point))
    verdict = "pass" if gap < 0.3 else "warn" if gap < 0.7 else "fail"
    return Finding("proportional_hazards", verdict,
                   f"hazard ratio is {early:.4g} in the first half of follow-up and "
                   f"{late:.4g} in the second; the model reports one number for both", gap)


def pre_trend(df: pd.DataFrame, lane: str, kwargs: dict, estimate: dict) -> Finding:
    """DiD only: were the groups moving together before? Often unanswerable."""
    if lane != "did":
        return Finding("pre_trend", "untestable", "only applies to difference in differences")
    period = kwargs.get("period")
    if not period:
        return Finding("pre_trend", "untestable", "no period column")
    n_periods = int(df[period].nunique())
    if n_periods <= 2:
        return Finding(
            "pre_trend", "untestable",
            f"only {n_periods} periods, so there is no pre-period to inspect. "
            "Parallel trends is assumed here, not shown, and no amount of this "
            "data can show it")
    return Finding("pre_trend", "warn",
                   f"{n_periods} periods are present; a pre-trend test is possible "
                   "but is not implemented yet")


CHECKS = {
    "balance": balance,
    "overlap": overlap,
    "subsample_stability": subsample_stability,
    "leave_one_out": leave_one_out,
    "placebo_cutoff": placebo_cutoff,
    "placebo_date": placebo_date,
    "proportional_hazards": proportional_hazards,
    "pre_trend": pre_trend,
}

DESCRIPTIONS = {
    "balance": "standardised differences between treated and control on each covariate",
    "overlap": "whether both arms occupy the same propensity range",
    "subsample_stability": "refit on ten random 70% subsamples and report the spread",
    "leave_one_out": "drop each covariate in turn and see how far the estimate moves",
    "placebo_cutoff": "run the discontinuity at fake cutoffs where no rule exists",
    "placebo_date": "run the interruption at dates when nothing happened",
    "proportional_hazards": "compare the hazard ratio early and late in follow-up",
    "pre_trend": "whether the groups were moving together before treatment",
}


def run(name: str, df: pd.DataFrame, lane: str, kwargs: dict, estimate: dict) -> Finding:
    fn = CHECKS.get(name)
    if fn is None:
        return Finding(name, "untestable", "no such check")
    try:
        return fn(df, lane, kwargs, estimate)
    except LaneError as exc:
        return Finding(name, "untestable", str(exc)[:160])
    except Exception as exc:  # a broken check must not sink the run
        return Finding(name, "untestable", f"{type(exc).__name__}: {exc}"[:160])

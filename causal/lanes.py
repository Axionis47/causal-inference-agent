"""The eight estimation lanes.

Each takes a DataFrame and named columns, and returns one Estimate. The
signature is the documentation: there is no settings dict and no registry to
consult. A lane that cannot run raises LaneError naming the column and reason.

Every lane here is deliberately the plain version of its method. What is left
out on purpose: inverse-probability weighting, bootstrap standard errors,
collinearity pruning, one-hot encoding of categoricals (pass numeric columns),
and plots. Each gets added when a test needs it, not before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .estimate import Estimate, LaneError
from .prep import as_binary, ci95, numeric_frame, require_variation


def _clean_covariates(covariates, *reserved: str) -> tuple[str, ...]:
    """Covariates minus anything already playing another part.

    A suggested list can name the treatment, the outcome, or a survival
    duration. Passing it through gives a perfectly collinear design and
    statsmodels raises "Singular matrix", which tells the reader nothing.
    """
    taken = {r for r in reserved if r}
    seen: list[str] = []
    for c in covariates:
        if c and c not in taken and c not in seen:
            seen.append(c)
    return tuple(seen)


def observational(
    df: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    covariates: tuple[str, ...] = (),
) -> Estimate:
    """Covariate-adjusted regression. The ATE is the treatment coefficient.

    Identifies a causal effect only if the covariates close every backdoor
    path. Nothing here can check that; it is an assumption you bring.
    """
    lane = "observational"
    covariates = _clean_covariates(covariates, outcome, treatment)
    cols = [outcome, treatment, *covariates]
    data = numeric_frame(df, cols, lane)
    require_variation(data[treatment], "treatment", lane)

    design = sm.add_constant(data[[treatment, *covariates]], has_constant="add")
    fit = sm.OLS(data[outcome], design).fit()
    value = float(fit.params[treatment])
    se = float(fit.bse[treatment])
    lo, hi = ci95(value, se)
    return Estimate(
        estimand="ate",
        value=value,
        se=se,
        ci_low=lo,
        ci_high=hi,
        p_value=float(fit.pvalues[treatment]),
        n=len(data),
        estimator="ols_adjusted",
        notes=[f"adjusted for {len(covariates)} covariate(s)"],
    )


def matching(
    df: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    covariates: tuple[str, ...],
) -> Estimate:
    """Propensity-score matching, nearest neighbour with replacement.

    Estimates the ATT: each treated unit is paired with the control whose
    propensity score is closest, and the effect is the mean paired difference.
    Standard error is the paired-difference standard error, which ignores the
    uncertainty in the propensity model itself and so runs slightly optimistic.
    """
    lane = "matching"
    covariates = _clean_covariates(covariates, outcome, treatment)
    if not covariates:
        raise LaneError(f"{lane}: needs covariates to match on")
    data = numeric_frame(df, [outcome, treatment, *covariates], lane)
    t = as_binary(data[treatment], "treatment", lane)

    treated, control = data[t == 1], data[t == 0]
    if len(treated) < 10 or len(control) < 10:
        raise LaneError(
            f"{lane}: needs 10+ units per arm, has {len(treated)} treated "
            f"and {len(control)} control"
        )

    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    x = data[list(covariates)].to_numpy(float)
    x = (x - x.mean(0)) / np.where(x.std(0) == 0, 1, x.std(0))
    ps = LogisticRegression(max_iter=2000).fit(x, t).predict_proba(x)[:, 1]

    ps_t, ps_c = ps[t.to_numpy() == 1], ps[t.to_numpy() == 0]
    nn = NearestNeighbors(n_neighbors=1).fit(ps_c.reshape(-1, 1))
    _, idx = nn.kneighbors(ps_t.reshape(-1, 1))

    y_t = treated[outcome].to_numpy(float)
    y_c = control[outcome].to_numpy(float)[idx.ravel()]
    diff = y_t - y_c
    value = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    lo, hi = ci95(value, se)
    return Estimate(
        estimand="att",
        value=value,
        se=se,
        ci_low=lo,
        ci_high=hi,
        p_value=None,
        n=len(data),
        estimator="propensity_nn_matching",
        notes=[
            f"{len(treated)} treated matched to {len(np.unique(idx))} distinct controls",
            "SE ignores propensity-model uncertainty",
        ],
    )


def iv(
    df: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    instrument: str,
    covariates: tuple[str, ...] = (),
) -> Estimate:
    """Two-stage least squares. The LATE is the fitted treatment coefficient.

    Valid only if the instrument moves treatment and affects the outcome by no
    other route. The first stage is testable and is checked here; the exclusion
    restriction is not testable and is an assumption you bring.
    """
    lane = "iv"
    # The instrument is not a control. Suggested covariate lists can include it,
    # and joining it twice raises deep inside pandas rather than saying so here.
    covariates = _clean_covariates(covariates, outcome, treatment, instrument)
    data = numeric_frame(df, [outcome, treatment, instrument, *covariates], lane)
    require_variation(data[instrument], "instrument", lane)
    require_variation(data[treatment], "treatment", lane)

    controls = sm.add_constant(data[list(covariates)], has_constant="add")

    first = sm.OLS(data[treatment], controls.join(data[[instrument]])).fit()
    f_stat = float(first.tvalues[instrument] ** 2)
    if float(first.pvalues[instrument]) >= 0.05:
        raise LaneError(
            f"{lane}: instrument '{instrument}' does not move treatment "
            f"(first-stage p={first.pvalues[instrument]:.3g})"
        )

    from statsmodels.sandbox.regression.gmm import IV2SLS

    fit = IV2SLS(
        data[outcome],
        controls.join(data[[treatment]]),
        instrument=controls.join(data[[instrument]]),
    ).fit()
    value = float(fit.params[treatment])
    se = float(fit.bse[treatment])
    lo, hi = ci95(value, se)
    notes = [f"first-stage F={f_stat:.1f}"]
    if f_stat < 10:
        notes.append("weak instrument: F below 10, treat the interval with suspicion")
    return Estimate(
        estimand="late",
        value=value,
        se=se,
        ci_low=lo,
        ci_high=hi,
        p_value=float(fit.pvalues[treatment]),
        n=len(data),
        estimator="two_stage_least_squares",
        notes=notes,
    )


def survival(
    df: pd.DataFrame,
    *,
    treatment: str,
    duration: str,
    event: str,
    covariates: tuple[str, ...] = (),
) -> Estimate:
    """Cox proportional hazards. Reports the hazard ratio for treatment.

    A ratio above 1 means the treated group fails faster. Assumes the hazard
    ratio is constant over time, which this does not check.
    """
    lane = "survival"
    covariates = _clean_covariates(covariates, duration, event, treatment)
    data = numeric_frame(df, [duration, event, treatment, *covariates], lane)
    require_variation(data[treatment], "treatment", lane)

    status = data[event]
    if not set(status.unique()) <= {0.0, 1.0}:
        raise LaneError(f"{lane}: event column '{event}' must be 0/1")
    if status.sum() < 10:
        raise LaneError(f"{lane}: only {int(status.sum())} events observed, need 10+")

    from statsmodels.duration.hazard_regression import PHReg

    try:
        fit = PHReg(
            data[duration], data[[treatment, *covariates]], status=status
        ).fit(disp=False)
    except Exception as exc:
        if "singular" in str(exc).lower():
            raise LaneError(
                f"{lane}: the design is collinear across "
                f"{[treatment, *covariates]}; two of these carry the same "
                f"information, so drop one"
            ) from exc
        raise
    log_hr = float(fit.params[0])
    log_se = float(fit.bse[0])
    lo, hi = ci95(log_hr, log_se)
    return Estimate(
        estimand="hazard_ratio",
        value=float(np.exp(log_hr)),
        se=log_se,  # on the log scale, where the interval was built
        ci_low=float(np.exp(lo)) if lo is not None else None,
        ci_high=float(np.exp(hi)) if hi is not None else None,
        p_value=float(fit.pvalues[0]),
        n=len(data),
        estimator="cox_proportional_hazards",
        notes=[f"{int(status.sum())} events", "SE is on the log-hazard scale"],
    )


def did(
    df: pd.DataFrame,
    *,
    outcome: str,
    group: str,
    period: str,
    treated_group: str,
    unit: str | None = None,
) -> Estimate:
    """Difference in differences on a two-group, two-period panel.

    The estimate is the interaction: how much more the treated group moved
    than the control group did. `treated_group` is required rather than
    guessed, because guessing it silently flips the sign.

    Rests on parallel trends: absent the intervention, both groups would have
    moved together. With two periods there is no pre-trend to inspect, so this
    is an assumption you bring, not one the data can support.
    """
    lane = "did"
    for col in (outcome, group, period):
        if col not in df.columns:
            raise LaneError(f"{lane}: dataset has no column '{col}'")

    data = df[[outcome, group, period] + ([unit] if unit else [])].copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna()

    groups = sorted(map(str, data[group].unique()))
    if len(groups) != 2:
        raise LaneError(f"{lane}: needs exactly 2 groups, found {len(groups)}: {groups}")
    if str(treated_group) not in groups:
        raise LaneError(f"{lane}: treated_group '{treated_group}' is not one of {groups}")
    periods = sorted(data[period].unique())
    if len(periods) != 2:
        raise LaneError(f"{lane}: needs exactly 2 periods, found {len(periods)}")

    treated = (data[group].astype(str) == str(treated_group)).astype(float)
    post = (data[period] == periods[1]).astype(float)
    design = pd.DataFrame(
        {"const": 1.0, "treated": treated, "post": post, "did": treated * post}
    )
    model = sm.OLS(data[outcome].to_numpy(float), design)
    if unit:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": data[unit]})
        se_note = f"SEs clustered on {unit}"
    else:
        fit = model.fit(cov_type="HC1")
        se_note = "heteroskedasticity-robust SEs, not clustered"

    value = float(fit.params["did"])
    se = float(fit.bse["did"])
    lo, hi = ci95(value, se)
    return Estimate(
        estimand="att",
        value=value,
        se=se,
        ci_low=lo,
        ci_high=hi,
        p_value=float(fit.pvalues["did"]),
        n=len(data),
        estimator="difference_in_differences",
        notes=[f"{treated_group} vs {[g for g in groups if g != str(treated_group)][0]}", se_note],
    )


def rdd(
    df: pd.DataFrame,
    *,
    outcome: str,
    running: str,
    cutoff: float,
) -> Estimate:
    """Regression discontinuity. The effect is the jump in the outcome at the cutoff.

    Local by construction: only units near the cutoff identify the effect, so a
    bigger sample narrows the window rather than sharpening the estimate.
    """
    lane = "rdd"
    data = numeric_frame(df, [outcome, running], lane)
    lo_n = int((data[running] < cutoff).sum())
    hi_n = int((data[running] >= cutoff).sum())
    if not data[running].min() < cutoff < data[running].max():
        raise LaneError(
            f"{lane}: cutoff {cutoff} is outside the range of '{running}' "
            f"[{data[running].min():.4g}, {data[running].max():.4g}]"
        )
    if lo_n < 10 or hi_n < 10:
        raise LaneError(
            f"{lane}: needs 10+ rows each side of the cutoff, has {lo_n} below and {hi_n} above"
        )

    import rdrobust

    fit = rdrobust.rdrobust(y=data[outcome], x=data[running], c=cutoff)
    value = float(fit.coef.iloc[0, 0])  # conventional local linear estimate
    se = float(fit.se.iloc[0, 0])
    lo, hi = ci95(value, se)
    return Estimate(
        estimand="jump",
        value=value,
        se=se,
        ci_low=lo,
        ci_high=hi,
        p_value=float(fit.pv.iloc[0, 0]),
        n=len(data),
        estimator="local_linear_rdd",
        notes=[
            f"bandwidth {float(fit.bws.iloc[0, 0]):.4g}, chosen by rdrobust",
            f"{lo_n} below / {hi_n} above the cutoff overall",
        ],
    )


def mediation(
    df: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    mediator: str,
    covariates: tuple[str, ...] = (),
) -> Estimate:
    """Product-of-coefficients mediation. Reports the indirect effect.

    Splits the total effect into the part running through the mediator
    (indirect) and the rest (direct). Assumes no unmeasured confounding of
    either the treatment-mediator or mediator-outcome relationship, which is a
    strong assumption and is not checked.
    """
    lane = "mediation"
    covariates = _clean_covariates(covariates, outcome, treatment, mediator)
    data = numeric_frame(df, [outcome, treatment, mediator, *covariates], lane)
    require_variation(data[treatment], "treatment", lane)

    controls = list(covariates)
    m_fit = sm.OLS(
        data[mediator],
        sm.add_constant(data[[treatment, *controls]], has_constant="add"),
    ).fit()
    y_fit = sm.OLS(
        data[outcome],
        sm.add_constant(data[[treatment, mediator, *controls]], has_constant="add"),
    ).fit()

    a, a_se = float(m_fit.params[treatment]), float(m_fit.bse[treatment])
    b, b_se = float(y_fit.params[mediator]), float(y_fit.bse[mediator])
    direct = float(y_fit.params[treatment])
    indirect = a * b
    # Sobel: the delta-method SE for a product of two independent coefficients
    se = float(np.sqrt(b**2 * a_se**2 + a**2 * b_se**2))
    lo, hi = ci95(indirect, se)
    return Estimate(
        estimand="indirect",
        value=indirect,
        se=se,
        ci_low=lo,
        ci_high=hi,
        p_value=None,
        n=len(data),
        estimator="product_of_coefficients",
        notes=[
            f"direct {direct:.4g}, total {direct + indirect:.4g}",
            "Sobel SE assumes a and b are independent",
        ],
    )


def time_series(
    df: pd.DataFrame,
    *,
    outcome: str,
    time: str,
    intervention: str,
) -> Estimate:
    """Interrupted time series. Reports the level shift at the intervention.

    Averages to one observation per timestamp, then fits trend, step, and
    change-in-slope. The step is measured at the intervention date, so it is a
    contrast between two extrapolated segment intercepts and is noisier than it
    looks. No control series, so anything else happening at the same moment is
    indistinguishable from the intervention.
    """
    lane = "time_series"
    for col in (outcome, time):
        if col not in df.columns:
            raise LaneError(f"{lane}: dataset has no column '{col}'")

    stamps = pd.to_datetime(df[time], errors="coerce", format="mixed")
    values = pd.to_numeric(
        df[outcome].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    series = (
        pd.DataFrame({"t": stamps, "y": values})
        .dropna()
        .groupby("t", as_index=False)["y"]
        .mean()
        .sort_values("t")
        .reset_index(drop=True)
    )
    cut = pd.to_datetime(intervention)
    n_pre = int((series["t"] < cut).sum())
    n_post = int((series["t"] >= cut).sum())
    if len(series) < 30:
        raise LaneError(f"{lane}: only {len(series)} time points, need 30+")
    if n_pre < 10 or n_post < 10:
        raise LaneError(
            f"{lane}: needs 10+ points each side of {intervention}, "
            f"has {n_pre} before and {n_post} after"
        )

    trend = np.arange(len(series), dtype=float)
    post = (series["t"] >= cut).to_numpy(float)
    design = pd.DataFrame(
        {
            "const": 1.0,
            "trend": trend,
            "post": post,
            "post_trend": (trend - n_pre) * post,
        }
    )
    fit = sm.OLS(series["y"], design).fit(
        cov_type="HAC", cov_kwds={"maxlags": max(1, round(len(series) ** (1 / 3)))}
    )
    value = float(fit.params["post"])
    se = float(fit.bse["post"])
    lo, hi = ci95(value, se)
    return Estimate(
        estimand="level_shift",
        value=value,
        se=se,
        ci_low=lo,
        ci_high=hi,
        p_value=float(fit.pvalues["post"]),
        n=len(series),
        estimator="interrupted_time_series",
        notes=[
            f"{n_pre} points before / {n_post} after {intervention}",
            f"slope change {float(fit.params['post_trend']):.4g} per period",
            "HAC standard errors; no control series",
        ],
    )

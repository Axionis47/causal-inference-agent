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
    data = numeric_frame(df, [duration, event, treatment, *covariates], lane)
    require_variation(data[treatment], "treatment", lane)

    status = data[event]
    if not set(status.unique()) <= {0.0, 1.0}:
        raise LaneError(f"{lane}: event column '{event}' must be 0/1")
    if status.sum() < 10:
        raise LaneError(f"{lane}: only {int(status.sum())} events observed, need 10+")

    from statsmodels.duration.hazard_regression import PHReg

    fit = PHReg(
        data[duration], data[[treatment, *covariates]], status=status
    ).fit(disp=False)
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

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

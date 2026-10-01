"""The only file that imports pyfixest. Reads a frozen Design and the canonical panel, returns artifacts.

One `run` for every estimator, by its engine: a formula on feols; Gardner's two stages on did2s; local projections on
lpdid; the saturated event study, whose pooled effects this file computes from the cohort-by-period coefficients because
the library's own aggregates are not implemented in this version. Inference is a vcov passed through. Placebos refit
the same shape on a perturbed panel. Failures come back as facts on the artifact, never as exceptions.
"""

from __future__ import annotations

import logging
import re
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from causal_agent.common.contracts import Estimate, Refutation
from causal_agent.families.diff_in_diff.lane.knowledge import EstimatorEntry, PlaceboEntry
from causal_agent.families.diff_in_diff.lane.shape import period_label

logging.getLogger("pyfixest").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="pyfixest")

SEED = 7
_QUIET = threading.Lock()
COEF = "treat"
DYNAMIC = re.compile(r"rel_time::(-?\d+)(?:\.0)?(?::treated)?$")  # the two-way and the two-stage dynamic spellings
SATURATED = re.compile(r"rel_time::(-?\d+)(?:\.0)?:first_treated_period::(\d+)")
Z = 1.959963984540054


def formula_for(entry: EstimatorEntry, controls: list[str]) -> str:
    """Fill {controls} from the entry's mode. Never sees a raw column name from the model."""
    if not controls or entry.controls_mode == "none":
        return entry.formula.replace("{controls}", "")
    if entry.controls_mode == "csw0" and len(controls) >= 2:  # pyfixest's stepwise operators need at least two terms
        return entry.formula.replace("{controls}", " + csw0(" + ", ".join(controls) + ")")
    return entry.formula.replace("{controls}", " + " + " + ".join(controls))


def first_stage_for(entry: EstimatorEntry, controls: list[str]) -> str:
    """The two-stage estimator's first stage with the controls in, or the fixed effects alone."""
    return entry.first_stage.replace("{controls}", " + ".join(controls) if controls else "0")


def describe_spec(entry: EstimatorEntry, controls: list[str]) -> str:
    """What the design records as the specification: the formula for feols, the two stages for did2s, the surface for the rest."""
    if entry.engine == "feols":
        return formula_for(entry, controls)
    if entry.engine == "did2s":
        return f"first stage {first_stage_for(entry, controls)}; second stage {entry.second_stage}"
    if entry.engine == "lpdid":
        return f"local projections, up to {int(entry.params.get('max_window', 8))} periods either side, pooled effect on the treated"
    return "saturated event study by cohort and period" + (f" with {', '.join(controls)}" if controls else "")


def _cluster_of(vcov: Any) -> str:
    """The column the two-stage and saturated estimators cluster on: the CRV column when the design clusters, else the unit."""
    if isinstance(vcov, dict):
        for k in ("CRV1", "CRV3"):
            if k in vcov:
                return str(vcov[k]).split("+")[0]
    return "unit"


def fit(formula: str, panel: pd.DataFrame, vcov: Any):
    import pyfixest as pf

    return pf.feols(formula, panel, vcov=vcov)


def estimate(
    entry: EstimatorEntry, formula: str, panel: pd.DataFrame, vcov: Any, contrast_key: str, target_units: str, *, secondary: bool = False
) -> tuple[list[Estimate], Any]:
    """Returns one Estimate per fitted model (csw0 yields several: no controls, then each added) and the primary fit object
    for follow-ups. Under csw0 the primary is the full formula the design advertises, the last model; the steps are secondary,
    labelled by how many controls they carry."""
    n_t = int(panel.loc[panel["treated"] == 1, "unit"].nunique())
    n_c = int(panel.loc[panel["treated"] == 0, "unit"].nunique())
    try:
        res = fit(formula, panel, vcov)
        models = res.to_list() if hasattr(res, "to_list") else [res]
        stepwise = entry.controls_mode == "csw0" and len(models) > 1
        primary_i = len(models) - 1 if stepwise else 0
        out: list[Estimate] = []
        for i, m in enumerate(models):
            label = entry.name if i == primary_i else f"{entry.name}+{i}"
            if COEF in m.coef().index:
                lo, hi = (float(x) for x in m.confint().loc[COEF].to_numpy())
                out.append(
                    Estimate(
                        contrast=contrast_key,
                        method=label,
                        value=float(m.coef()[COEF]),
                        ci_low=lo,
                        ci_high=hi,
                        n_treated=n_t,
                        n_control=n_c,
                        target_units=target_units,
                        secondary=secondary or i != primary_i,
                        p_value=float(m.pvalue()[COEF]),
                        p_value_source=_vcov_words(vcov),
                    )
                )
            else:  # dynamic model: the effect is the mean of post-period coefficients; each period is reported separately
                names = [n for n in m.coef().index if DYNAMIC.search(n)]
                lags = [n for n in names if int(DYNAMIC.search(n).group(1)) >= 0]
                if lags:
                    vals = m.coef()[lags]
                    out.append(
                        Estimate(
                            contrast=contrast_key,
                            method=label,
                            value=float(vals.mean()),
                            n_treated=n_t,
                            n_control=n_c,
                            target_units=target_units,
                            secondary=True,
                        )
                    )
        return out, models[primary_i]
    except Exception as ex:
        return [
            Estimate(
                contrast=contrast_key,
                method=entry.name,
                n_treated=n_t,
                n_control=n_c,
                target_units=target_units,
                secondary=secondary,
                error=f"{type(ex).__name__}: {str(ex)[:300]}",
            )
        ], None


def _vcov_words(vcov: Any) -> str:
    if isinstance(vcov, dict):
        return "; ".join(f"{k} by {v}" for k, v in vcov.items())
    return str(vcov)


def _counts(panel: pd.DataFrame) -> tuple[int, int]:
    return int(panel.loc[panel["treated"] == 1, "unit"].nunique()), int(panel.loc[panel["treated"] == 0, "unit"].nunique())


@dataclass
class Fit:
    """What one estimator run yields: the estimates (the primary first, secondaries flagged), the fit object for follow-ups,
    the period-by-period coefficients when the entry reports them, the effect within each cohort, and the error if it failed."""

    estimates: list[Estimate] = field(default_factory=list)
    model: Any = None
    dynamic: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    by_cohort: list[Estimate] = field(default_factory=list)
    error: str | None = None

    @property
    def primary(self) -> Estimate:
        return next((e for e in self.estimates if not e.secondary), self.estimates[0])


def run(entry: EstimatorEntry, panel: pd.DataFrame, vcov: Any, contrast_key: str, target_units: str, *, controls: list[str], secondary: bool = False) -> Fit:
    """One estimator on the canonical panel, by its engine. Failures come back on the record, never as exceptions."""
    if entry.engine == "feols":
        ests, model = estimate(entry, formula_for(entry, controls), panel, vcov, contrast_key, target_units, secondary=secondary)
        f = Fit(estimates=ests, model=model, error=ests[0].error if ests and all(e.error for e in ests) else None)
        if model is not None and "dynamic" in entry.reports:
            f.dynamic = dynamic_coefficients(model)
        return f
    if entry.engine == "did2s":
        return _did2s(entry, panel, vcov, contrast_key, target_units, controls, secondary)
    if entry.engine == "lpdid":
        return _lpdid(entry, panel, vcov, contrast_key, target_units, secondary)
    return _saturated(entry, panel, vcov, contrast_key, target_units, controls, secondary)


def _failed(entry: EstimatorEntry, panel: pd.DataFrame, contrast_key: str, target_units: str, secondary: bool, ex: Exception) -> Fit:
    n_t, n_c = _counts(panel)
    msg = f"{type(ex).__name__}: {str(ex)[:300]}"
    return Fit(
        estimates=[Estimate(contrast=contrast_key, method=entry.name, n_treated=n_t, n_control=n_c, target_units=target_units, secondary=secondary, error=msg)],
        error=msg,
    )


def _did2s(entry: EstimatorEntry, panel: pd.DataFrame, vcov: Any, contrast_key: str, target_units: str, controls: list[str], secondary: bool) -> Fit:
    import pyfixest as pf

    n_t, n_c = _counts(panel)
    cluster = _cluster_of(vcov)
    try:
        m = pf.did2s(panel, yname="y", first_stage=first_stage_for(entry, controls), second_stage=entry.second_stage, treatment=COEF, cluster=cluster)
    except Exception as ex:
        return _failed(entry, panel, contrast_key, target_units, secondary, ex)
    if COEF in m.coef().index:
        lo, hi = (float(x) for x in m.confint().loc[COEF].to_numpy())
        est = Estimate(
            contrast=contrast_key,
            method=entry.name,
            value=float(m.coef()[COEF]),
            ci_low=lo,
            ci_high=hi,
            n_treated=n_t,
            n_control=n_c,
            target_units=target_units,
            secondary=secondary,
            p_value=float(m.pvalue()[COEF]),
            p_value_source=f"two-stage GMM, clustered by {cluster}",
        )
        return Fit(estimates=[est], model=m)
    dyn = dynamic_coefficients(m)
    lags = [k for k in dyn if k >= 0]
    if not lags:
        return _failed(entry, panel, contrast_key, target_units, secondary, ValueError("no post-period coefficient in the two-stage dynamic fit"))
    est = Estimate(
        contrast=contrast_key,
        method=entry.name,
        value=float(np.mean([dyn[k][0] for k in lags])),
        n_treated=n_t,
        n_control=n_c,
        target_units=target_units,
        secondary=True,  # the mean of the post-period coefficients is a summary, not the estimate the design answers with
    )
    return Fit(estimates=[est], model=m, dynamic=dyn)


def projection_windows(panel: pd.DataFrame, cap: int) -> tuple[int, int]:
    """How many horizons local projections may reach: the panel's own reach, capped; and, with no never-treated unit, only the
    post horizons at which every cohort but the last still has a unit not yet treated to compare with (the smallest gap between
    consecutive first-treated periods, less one)."""
    rel = panel.loc[panel["treated"] == 1, "rel_time"]
    pre_w, post_w = min(cap, max(int(-rel.min()), 1)), min(cap, max(int(rel.max()), 0))
    if not (panel["cohort"] == 0).any():
        starts = sorted(panel.loc[panel["cohort"] > 0, "cohort"].unique())
        gaps = [int(b - a) for a, b in zip(starts, starts[1:], strict=False)]
        post_w = max(0, min(post_w, (min(gaps) - 1) if gaps else 0))
    return -pre_w, post_w


def _lpdid_dynamic(panel: pd.DataFrame, kw: dict[str, Any], pre_w: int, post_w: int) -> tuple[dict[int, tuple[float, float, float]], tuple[int, int]]:
    """One regression per horizon; a horizon with no comparison left makes the library fail, so the window shrinks until it runs."""
    import pyfixest as pf

    lo, hi = pre_w, post_w
    while True:
        try:
            t = pf.lpdid(panel.copy(), att=False, **{**kw, "pre_window": lo, "post_window": hi}).tidy()
            dyn: dict[int, tuple[float, float, float]] = {}
            for name, r in t.iterrows():
                k = int(float(str(name).split("::")[-1]))
                dyn[k] = (float(r["Estimate"]), float(r["2.5%"]), float(r["97.5%"]))
            return dyn, (lo, hi)
        except Exception:
            if hi > 0:
                hi -= 1
            elif lo < -1:
                lo += 1
            else:
                return {}, (lo, hi)


def _lpdid(entry: EstimatorEntry, panel: pd.DataFrame, vcov: Any, contrast_key: str, target_units: str, secondary: bool) -> Fit:
    import pyfixest as pf

    n_t, n_c = _counts(panel)
    pre_w, post_w = projection_windows(panel, int(entry.params.get("max_window", 8)))
    cluster = _cluster_of(vcov)
    kw = dict(yname="y", idname="unit", tname="time_index", gname="cohort", vcov={"CRV1": cluster}, pre_window=pre_w, post_window=post_w)
    try:
        m = pf.lpdid(panel.copy(), att=True, **kw)  # the library writes into its input; give it a copy
        row = m.tidy().loc["treat_diff"]
        est = Estimate(
            contrast=contrast_key,
            method=entry.name,
            value=float(row["Estimate"]),
            ci_low=float(row["2.5%"]),
            ci_high=float(row["97.5%"]),
            n_treated=n_t,
            n_control=n_c,
            target_units=target_units,
            secondary=secondary,
            p_value=float(row["Pr(>|t|)"]),
            p_value_source=f"local projections over horizons {pre_w} to {post_w}, clustered by {cluster}",
        )
        dyn: dict[int, tuple[float, float, float]] = {}
        if "dynamic" in entry.reports:
            dyn, _ = _lpdid_dynamic(panel, kw, pre_w, post_w)
        return Fit(estimates=[est], model=m, dynamic=dyn)
    except Exception as ex:
        return _failed(entry, panel, contrast_key, target_units, secondary, ex)


def _lincomb(model, weights: dict[str, float]) -> tuple[float, float, float]:
    """A weighted sum of coefficients with its 95% interval from the fit's covariance."""
    names = list(model.coef().index)
    w = np.array([weights.get(n, 0.0) for n in names])
    coef = model.coef().to_numpy(dtype=float)
    v = np.asarray(model._vcov, dtype=float)
    value = float(w @ coef)
    se = float(np.sqrt(max(w @ v @ w, 0.0)))
    return value, value - Z * se, value + Z * se


def _saturated(entry: EstimatorEntry, panel: pd.DataFrame, vcov: Any, contrast_key: str, target_units: str, controls: list[str], secondary: bool) -> Fit:
    """The saturated event study: a coefficient per cohort and period relative to the change. The library's own pooled
    aggregates are not implemented in this version, so the effect on the treated, each cohort's effect and each period's
    effect are share-weighted sums of the post-change coefficients computed here, with intervals from the fit's covariance."""
    import pyfixest as pf

    n_t, n_c = _counts(panel)
    cluster = _cluster_of(vcov)
    try:
        m = pf.event_study(
            panel, yname="y", idname="unit", tname="time_index", gname="cohort", estimator="saturated", cluster=cluster, xfml=" + ".join(controls) or None
        )
        cells: dict[str, tuple[int, int]] = {}
        for name in m.coef().index:
            hit = SATURATED.search(name)
            if hit:
                cells[name] = (int(hit.group(1)), int(hit.group(2)))
        if not cells:
            raise ValueError("no cohort-by-period coefficient in the saturated fit")
        counts = panel[panel["treated"] == 1].groupby(["cohort", "rel_time"]).size()
        label_of = {int(i): period_label(v) for i, v in zip(panel["time_index"], panel["time"], strict=False)}

        def weighted(names: list[str]) -> tuple[float, float, float]:
            raw = {n: float(counts.get((cells[n][1], cells[n][0]), 0)) for n in names}
            total = sum(raw.values()) or 1.0
            return _lincomb(m, {n: v / total for n, v in raw.items()})

        post = [n for n, (k, _) in cells.items() if k >= 0]
        if not post:
            raise ValueError("no post-change coefficient in the saturated fit")
        value, lo, hi = weighted(post)
        se = (hi - lo) / (2 * Z)
        p = float(2 * (1 - _phi(abs(value / se)))) if se > 0 else 0.0
        est = Estimate(
            contrast=contrast_key,
            method=entry.name,
            value=value,
            ci_low=lo,
            ci_high=hi,
            n_treated=n_t,
            n_control=n_c,
            target_units=target_units,
            secondary=secondary,
            p_value=p,
            p_value_source=f"share-weighted post-change coefficients, clustered by {cluster}",
        )
        dyn = {k: weighted([n for n, (kk, _) in cells.items() if kk == k]) for k in sorted({k for k, _ in cells.values()})}
        by_cohort = []
        for g in sorted({g for _, g in cells.values()}):
            names = [n for n, (k, gg) in cells.items() if gg == g and k >= 0]
            if not names:
                continue
            v, lo_g, hi_g = weighted(names)
            n_g = int(panel.loc[panel["cohort"] == g, "unit"].nunique())
            by_cohort.append(
                Estimate(
                    contrast=contrast_key,
                    method=entry.name,
                    value=v,
                    ci_low=lo_g,
                    ci_high=hi_g,
                    n_treated=n_g,
                    n_control=n_c,
                    target_units=target_units,
                    modifier="cohort",
                    level=label_of.get(g, str(g)),
                )
            )
        return Fit(estimates=[est], model=m, dynamic=dyn, by_cohort=by_cohort)
    except Exception as ex:
        return _failed(entry, panel, contrast_key, target_units, secondary, ex)


def _phi(z: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1 + erf(z / sqrt(2)))


def cohort_heterogeneity(panel: pd.DataFrame, cluster: str = "unit") -> dict[str, Any]:
    """Whether the cohorts' effects differ: a joint test, on the saturated event study, that every cohort's share-weighted
    post-change effect equals the first cohort's. Computed here rather than by the library's own test, which in this version
    tests whether the post-change coefficients are zero at all. An error comes back as a fact."""
    import pyfixest as pf

    try:
        m = pf.event_study(panel, yname="y", idname="unit", tname="time_index", gname="cohort", estimator="saturated", cluster=cluster)
        names = list(m.coef().index)
        cells = {n: (int(h.group(1)), int(h.group(2))) for n in names if (h := SATURATED.search(n))}
        counts = panel[panel["treated"] == 1].groupby(["cohort", "rel_time"]).size()
        rows = []
        for g in sorted({g for _, g in cells.values()}):
            post = [n for n, (k, gg) in cells.items() if gg == g and k >= 0]
            raw = np.array([float(counts.get((g, cells[n][0]), 0)) for n in post])
            w = raw / (raw.sum() or 1.0)
            row = np.zeros(len(names))
            for n, wi in zip(post, w, strict=True):
                row[names.index(n)] = wi
            rows.append(row)
        if len(rows) < 2:
            return dict(error="fewer than two cohorts with a post-change coefficient")
        R = np.array([r - rows[0] for r in rows[1:]])
        t = m.wald_test(R=R, q=np.zeros(len(R)), distribution="chi2")
        return dict(statistic=float(t["statistic"]), p=float(t["pvalue"]), cohorts=len(rows))
    except Exception as ex:
        return dict(error=f"{type(ex).__name__}: {str(ex)[:200]}")


def dynamic_coefficients(model) -> dict[int, tuple[float, float, float]]:
    """rel_time -> (estimate, ci_low, ci_high) from a dynamic fit."""
    out = {}
    ci = model.confint()
    for name, val in model.coef().items():
        m = DYNAMIC.search(name)
        if m:
            k = int(m.group(1))
            out[k] = (float(val), float(ci.loc[name].iloc[0]), float(ci.loc[name].iloc[1]))
    return out


def leads_model(panel: pd.DataFrame, controls: list[str], vcov: Any, *, cohorts: int, never_treated: bool = True):
    """The dynamic fit whose lead coefficients test parallel paths: two-way fixed effects with one cohort; Gardner's two stages
    with several, so a unit already treated never serves as a control for one treated later and the leads stay clean. Without
    never-treated units the two stages cannot run in this version; see `leads_by_projection`."""
    import pyfixest as pf

    if cohorts > 1:
        first = "~ " + (" + ".join(controls) if controls else "0") + " | unit+time"
        return pf.did2s(panel, yname="y", first_stage=first, second_stage="~ i(rel_time, ref=-1)", treatment=COEF, cluster=_cluster_of(vcov))
    formula = "y ~ i(rel_time, treated, ref=-1)" + (" + " + " + ".join(controls) if controls else "") + " | unit+time"
    return fit(formula, panel, vcov)


def leads_by_projection(
    panel: pd.DataFrame, vcov: Any, *, max_window: int = 8
) -> tuple[tuple[float, float, int] | None, dict[int, tuple[float, float, float]]]:
    """The leads from local projections, one regression per horizon against units not yet treated, for a staggered panel with
    no never-treated unit: each lead's estimate and interval, and a Bonferroni-corrected joint p over the leads (the smallest
    lead p times their number, capped at one). Returns (statistic, p, k) or None when there is no lead, and the horizons."""
    from scipy.stats import norm

    pre_w, post_w = projection_windows(panel, max_window)
    kw = dict(yname="y", idname="unit", tname="time_index", gname="cohort", vcov={"CRV1": _cluster_of(vcov)})
    dyn, _ = _lpdid_dynamic(panel, kw, pre_w, post_w)
    ps: list[float] = []
    zs: list[float] = []
    for k, (est, lo, hi) in dyn.items():
        if k < 0:
            se = (hi - lo) / (2 * Z)
            z = est / se if se > 0 else 0.0
            zs.append(z)
            ps.append(float(2 * norm.sf(abs(z))))
    if not ps:
        return None, dyn
    return (float(max(abs(z) for z in zs)), float(min(1.0, min(ps) * len(ps))), len(ps)), dyn


def leads_test(model) -> tuple[float, float, int] | None:
    """Joint test that every pre-period coefficient is zero. (statistic, p_value, number of leads) or None."""
    names = list(model.coef().index)
    leads = [n for n in names if DYNAMIC.search(n) and int(DYNAMIC.search(n).group(1)) < 0]
    if not leads:
        return None
    R = np.zeros((len(leads), len(names)))
    for i, n in enumerate(leads):
        R[i, names.index(n)] = 1
    w = model.wald_test(R=R, q=np.zeros(len(leads)), distribution="chi2")
    return float(w["statistic"]), float(w["pvalue"]), len(leads)


def wild_bootstrap(model, *, vcov: Any, reps: int, seed: int, weights_type: str = "rademacher", bootstrap_type: str = "11") -> float | None:
    """The wild cluster bootstrap p-value for the effect, clustered as the fit was. The bootstrap needs a numeric cluster
    column, so the fit is redone on its own rows with the cluster coded as integers when it is not."""
    try:
        m = _numeric_cluster_fit(model, vcov)
        r = m.wildboottest(param=COEF, reps=int(reps), seed=int(seed), weights_type=weights_type, bootstrap_type=str(bootstrap_type))
        p = r.get("Pr(>|t|)") if hasattr(r, "get") else None
        return float(p) if p is not None else None
    except Exception:
        return None


def _numeric_cluster_fit(model, vcov: Any):
    """The fit redone on its own rows with the cluster coded as integers when it is not: the bootstrap code needs it so."""
    import pyfixest as pf

    col = _cluster_of(vcov) if isinstance(vcov, dict) else None
    data = getattr(model, "_data", None)
    if col is None or data is None or col not in data.columns or pd.api.types.is_numeric_dtype(data[col]):
        return model
    coded = data.copy()
    coded[col] = pd.factorize(coded[col])[0]
    return pf.feols(model._fml, coded, vcov=vcov)


def rwolf_p(models: list, *, vcov: Any, reps: int, seed: int) -> list[float] | None:
    """Romano-Wolf step-down p-values for the effect across a family of fits (the levels of one modifier), resampled by the wild
    cluster bootstrap, clustered as the fits were. None when the correction could not run; the fits keep their own p-values."""
    import pyfixest as pf

    try:
        coded = [_numeric_cluster_fit(m, vcov) for m in models]
        table = pf.rwolf(coded, COEF, reps=int(reps), seed=int(seed))
        row = table.loc["RW Pr(>|t|)"]
        return [float(row.iloc[i]) for i in range(len(models))]
    except Exception:
        return None


def ritest_collapsed(panel: pd.DataFrame, controls: list[str], *, reps: int, seed: int, kind: str = "randomization-c") -> tuple[float | None, float | None]:
    """Randomisation inference on one before and one after value per unit: the change in the outcome (and in each control that
    varies over time) from the periods before the unit's change to the periods after it, regressed on the treated label, with
    that label reassigned across units at random. Returns the p-value and the collapsed estimate. The panel's fixed effects
    cannot carry this test in this version of the library (a unit label permuted within units is singular), so it runs here."""
    import contextlib
    import io

    import pyfixest as pf

    rows = []
    for unit, g in panel.groupby("unit"):
        post, pre = g[g["post"] == 1], g[g["post"] == 0]
        if post.empty or pre.empty:
            continue
        row = {"unit": unit, "treated": int(g["treated"].iloc[0]), "dy": float(post["y"].mean() - pre["y"].mean())}
        for c in controls:
            if c in g.columns and pd.api.types.is_numeric_dtype(g[c]) and g[c].nunique() > 1:
                row[f"d_{c}"] = float(post[c].mean() - pre[c].mean())
        rows.append(row)
    col = pd.DataFrame(rows)
    terms = [c for c in col.columns if c.startswith("d_") and col[c].notna().all()]
    try:
        m = pf.feols("dy ~ treated" + ("".join(f" + {c}" for c in terms)), col, vcov="hetero")
        with _QUIET, contextlib.redirect_stderr(io.StringIO()):  # the redirect swaps process-wide state; workers must not interleave
            r = m.ritest(resampvar="treated", reps=int(reps), type=kind, rng=np.random.default_rng(int(seed)), choose_algorithm="slow")
        return float(r["Pr(>|t|)"]), float(m.coef()["treated"])
    except Exception:
        return None, None


def passes(pass_when: dict[str, Any], primary: Estimate, *, new: float | None, lo: float | None, hi: float | None, p: float | None) -> bool | None:
    """One falsification's verdict under its declared rule: an interval that covers zero, an interval that overlaps the primary's,
    a sign that holds when the primary excludes zero, a p below a line. None when the refit could not say (or the primary covers
    zero and the rule is informative only when it does not)."""
    primary_excludes_zero = primary.ci_low is not None and primary.ci_high is not None and not (primary.ci_low <= 0 <= primary.ci_high)
    if pass_when.get("informative_only_when_primary_excludes_zero") and not primary_excludes_zero:
        return None
    verdicts: list[bool] = []
    if "p_value_lt" in pass_when:
        if p is None:
            return None
        verdicts.append(p < float(pass_when["p_value_lt"]))
    if pass_when.get("interval_covers_zero"):
        if lo is None or hi is None:
            return None
        verdicts.append(lo <= 0 <= hi)
    if pass_when.get("interval_overlaps_primary"):
        if lo is None or hi is None or primary.ci_low is None or primary.ci_high is None:
            return None
        verdicts.append(lo <= primary.ci_high and primary.ci_low <= hi)
    if pass_when.get("sign_stable_when_primary_excludes_zero") and primary_excludes_zero:
        if new is None or primary.value is None:
            return None
        verdicts.append(np.sign(new) == np.sign(primary.value))
    return all(verdicts) if verdicts else None


def _point(formula: str, panel: pd.DataFrame, vcov: Any) -> tuple[float, float, float] | None:
    """One refit's effect and interval, or None when it could not run."""
    try:
        m = fit(formula, panel, vcov)
        m = m.to_list()[-1] if hasattr(m, "to_list") else m
        lo, hi = (float(x) for x in m.confint().loc[COEF].to_numpy())
        return float(m.coef()[COEF]), lo, hi
    except Exception:
        return None


def falsify(
    entry: PlaceboEntry,
    design_formula: str,
    panel: pd.DataFrame,
    vcov: Any,
    primary: Estimate,
    controls: list[str],
    contrast_key: str,
    *,
    excluded: int = 0,
    outcome_columns: list[str] | None = None,
) -> tuple[Refutation, dict[str, Any]]:
    """One falsification or sensitivity on the frozen design, by name: the design's formula with its controls, its rows, its
    inference. `excluded` is the anticipation window the design left out; `outcome_columns` the columns the change could not have
    moved. Returns the refutation and what a figure can draw."""
    name = entry.name
    if name == "placebo_group":
        r, draws = placebo_group(design_formula, panel, primary, entry, contrast_key)
        return r, {"draws": draws}
    if name == "placebo_timing":
        return placebo_timing(design_formula, panel, vcov, entry, primary, contrast_key), {}
    if name == "placebo_outcome":
        return placebo_outcome(design_formula, panel, vcov, entry, primary, contrast_key, outcome_columns or []), {}
    if name == "leave_one_out":
        return leave_one_out(design_formula, panel, vcov, entry, primary, contrast_key)
    if name == "anticipation_shift":
        return anticipation_shift(design_formula, panel, vcov, entry, primary, contrast_key, excluded=excluded), {}
    if name in ("unit_trends", "group_time_fe", "twfe_naive"):
        return sensitivity_refit(name, design_formula, panel, vcov, entry, contrast_key, controls), {}
    return Refutation(contrast=contrast_key, refuter=name, kind=entry.kind, detail=f"no falsification named {name!r} in the adapter"), {}


def _stepless(formula: str) -> str:
    """The design formula with its cumulative-controls operator flattened: a placebo refits the full design once."""
    m = re.search(r" \+ csw0\(([^)]*)\)", formula)
    return formula.replace(m.group(0), " + " + " + ".join(x.strip() for x in m.group(1).split(","))) if m else formula


def placebo_group(formula: str, panel: pd.DataFrame, primary: Estimate, entry: PlaceboEntry, contrast_key: str) -> tuple[Refutation, list[float]]:
    """Reassign the treated label across units at random and refit the design. p = share of |placebo| >= |observed|. Every
    placebo effect comes back too, so the spread can be drawn."""
    rng = np.random.default_rng(SEED)
    formula = _stepless(formula)
    labels = panel.groupby("unit")["treated"].first()
    draws = int(entry.params.get("draws", 200))
    effects: list[float] = []
    for _ in range(draws):
        perm = pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)
        p2 = panel.copy()
        p2["treated"] = p2["unit"].map(perm).astype(int)
        p2["treat"] = (p2["treated"] * p2["post"]).astype(float)
        pt = _point(formula, p2, "iid")
        if pt is not None:
            effects.append(pt[0])
    if not effects or primary.value is None:
        return Refutation(contrast=contrast_key, refuter=entry.name, kind="falsification", detail="no placebo fit succeeded"), []
    p = float(np.mean(np.abs(effects) >= abs(primary.value)))
    passed = passes(entry.pass_when, primary, new=float(np.mean(effects)), lo=None, hi=None, p=p)
    return Refutation(
        contrast=contrast_key,
        refuter=entry.name,
        kind="falsification",
        new_effect=float(np.mean(effects)),
        p_value=p,
        passed=passed,
        detail=f"{len(effects)} reassignments; share with an effect at least as large: {p:.2f}"
        + (
            " (pass)"
            if passed
            else " (FAIL: the observed effect is not unusual)"
            if passed is False
            else " (uninformative: the primary's own interval covers zero)"
        ),
    ), effects


def placebo_timing(formula: str, panel: pd.DataFrame, vcov: Any, entry: PlaceboEntry, primary: Estimate, contrast_key: str) -> Refutation:
    """Pre-period rows only, with a fake change where the entry says (the middle of the pre window). The effect should be about zero."""
    pre = panel[panel["post"] == 0].copy()
    times = sorted(pre["time"].unique())
    if len(times) < 3:
        return Refutation(contrast=contrast_key, refuter=entry.name, kind="falsification", detail="fewer than three pre periods")
    cut = times[len(times) // 2] if entry.params.get("cut", "middle") == "middle" else times[-1]
    pre["post"] = (pre["time"] >= cut).astype(int)
    pre["treat"] = (pre["treated"] * pre["post"]).astype(float)
    pre["rel_time"] = pre["time"].map({v: i for i, v in enumerate(times)}) - times.index(cut)
    pt = _point(_stepless(formula), pre, vcov)
    if pt is None:
        return Refutation(contrast=contrast_key, refuter=entry.name, kind="falsification", detail="the pre-period refit did not run")
    val, lo, hi = pt
    passed = passes(entry.pass_when, primary, new=val, lo=lo, hi=hi, p=None)
    return Refutation(
        contrast=contrast_key,
        refuter=entry.name,
        kind="falsification",
        new_effect=val,
        passed=passed,
        detail=f"fake change at {period_label(cut)}: effect {val:.3g} [{lo:.3g}, {hi:.3g}]"
        + (" (pass)" if passed else " (FAIL: a pre-period 'effect' that is not zero)"),
    )


def placebo_outcome(formula: str, panel: pd.DataFrame, vcov: Any, entry: PlaceboEntry, primary: Estimate, contrast_key: str, columns: list[str]) -> Refutation:
    """A column the change could not have moved as the outcome, one refit per column (at most the entry's cap); each interval should cover zero."""
    cols = [c for c in columns if c in panel.columns][: int(entry.params.get("max_columns", 2))]
    if not cols:
        return Refutation(contrast=contrast_key, refuter=entry.name, kind="falsification", detail="no column fixed before the change moves within a unit")
    parts, verdicts, values = [], [], []
    for c in cols:
        f = _stepless(formula).replace("y ~", f"{c} ~", 1)
        f = re.sub(rf" \+ {re.escape(c)}(?= \+| \|)", "", f)  # the placebo outcome is not also a control
        pt = _point(f, panel, vcov)
        if pt is None:
            parts.append(f"{c}: could not run")
            continue
        val, lo, hi = pt
        ok = passes(entry.pass_when, primary, new=val, lo=lo, hi=hi, p=None)
        verdicts.append(bool(ok))
        values.append(val)
        parts.append(f"{c}: effect {val:.3g} [{lo:.3g}, {hi:.3g}]" + ("" if ok else " (FAIL)"))
    passed = all(verdicts) if verdicts else None
    return Refutation(
        contrast=contrast_key,
        refuter=entry.name,
        kind="falsification",
        new_effect=float(np.mean(values)) if values else None,
        passed=passed,
        detail="; ".join(parts)
        + ("" if passed is None else " (pass)" if passed else " (FAIL: the design finds an effect on a column the change could not have moved)"),
    )


def leave_one_out(formula: str, panel: pd.DataFrame, vcov: Any, entry: PlaceboEntry, primary: Estimate, contrast_key: str) -> tuple[Refutation, dict[str, Any]]:
    """Drop each treated unit in turn and refit the design; the farthest refit is the new effect, the range is every refit."""
    units = sorted(panel.loc[panel["treated"] == 1, "unit"].unique())
    points: list[dict[str, Any]] = []
    for u in units:
        pt = _point(_stepless(formula), panel[panel["unit"] != u], vcov)
        points.append({"label": f"without {u}", "value": pt[0] if pt else None, "lo": pt[1] if pt else None, "hi": pt[2] if pt else None})
    fitted = [pt for pt in points if pt["value"] is not None]
    if not fitted or primary.value is None:
        return Refutation(contrast=contrast_key, refuter=entry.name, kind="falsification", detail="no leave-one-out refit ran"), {"points": points}
    far = max(fitted, key=lambda pt: abs(pt["value"] - primary.value))
    verdicts = [passes(entry.pass_when, primary, new=pt["value"], lo=pt["lo"], hi=pt["hi"], p=None) for pt in fitted]
    passed = all(v for v in verdicts if v is not None) if any(v is not None for v in verdicts) else None
    kept_sign = sum(1 for pt in fitted if np.sign(pt["value"]) == np.sign(primary.value))
    return Refutation(
        contrast=contrast_key,
        refuter=entry.name,
        kind="falsification",
        new_effect=float(far["value"]),
        range_low=float(min(pt["value"] for pt in fitted)),
        range_high=float(max(pt["value"] for pt in fitted)),
        passed=passed,
        detail=f"{len(fitted)} refits, each without one treated unit: effects from {min(pt['value'] for pt in fitted):.3g} to {max(pt['value'] for pt in fitted):.3g}; "
        f"the farthest, {far['label']}, gives {far['value']:.3g} [{far['lo']:.3g}, {far['hi']:.3g}]; {kept_sign} of {len(fitted)} keep the sign"
        + (" (pass)" if passed else " (FAIL: one unit carries the conclusion)" if passed is False else ""),
    ), {"points": points}


def anticipation_shift(
    formula: str, panel: pd.DataFrame, vcov: Any, entry: PlaceboEntry, primary: Estimate, contrast_key: str, *, excluded: int = 0
) -> Refutation:
    """A fake change `lead` periods before the real one (past the `excluded` periods the design left out), on the rows before the
    real change only; an effect there means units moved before the change."""
    lead = int(entry.params.get("lead", 1))
    pre = panel[panel["post"] == 0].copy()
    k = -(excluded + lead)  # the first period the fake change covers, relative to the real one
    pre["treat"] = ((pre["treated"] == 1) & (pre["rel_time"] >= k)).astype(float)
    if pre.loc[pre["treated"] == 1, "treat"].nunique() < 2:
        return Refutation(
            contrast=contrast_key, refuter=entry.name, kind="falsification", detail=f"no pre period left to test a lead of {lead} past the window"
        )
    pt = _point(_stepless(formula), pre, vcov)
    if pt is None:
        return Refutation(contrast=contrast_key, refuter=entry.name, kind="falsification", detail="the lead refit did not run")
    val, lo, hi = pt
    passed = passes(entry.pass_when, primary, new=val, lo=lo, hi=hi, p=None)
    return Refutation(
        contrast=contrast_key,
        refuter=entry.name,
        kind="falsification",
        new_effect=val,
        passed=passed,
        detail=f"a fake change {abs(k)} period(s) before the real one, on the periods before it: effect {val:.3g} [{lo:.3g}, {hi:.3g}]"
        + (" (pass)" if passed else " (FAIL: the outcome moved before the change)"),
    )


def sensitivity_refit(name: str, formula: str, panel: pd.DataFrame, vcov: Any, entry: PlaceboEntry, contrast_key: str, controls: list[str]) -> Refutation:
    """A refit that reports where the estimate lands under another structure, with no verdict: a trend per unit, period effects
    per group above the unit, or the two-way fixed effects the design did not use on a staggered panel."""
    base = _stepless(formula) if name != "twfe_naive" else "y ~ treat" + ("".join(f" + {c}" for c in controls)) + " | unit+time"
    if name == "unit_trends":
        f = base.replace("| unit+time", "| unit[time_index]+time")
    elif name == "group_time_fe":
        f = base.replace("| unit+time", "| unit+cluster^time")
    else:
        f = base
    pt = _point(f, panel, vcov)
    if pt is None:
        return Refutation(contrast=contrast_key, refuter=name, kind="sensitivity", detail=f"the refit did not run ({f})")
    val, lo, hi = pt
    return Refutation(
        contrast=contrast_key,
        refuter=name,
        kind="sensitivity",
        new_effect=val,
        range_low=lo,
        range_high=hi,
        passed=None,
        detail=f"{entry.in_words}: {val:.3g} [{lo:.3g}, {hi:.3g}]",
    )

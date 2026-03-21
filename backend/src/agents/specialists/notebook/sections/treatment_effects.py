"""Treatment effects section renderer.

Reports pipeline estimates and provides executable verification cells
for OLS, IPW, AIPW, and DML so users can independently reproduce results.
"""

import json

import numpy as np
from nbformat.v4 import new_code_cell, new_markdown_cell

from src.agents.base import AnalysisState

from ..helpers import deduplicate_effects


def _holm_bonferroni(pvals: list[float | None]) -> list[float | None]:
    """Apply Holm-Bonferroni correction to a list of p-values.

    Returns adjusted p-values in the same order. None entries are preserved.
    """
    valid = [(i, p) for i, p in enumerate(pvals) if p is not None]
    if len(valid) <= 1:
        return pvals

    valid_sorted = sorted(valid, key=lambda x: x[1])
    k = len(valid_sorted)

    adjusted = [None] * len(pvals)
    cumulative_max = 0.0
    for rank, (orig_idx, p) in enumerate(valid_sorted):
        adj = p * (k - rank)
        cumulative_max = max(cumulative_max, adj)
        adjusted[orig_idx] = min(cumulative_max, 1.0)

    return adjusted


def _resolve_covariates(state: AnalysisState) -> tuple[list[str], str]:
    """Resolve covariates using the pipeline's priority chain.

    Returns (covariates_list, source_label) where source_label describes
    which pipeline output the covariates were drawn from.
    """
    treatment = state.treatment_variable
    outcome = state.outcome_variable

    def _exclude_t_y(cols: list[str]) -> list[str]:
        return [c for c in cols if c != treatment and c != outcome]

    # Priority 1: DAG adjustment set (backdoor criterion)
    if state.proposed_dag and state.proposed_dag.adjustment_set:
        covs = _exclude_t_y(state.proposed_dag.adjustment_set)
        if covs:
            return covs, "DAG adjustment set (backdoor criterion)"

    # Priority 2: Confounder discovery agent ranked list
    if state.confounder_discovery:
        ranked = state.confounder_discovery.get("ranked_confounders", [])
        if ranked:
            if isinstance(ranked[0], dict):
                names = [c.get("variable", c.get("name", "")) for c in ranked]
            else:
                names = list(ranked)
            covs = _exclude_t_y([n for n in names if n])
            if covs:
                return covs, "confounder discovery agent"

    # Priority 3: Data profiler potential confounders
    if state.data_profile and state.data_profile.potential_confounders:
        covs = _exclude_t_y(state.data_profile.potential_confounders)
        if covs:
            return covs, "data profiler (potential confounders)"

    return [], "none (no confounders identified)"


def _binarization_code(state: AnalysisState) -> str:
    """Generate treatment binarization code consistent with the pipeline."""
    treatment = state.treatment_variable

    # Categorical with value mapping
    if state.treatment_encoding and state.treatment_encoding.value_mapping:
        mapping = state.treatment_encoding.value_mapping
        mapping_repr = repr(mapping)
        return (
            f"# Treatment encoding from pipeline (categorical → numeric)\n"
            f"_mapping = {mapping_repr}\n"
            f"T = df['{treatment}'].map(_mapping).values.astype(float)\n"
            f"_unmapped = np.isnan(T).sum()\n"
            f"if _unmapped > 0:\n"
            f"    print(f'Warning: {{_unmapped}} unmapped treatment values')\n"
            f"T_binary = T  # Already encoded\n"
            f"print(f'Treatment encoded via mapping: {{_mapping}}')\n"
        )

    # Pipeline stored a binarization threshold
    if state.treatment_binarization_threshold is not None:
        thr = state.treatment_binarization_threshold
        return (
            f"# Binarize continuous treatment at pipeline threshold\n"
            f"T = df['{treatment}'].values.astype(float)\n"
            f"T_binary = (T > {thr}).astype(int)\n"
            f"print(f'Treatment binarized at pipeline threshold ({thr:.4f}): "
            f"{{T_binary.sum()}} treated, {{len(T_binary) - T_binary.sum()}} control')\n"
        )

    # Default: auto-detect
    return (
        f"# Binarize treatment (auto-detect)\n"
        f"T = df['{treatment}'].values.astype(float)\n"
        f"if len(np.unique(T[~np.isnan(T)])) <= 2:\n"
        f"    T_binary = T.astype(int)\n"
        f"    print(f'Treatment already binary: {{T_binary.sum()}} treated, "
        f"{{len(T_binary) - T_binary.sum()}} control')\n"
        f"else:\n"
        f"    _median_t = np.median(T[~np.isnan(T)])\n"
        f"    T_binary = (T > _median_t).astype(int)\n"
        f"    print(f'Treatment binarized at median ({{_median_t:.4f}}): "
        f"{{T_binary.sum()}} treated, {{len(T_binary) - T_binary.sum()}} control')\n"
    )


def _make_ipw_cell(
    state: AnalysisState,
    covariates_json: str,
    pipeline_est: float,
    pipeline_se: float,
) -> str:
    """Generate IPW (Inverse Probability Weighting) verification cell."""
    return f'''# Verification: Inverse Probability Weighting (Hajek estimator)
from sklearn.linear_model import LogisticRegression

COVARIATES = {covariates_json}
covariates = [c for c in COVARIATES if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

{_binarization_code(state)}
Y = df['{state.outcome_variable}'].values.astype(float)

# Clean data
_mask = ~(np.isnan(T_binary) | np.isnan(Y))
for _c in covariates:
    _mask &= ~np.isnan(df[_c].values.astype(float))
T_clean, Y_clean = T_binary[_mask], Y[_mask]
X_clean = df.loc[_mask, covariates].values.astype(float)

print(f"IPW sample: N={{len(T_clean)}}, treated={{T_clean.sum():.0f}}, control={{(1-T_clean).sum():.0f}}")

# Propensity score model
ps_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
ps_model.fit(X_clean, T_clean)
ps = ps_model.predict_proba(X_clean)[:, 1]
ps = np.clip(ps, 0.01, 0.99)

# Hajek estimator (stabilized)
w1 = T_clean / ps
w0 = (1 - T_clean) / (1 - ps)
ate_ipw = (w1 * Y_clean).sum() / w1.sum() - (w0 * Y_clean).sum() / w0.sum()

# Bootstrap SE (200 iterations, re-estimate PS each time)
np.random.seed(42)
_n = len(T_clean)
_boot_ates = []
for _b in range(200):
    _idx = np.random.choice(_n, _n, replace=True)
    _Tb, _Yb, _Xb = T_clean[_idx], Y_clean[_idx], X_clean[_idx]
    if _Tb.sum() < 2 or (1 - _Tb).sum() < 2:
        continue
    try:
        _ps_b = LogisticRegression(max_iter=500, C=1.0, random_state=_b)
        _ps_b.fit(_Xb, _Tb)
        _psb = np.clip(_ps_b.predict_proba(_Xb)[:, 1], 0.01, 0.99)
        _w1b = _Tb / _psb
        _w0b = (1 - _Tb) / (1 - _psb)
        _ate_b = (_w1b * _Yb).sum() / _w1b.sum() - (_w0b * _Yb).sum() / _w0b.sum()
        _boot_ates.append(_ate_b)
    except Exception:
        pass

se_ipw = np.std(_boot_ates) if _boot_ates else float('nan')

# Effective sample size diagnostics
ess_treated = (w1.sum())**2 / (w1**2).sum()
ess_control = (w0.sum())**2 / (w0**2).sum()
print(f"\\nPropensity score range: [{{ps.min():.4f}}, {{ps.max():.4f}}]")
print(f"Effective sample size: treated={{ess_treated:.0f}}, control={{ess_control:.0f}}")

print(f"\\nPipeline IPW:      {pipeline_est:.4f} (SE: {pipeline_se:.4f})")
print(f"Verification IPW:  {{ate_ipw:.4f}} (SE: {{se_ipw:.4f}})")
print(f"Bootstrap iterations used: {{len(_boot_ates)}}/200")'''


def _make_aipw_cell(
    state: AnalysisState,
    covariates_json: str,
    pipeline_est: float,
    pipeline_se: float,
) -> str:
    """Generate AIPW (Augmented IPW / Doubly Robust) verification cell."""
    return f'''# Verification: Augmented IPW (Doubly Robust) with 5-fold cross-fitting
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold

COVARIATES = {covariates_json}
covariates = [c for c in COVARIATES if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

{_binarization_code(state)}
Y = df['{state.outcome_variable}'].values.astype(float)

# Clean data
_mask = ~(np.isnan(T_binary) | np.isnan(Y))
for _c in covariates:
    _mask &= ~np.isnan(df[_c].values.astype(float))
T_clean, Y_clean = T_binary[_mask], Y[_mask]
X_clean = df.loc[_mask, covariates].values.astype(float)

print(f"AIPW sample: N={{len(T_clean)}}, treated={{T_clean.sum():.0f}}, control={{(1-T_clean).sum():.0f}}")

# Cross-fitted AIPW
n = len(T_clean)
ps_hat = np.zeros(n)
mu1_hat = np.zeros(n)
mu0_hat = np.zeros(n)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(X_clean, T_clean):
    X_tr, T_tr, Y_tr = X_clean[train_idx], T_clean[train_idx], Y_clean[train_idx]
    X_te = X_clean[test_idx]

    # Propensity model
    _ps = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    _ps.fit(X_tr, T_tr)
    ps_hat[test_idx] = np.clip(_ps.predict_proba(X_te)[:, 1], 0.01, 0.99)

    # Outcome models (separate for treated/control)
    _m1 = LinearRegression()
    _m0 = LinearRegression()
    if T_tr.sum() >= 2:
        _m1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
    if (1 - T_tr).sum() >= 2:
        _m0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])

    mu1_hat[test_idx] = _m1.predict(X_te) if T_tr.sum() >= 2 else Y_tr[T_tr == 1].mean()
    mu0_hat[test_idx] = _m0.predict(X_te) if (1 - T_tr).sum() >= 2 else Y_tr[T_tr == 0].mean()

# AIPW influence function scores
score1 = T_clean * (Y_clean - mu1_hat) / ps_hat + mu1_hat
score0 = (1 - T_clean) * (Y_clean - mu0_hat) / (1 - ps_hat) + mu0_hat
aipw_scores = score1 - score0

ate_aipw = aipw_scores.mean()
se_aipw = np.std(aipw_scores) / np.sqrt(n)

# Diagnostics
_overlap = ((ps_hat > 0.1) & (ps_hat < 0.9)).mean()
print(f"\\nPropensity score overlap (0.1-0.9): {{_overlap:.1%}}")
print(f"Propensity score range: [{{ps_hat.min():.4f}}, {{ps_hat.max():.4f}}]")

print(f"\\nPipeline AIPW:      {pipeline_est:.4f} (SE: {pipeline_se:.4f})")
print(f"Verification AIPW:  {{ate_aipw:.4f}} (SE: {{se_aipw:.4f}})")'''


def _make_dml_cell(
    state: AnalysisState,
    covariates_json: str,
    pipeline_est: float,
    pipeline_se: float,
) -> str:
    """Generate Double/Debiased ML verification cell."""
    return f'''# Verification: Double/Debiased Machine Learning (Chernozhukov et al.)
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

COVARIATES = {covariates_json}
covariates = [c for c in COVARIATES if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

{_binarization_code(state)}
Y = df['{state.outcome_variable}'].values.astype(float)

# Clean data
_mask = ~(np.isnan(T_binary) | np.isnan(Y))
for _c in covariates:
    _mask &= ~np.isnan(df[_c].values.astype(float))
T_clean, Y_clean = T_binary[_mask], Y[_mask]
X_clean = df.loc[_mask, covariates].values.astype(float)

print(f"DML sample: N={{len(T_clean)}}, treated={{T_clean.sum():.0f}}, control={{(1-T_clean).sum():.0f}}")

# Cross-fitted residualization
n = len(T_clean)
Y_residuals = np.zeros(n)
T_residuals = np.zeros(n)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(X_clean, T_clean):
    X_tr, T_tr, Y_tr = X_clean[train_idx], T_clean[train_idx], Y_clean[train_idx]
    X_te, T_te, Y_te = X_clean[test_idx], T_clean[test_idx], Y_clean[test_idx]

    # Outcome nuisance: E[Y|X]
    _m_y = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    _m_y.fit(X_tr, Y_tr)
    Y_residuals[test_idx] = Y_te - _m_y.predict(X_te)

    # Treatment nuisance: E[T|X]
    _m_t = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    _m_t.fit(X_tr, T_tr)
    T_residuals[test_idx] = T_te - _m_t.predict_proba(X_te)[:, 1]

# Residualized regression: theta = cov(Y_res, T_res) / var(T_res)
theta_dml = np.dot(T_residuals, Y_residuals) / np.dot(T_residuals, T_residuals)

# Chernozhukov SE: sqrt(E[psi^2]) / sqrt(n) / sqrt(E[T_res^2])
psi = (Y_residuals - theta_dml * T_residuals) * T_residuals
J = np.mean(T_residuals ** 2)
se_dml = np.sqrt(np.mean(psi ** 2)) / (J * np.sqrt(n))

# Diagnostics
_y_r2 = 1 - np.var(Y_residuals) / np.var(Y_clean)
_t_r2 = 1 - np.var(T_residuals) / np.var(T_clean)
print(f"\\nOutcome nuisance R²: {{_y_r2:.4f}}")
print(f"Treatment nuisance R²: {{_t_r2:.4f}}")
print(f"T-residual mean: {{T_residuals.mean():.6f}} (should be ~0)")

print(f"\\nPipeline DML:      {pipeline_est:.4f} (SE: {pipeline_se:.4f})")
print(f"Verification DML:  {{theta_dml:.4f}} (SE: {{se_dml:.4f}})")'''


# Map method names to cell generators
_METHOD_ALIASES = {
    "ipw": "ipw",
    "inverse probability weighting": "ipw",
    "inverse_probability_weighting": "ipw",
    "aipw": "aipw",
    "augmented ipw": "aipw",
    "augmented_ipw": "aipw",
    "doubly robust": "aipw",
    "doubly_robust": "aipw",
    "double_ml": "dml",
    "double ml": "dml",
    "doubleml": "dml",
    "dml": "dml",
}


def render_treatment_effects(state: AnalysisState) -> list:
    """Report treatment effect estimation results with verification cells."""
    cells = []

    effects = deduplicate_effects(state.treatment_effects)

    md = "## Treatment Effect Estimation\n\n"
    md += "*Results from the Effect Estimator agent.*\n\n"
    md += f"**Treatment**: {state.treatment_variable}\n"
    md += f"**Outcome**: {state.outcome_variable}\n"
    md += f"**Methods applied**: {len(effects)}\n\n"
    cells.append(new_markdown_cell(md))

    # Caveat banner
    if effects:
        caveat_md = (
            "> **Interpretation Note**: These are *estimated* treatment effects from "
            "observational data. They rely on the assumption that all relevant confounders "
            "have been measured and adjusted for (no unmeasured confounding). "
            "Results should be interpreted as associations adjusted for observed covariates, "
            "not as proof of causation.\n\n"
        )
        cells.append(new_markdown_cell(caveat_md))

    # Multiple testing correction
    raw_pvals = [e.p_value for e in effects]
    adjusted_pvals = _holm_bonferroni(raw_pvals)
    n_tests = sum(1 for p in raw_pvals if p is not None)

    # Results table
    if effects:
        table_md = "### Results Summary\n\n"
        if n_tests > 1:
            table_md += "| Method | Estimand | Estimate | Std Error | 95% CI | p-value | Adjusted p |\n"
            table_md += "|--------|----------|----------|-----------|--------|---------|------------|\n"
            for e, adj_p in zip(effects, adjusted_pvals):
                pval = f"{e.p_value:.4f}" if e.p_value is not None else "N/A"
                adj_pval = f"{adj_p:.4f}" if adj_p is not None else "N/A"
                table_md += (
                    f"| {e.method} | {e.estimand} | {e.estimate:.4f} | "
                    f"{e.std_error:.4f} | [{e.ci_lower:.4f}, {e.ci_upper:.4f}] | {pval} | {adj_pval} |\n"
                )
            table_md += f"\n*p-values adjusted for multiple comparisons (Holm-Bonferroni, k={n_tests}).*\n\n"
        else:
            table_md += "| Method | Estimand | Estimate | Std Error | 95% CI | p-value |\n"
            table_md += "|--------|----------|----------|-----------|--------|--------|\n"
            for e in effects:
                pval = f"{e.p_value:.4f}" if e.p_value is not None else "N/A"
                table_md += (
                    f"| {e.method} | {e.estimand} | {e.estimate:.4f} | "
                    f"{e.std_error:.4f} | [{e.ci_lower:.4f}, {e.ci_upper:.4f}] | {pval} |\n"
                )
            table_md += "\n"
        cells.append(new_markdown_cell(table_md))

    # Per-method details
    for e in effects:
        if e.details or e.assumptions_tested:
            detail_md = f"#### {e.method} Details\n\n"
            if e.assumptions_tested:
                detail_md += "**Assumptions tested:**\n"
                for a in e.assumptions_tested:
                    detail_md += f"- {a}\n"
                detail_md += "\n"
            if e.details:
                detail_md += "**Diagnostics:**\n"
                for k, v in e.details.items():
                    if isinstance(v, float):
                        detail_md += f"- {k}: {v:.4f}\n"
                    elif not isinstance(v, (list, dict)):
                        detail_md += f"- {k}: {v}\n"
                detail_md += "\n"
            cells.append(new_markdown_cell(detail_md))

    # Forest plot
    if effects:
        cells.append(new_markdown_cell("### Treatment Effect Comparison"))
        results_json = json.dumps([
            {
                "method": e.method,
                "estimate": e.estimate,
                "ci_lower": e.ci_lower,
                "ci_upper": e.ci_upper,
            }
            for e in effects
        ])

        plot_code = f'''# Forest plot of treatment effect estimates
import json
results = json.loads('{results_json}')

fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 1.2)))

methods = [r['method'] for r in results]
estimates = [r['estimate'] for r in results]
ci_lower = [r['ci_lower'] for r in results]
ci_upper = [r['ci_upper'] for r in results]

y_pos = list(range(len(methods)))
xerr_lower = [e - l for e, l in zip(estimates, ci_lower)]
xerr_upper = [u - e for e, u in zip(estimates, ci_upper)]

# Traditional forest plot: point estimates with CI whiskers
ax.errorbar(estimates, y_pos, xerr=[xerr_lower, xerr_upper],
            fmt='o', color='steelblue', markersize=8, capsize=6,
            elinewidth=2, markeredgewidth=2)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero effect')
ax.set_yticks(y_pos)
ax.set_yticklabels(methods)
ax.set_xlabel('Treatment Effect Estimate')
ax.set_title('Forest Plot: Treatment Effect Estimates Across Methods')
ax.legend()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()'''
        cells.append(new_code_cell(plot_code))

    # ── Verification cells ─────────────────────────────────────
    if effects:
        covariates, cov_source = _resolve_covariates(state)

        # Filter to numeric-only from data profile
        if state.data_profile and state.data_profile.feature_types:
            numeric_types = {"numeric", "binary", "ordinal"}
            ft = state.data_profile.feature_types
            covariates = [c for c in covariates if ft.get(c) in numeric_types]

        covariates_json = json.dumps(covariates)
        method_set = {m.lower().replace("-", "_").replace(" ", "_") for m in [e.method for e in effects]}

        # Covariate source note
        cov_md = (
            "### Verification Cells\n\n"
            "The cells below independently re-estimate treatment effects using standard "
            "Python data-science packages (no backend imports). Each cell is self-contained.\n\n"
            f"**Covariates source**: {cov_source}\n"
            f"**Covariates ({len(covariates)})**: {', '.join(covariates) if covariates else 'none'}\n"
        )
        cells.append(new_markdown_cell(cov_md))

        # --- OLS verification (always emitted) ---
        cells.append(new_markdown_cell(
            "#### Verification: OLS Regression\n\n"
            "Run this cell to independently verify the OLS estimate."
        ))

        verify_code = f'''# Verification: OLS regression
import statsmodels.api as sm

COVARIATES = {covariates_json}
covariates = [c for c in COVARIATES if c in df.columns
              and pd.api.types.is_numeric_dtype(df[c])]

{_binarization_code(state)}
Y = df['{state.outcome_variable}'].values.astype(float)

# Clean data
all_cols = ['{state.treatment_variable}', '{state.outcome_variable}'] + covariates
df_clean = df[all_cols].dropna()
T_binary_clean = T_binary[:len(df_clean)] if len(T_binary) == len(df) else T_binary
Y_clean = df_clean['{state.outcome_variable}'].values.astype(float)

# Re-extract after dropna (safe approach)
_idx = df_clean.index
T_clean = T_binary[df.index.get_indexer(_idx)]
Y_clean = df_clean['{state.outcome_variable}'].values.astype(float)

if covariates:
    X = df_clean[covariates].values.astype(float)
    design = np.column_stack([np.ones(len(T_clean)), T_clean, X])
else:
    design = np.column_stack([np.ones(len(T_clean)), T_clean])

model = sm.OLS(Y_clean, design)
results = model.fit()

print(f"Verification OLS Results:")
print(f"  ATE:       {{results.params[1]:.4f}}")
print(f"  SE:        {{results.bse[1]:.4f}}")
print(f"  95% CI:    [{{results.conf_int()[1][0]:.4f}}, {{results.conf_int()[1][1]:.4f}}]")
print(f"  p-value:   {{results.pvalues[1]:.4f}}")
print(f"  R-squared: {{results.rsquared:.4f}}")
print(f"  N:         {{len(df_clean)}}")'''
        cells.append(new_code_cell(verify_code))

        # --- IPW verification (only if pipeline used IPW) ---
        _ipw_aliases = {"ipw", "inverse_probability_weighting", "inverse_probability_weighting"}
        ipw_effects = [e for e in effects if _METHOD_ALIASES.get(
            e.method.lower().replace("-", "_").replace(" ", "_"), ""
        ) == "ipw"]
        if ipw_effects:
            e = ipw_effects[0]
            cells.append(new_markdown_cell(
                "#### Verification: Inverse Probability Weighting\n\n"
                "Hajek estimator with bootstrap SEs (200 iterations)."
            ))
            cells.append(new_code_cell(
                _make_ipw_cell(state, covariates_json, e.estimate, e.std_error)
            ))

        # --- AIPW verification (only if pipeline used AIPW/DR) ---
        aipw_effects = [e for e in effects if _METHOD_ALIASES.get(
            e.method.lower().replace("-", "_").replace(" ", "_"), ""
        ) == "aipw"]
        if aipw_effects:
            e = aipw_effects[0]
            cells.append(new_markdown_cell(
                "#### Verification: Augmented IPW (Doubly Robust)\n\n"
                "5-fold cross-fitted AIPW with influence function SEs."
            ))
            cells.append(new_code_cell(
                _make_aipw_cell(state, covariates_json, e.estimate, e.std_error)
            ))

        # --- DML verification (only if pipeline used DML) ---
        dml_effects = [e for e in effects if _METHOD_ALIASES.get(
            e.method.lower().replace("-", "_").replace(" ", "_"), ""
        ) == "dml"]
        if dml_effects:
            e = dml_effects[0]
            cells.append(new_markdown_cell(
                "#### Verification: Double/Debiased ML\n\n"
                "Gradient boosting nuisance models with 5-fold cross-fitting."
            ))
            cells.append(new_code_cell(
                _make_dml_cell(state, covariates_json, e.estimate, e.std_error)
            ))

    return cells

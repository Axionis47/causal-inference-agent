"""IPW (Inverse Probability Weighting, Hajek) verification cell with bootstrap SE."""

from src.analysis.agents.base import AnalysisState

from .binarization import binarization_code


def make_ipw_cell(
    state: AnalysisState,
    covariates_json: str,
    pipeline_est: float,
    pipeline_se: float,
) -> str:
    return f'''# Verification: Inverse Probability Weighting (Hajek estimator)
from sklearn.linear_model import LogisticRegression

COVARIATES = {covariates_json}
covariates = [c for c in COVARIATES if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

{binarization_code(state)}
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

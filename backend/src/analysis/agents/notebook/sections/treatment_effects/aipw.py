"""AIPW (Augmented IPW / Doubly Robust) verification cell with 5-fold cross-fitting."""

from src.analysis.agents.base import AnalysisState

from .binarization import binarization_code


def make_aipw_cell(
    state: AnalysisState,
    covariates_json: str,
    pipeline_est: float,
    pipeline_se: float,
) -> str:
    return f'''# Verification: Augmented IPW (Doubly Robust) with 5-fold cross-fitting
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold

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

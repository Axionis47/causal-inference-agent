"""DML (Double/Debiased Machine Learning) verification cell with gradient boosting nuisances."""

from src.analysis.agents.base import AnalysisState

from .binarization import binarization_code


def make_dml_cell(
    state: AnalysisState,
    covariates_json: str,
    pipeline_est: float,
    pipeline_se: float,
) -> str:
    return f'''# Verification: Double/Debiased Machine Learning (Chernozhukov et al.)
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
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

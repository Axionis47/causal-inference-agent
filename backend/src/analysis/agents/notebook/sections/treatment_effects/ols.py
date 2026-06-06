"""OLS verification cell - always emitted alongside any treatment-effect output."""

from src.analysis.agents.base import AnalysisState

from .binarization import binarization_code


def make_ols_cell(state: AnalysisState, covariates_json: str) -> str:
    """Self-contained statsmodels OLS regression cell."""
    return f'''# Verification: OLS regression
import statsmodels.api as sm

COVARIATES = {covariates_json}
covariates = [c for c in COVARIATES if c in df.columns
              and pd.api.types.is_numeric_dtype(df[c])]

{binarization_code(state)}
Y = df['{state.outcome_variable}'].values.astype(float)

# Drop rows with NaN in any column we use. A boolean mask aligned to df's
# row positions is robust to non-default indexes; df.index.get_indexer would
# fail or silently misalign on duplicate or non-monotonic labels.
all_cols = ['{state.treatment_variable}', '{state.outcome_variable}'] + covariates
mask = df[all_cols].notna().all(axis=1).values & ~np.isnan(T_binary)
df_clean = df.loc[mask].reset_index(drop=True)
T_clean = T_binary[mask]
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

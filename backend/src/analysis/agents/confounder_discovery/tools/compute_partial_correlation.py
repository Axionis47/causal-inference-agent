"""compute_partial_correlation - linear-residual partial correlation."""

from scipy import stats

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "compute_partial_correlation",
    "description": "Compute partial correlation between var1 and var2, controlling for control_var. Helps distinguish confounders from mediators.",
    "parameters": {
        "var1": {"type": "string", "description": "First variable"},
        "var2": {"type": "string", "description": "Second variable"},
        "control_var": {"type": "string", "description": "Variable to control for"},
    },
}


async def handle(
    agent,
    state: AnalysisState,
    var1: str = "",
    var2: str = "",
    control_var: str = "",
    **kwargs,
) -> ToolResult:
    if agent._df is None:
        return ToolResult(status=ToolResultStatus.ERROR, error="No data loaded")

    for v in [var1, var2, control_var]:
        if v not in agent._df.columns:
            return ToolResult(status=ToolResultStatus.ERROR, error=f"Variable '{v}' not found")

    from sklearn.linear_model import LinearRegression

    X = agent._df[[var1, var2, control_var]].dropna()
    v1_vals = X[var1].values
    v2_vals = X[var2].values
    ctrl = X[control_var].values.reshape(-1, 1)

    v1_resid = v1_vals - LinearRegression().fit(ctrl, v1_vals).predict(ctrl)
    v2_resid = v2_vals - LinearRegression().fit(ctrl, v2_vals).predict(ctrl)

    partial_corr, pval = stats.pearsonr(v1_resid, v2_resid)

    result = {
        "var1": var1,
        "var2": var2,
        "control_var": control_var,
        "partial_correlation": float(partial_corr),
        "p_value": float(pval),
    }
    agent._investigation_log.append({
        "tool": "compute_partial_correlation",
        "args": {"var1": var1, "var2": var2, "control_var": control_var},
        "result": result,
    })
    return ToolResult(status=ToolResultStatus.SUCCESS, output=result)

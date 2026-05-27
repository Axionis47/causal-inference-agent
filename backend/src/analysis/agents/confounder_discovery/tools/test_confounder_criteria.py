"""test_confounder_criteria - three-criterion check filtered by causal role."""

from scipy import stats

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from ..helpers import classify_variable_role

SCHEMA = {
    "name": "test_confounder_criteria",
    "description": "Test if a variable meets the statistical criteria for being a confounder (associated with both T and Y)",
    "parameters": {
        "variable": {"type": "string", "description": "Variable to test"},
    },
}


async def handle(agent, state: AnalysisState, variable: str = "", **kwargs) -> ToolResult:
    if agent._df is None:
        return ToolResult(status=ToolResultStatus.ERROR, error="No data loaded")

    if variable not in agent._df.columns:
        return ToolResult(status=ToolResultStatus.ERROR, error=f"Variable '{variable}' not found")

    corr_t, pval_t = stats.pearsonr(agent._df[variable], agent._df[agent._treatment_var])
    corr_y, pval_y = stats.pearsonr(agent._df[variable], agent._df[agent._outcome_var])

    from src.config.settings import get_settings
    s = get_settings()
    corr_thresh = s.confounder_correlation_threshold
    prog_thresh = s.confounder_prognostic_threshold
    bal_thresh = s.balance_pvalue_threshold

    affects_treatment = abs(corr_t) > corr_thresh
    affects_outcome = abs(corr_y) > corr_thresh
    is_classic = affects_treatment and affects_outcome
    is_prognostic = abs(corr_y) > prog_thresh

    try:
        treated_mask = agent._df[agent._treatment_var] == 1
        if treated_mask.sum() == 0 or (~treated_mask).sum() == 0:
            treated_mask = agent._df[agent._treatment_var] > agent._df[agent._treatment_var].median()
        treated_vals = agent._df.loc[treated_mask, variable].dropna()
        control_vals = agent._df.loc[~treated_mask, variable].dropna()
        if len(treated_vals) > 2 and len(control_vals) > 2:
            _, balance_p = stats.ttest_ind(treated_vals, control_vals, equal_var=False)
            is_imbalanced = balance_p < bal_thresh
        else:
            is_imbalanced = False
            balance_p = 1.0
    except Exception:
        is_imbalanced = False
        balance_p = 1.0

    statistically_associated = is_classic or is_prognostic or is_imbalanced
    confounder_strength = abs(corr_y)

    causal_role = classify_variable_role(
        dag=state.proposed_dag,
        variable=variable,
        treatment=agent._treatment_var,
        outcome=agent._outcome_var,
        domain_knowledge=state.domain_knowledge if hasattr(state, "domain_knowledge") else None,
    )

    is_confounder = (
        statistically_associated
        and causal_role not in ("potential_mediator", "potential_collider")
    )

    reasons: list[str] = []
    if is_classic:
        reasons.append(f"correlated with T ({abs(corr_t):.3f}) and Y ({abs(corr_y):.3f})")
    if is_prognostic:
        reasons.append(f"prognostic variable (corr_Y={abs(corr_y):.3f})")
    if is_imbalanced:
        reasons.append(f"imbalanced across groups (p={balance_p:.3f})")

    result = {
        "variable": variable,
        "treatment_var": agent._treatment_var,
        "outcome_var": agent._outcome_var,
        "corr_with_treatment": float(corr_t),
        "pval_treatment": float(pval_t),
        "affects_treatment": affects_treatment,
        "corr_with_outcome": float(corr_y),
        "pval_outcome": float(pval_y),
        "affects_outcome": affects_outcome,
        "is_prognostic": is_prognostic,
        "is_imbalanced": is_imbalanced,
        "balance_p_value": float(balance_p),
        "reasons": reasons,
        "causal_role": causal_role,
        "is_confounder": is_confounder,
        "confounder_strength": float(confounder_strength) if is_confounder else 0.0,
    }
    agent._investigation_log.append({
        "tool": "test_confounder_criteria",
        "args": {"variable": variable},
        "result": result,
    })
    return ToolResult(status=ToolResultStatus.SUCCESS, output=result)

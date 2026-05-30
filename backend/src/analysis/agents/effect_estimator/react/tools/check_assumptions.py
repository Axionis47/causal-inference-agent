"""check_assumptions - method-specific assumption checks before run_method."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "check_assumptions",
    "description": "Check assumptions for a specific causal inference method.",
    "parameters": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["ols", "psm", "ipw", "aipw", "did", "iv", "rdd"],
                "description": "Method to check assumptions for",
            },
            "treatment_col": {
                "type": "string",
                "description": "Treatment column",
            },
            "outcome_col": {
                "type": "string",
                "description": "Outcome column",
            },
        },
        "required": ["method", "treatment_col", "outcome_col"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    method: str,
    treatment_col: str,
    outcome_col: str,
    **kwargs,
) -> ToolResult:
    if agent._df is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="No data loaded",
        )

    df = agent._df
    checks: dict = {}
    warnings: list[str] = []
    can_proceed = True

    if treatment_col not in df.columns:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=f"Treatment '{treatment_col}' not found",
        )
    if outcome_col not in df.columns:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=f"Outcome '{outcome_col}' not found",
        )

    T = df[treatment_col]

    n = len(df.dropna(subset=[treatment_col, outcome_col]))
    checks["sample_size"] = {"n": n, "sufficient": n >= 100}
    if n < 100:
        warnings.append("Small sample size may lead to unreliable estimates")

    n_treated = (T == 1).sum() if T.nunique() == 2 else (T > T.median()).sum()
    n_control = len(T) - n_treated
    checks["treatment_variation"] = {
        "n_treated": int(n_treated),
        "n_control": int(n_control),
        "ratio": f"{n_treated/n_control:.2f}" if n_control > 0 else "inf",
    }
    if n_treated < 20 or n_control < 20:
        warnings.append("Low treatment/control counts")
        can_proceed = False

    if method in ["psm", "ipw", "aipw"]:
        checks["overlap"] = "Requires propensity score overlap (will check during estimation)"
        checks["unconfoundedness"] = "Assumes no unmeasured confounders (untestable)"

    if method == "did":
        if not state.data_profile or not state.data_profile.has_time_dimension:
            warnings.append("DiD requires time dimension - not detected in data")
            can_proceed = False
        else:
            checks["parallel_trends"] = "Requires parallel trends assumption (partially testable)"

    if method == "iv":
        if not state.data_profile or not state.data_profile.potential_instruments:
            warnings.append("IV requires instruments - none detected")
            can_proceed = False
        else:
            checks["instruments"] = f"Found instruments: {state.data_profile.potential_instruments}"

    if method == "rdd":
        if not state.data_profile or not state.data_profile.discontinuity_candidates:
            warnings.append("RDD requires running variable - none detected")
            can_proceed = False

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "method": method,
            "checks": checks,
            "warnings": warnings,
            "can_proceed": can_proceed,
        },
    )

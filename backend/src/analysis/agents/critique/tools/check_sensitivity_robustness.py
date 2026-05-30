"""check_sensitivity_robustness - read state.sensitivity_results + label-based concern check."""

SCHEMA = {
    "name": "check_sensitivity_robustness",
    "description": "Check sensitivity analysis results for robustness to unmeasured confounding.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def handle(agent, **kwargs) -> str:
    state = agent._state

    output = "Sensitivity Analysis Review:\n"
    output += "=" * 50 + "\n"

    if not state.sensitivity_results:
        output += "No sensitivity analysis performed.\n"
        output += "RECOMMENDATION: Add sensitivity analysis to assess robustness.\n"
        agent._investigation_evidence.append("No sensitivity analysis - robustness unknown")
        return output

    output += f"Analyses performed: {len(state.sensitivity_results)}\n\n"

    robust = True
    for sens in state.sensitivity_results:
        output += f"{sens.method}:\n"
        output += f"  Result: {sens.robustness_value}\n"
        output += f"  Interpretation: {sens.interpretation}\n\n"

        interp_lower = sens.interpretation.lower()
        if any(x in interp_lower for x in ["not robust", "concern", "sensitive", "weak"]):
            robust = False

    if robust:
        output += "Overall: Results appear robust to unmeasured confounding.\n"
        agent._investigation_evidence.append("Sensitivity analysis supports robustness")
    else:
        output += "WARNING: Sensitivity analysis raises concerns about robustness.\n"
        agent._investigation_evidence.append(
            "CONCERN: Sensitivity analysis raises robustness concerns"
        )

    return output

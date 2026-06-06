"""verify_propensity_scores - inspect PS quality from state.ps_diagnostics + treatment_effects."""

SCHEMA = {
    "name": "verify_propensity_scores",
    "description": "Verify propensity score quality (overlap, calibration).",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def handle(agent, **kwargs) -> str:
    state = agent._state

    output = "Propensity Score Verification:\n"
    output += "=" * 50 + "\n"

    has_ps_results = False
    for effect in state.treatment_effects:
        if effect.details and "propensity_scores" in str(effect.details):
            has_ps_results = True
            break

    if not has_ps_results:
        output += "No propensity score results found in analysis.\n"
        output += "Consider: Were PS-based methods used (IPW, AIPW, Matching)?\n"
        agent._investigation_evidence.append("PS methods may not have been used")
        return output

    if hasattr(state, "ps_diagnostics") and state.ps_diagnostics:
        diag = state.ps_diagnostics
        output += "PS Diagnostics Available:\n"

        if hasattr(diag, "overlap"):
            output += f"  Overlap: {diag.overlap}\n"
        if hasattr(diag, "balance_achieved"):
            output += f"  Balance achieved: {diag.balance_achieved}\n"

        agent._investigation_evidence.append("Verified PS diagnostics")
    else:
        output += "PS diagnostics not stored separately.\n"
        output += "Check treatment effect details for PS quality metrics.\n"

    for effect in state.treatment_effects:
        if effect.method in ["IPW", "AIPW", "Matching"]:
            output += f"\n{effect.method} details:\n"
            if effect.assumptions_tested:
                output += f"  Assumptions: {effect.assumptions_tested}\n"
            if effect.details:
                for key, val in effect.details.items():
                    if any(x in key.lower() for x in ["overlap", "balance", "ps", "propensity"]):
                        output += f"  {key}: {val}\n"

    return output

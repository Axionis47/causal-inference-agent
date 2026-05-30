"""get_analysis_summary - markdown-style overview of the analysis so far."""

SCHEMA = {
    "name": "get_analysis_summary",
    "description": "Get overview of the analysis including methods used, estimates, and diagnostics.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def handle(agent, **kwargs) -> str:
    state = agent._state
    treatment_var, outcome_var = agent._resolve_treatment_outcome()

    output = "Analysis Summary:\n"
    output += "=" * 50 + "\n"

    if state.data_profile:
        output += "\nDataset:\n"
        output += f"  Samples: {state.data_profile.n_samples}\n"
        output += f"  Features: {state.data_profile.n_features}\n"
        output += f"  Treatment: {treatment_var or 'Unknown'}\n"
        output += f"  Outcome: {outcome_var or 'Unknown'}\n"

    if state.proposed_dag:
        output += "\nCausal Graph:\n"
        output += f"  Method: {state.proposed_dag.discovery_method}\n"
        output += f"  Nodes: {len(state.proposed_dag.nodes)}\n"
        output += f"  Edges: {len(state.proposed_dag.edges)}\n"

    output += f"\nTreatment Effect Estimates ({len(state.treatment_effects)}):\n"
    for effect in state.treatment_effects:
        if effect.p_value and effect.p_value < 0.001:
            sig = "***"
        elif effect.p_value and effect.p_value < 0.01:
            sig = "**"
        elif effect.p_value and effect.p_value < 0.05:
            sig = "*"
        else:
            sig = ""
        output += f"  {effect.method}: {effect.estimate:.4f} (SE={effect.std_error:.4f}) {sig}\n"
        output += f"    95% CI: [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]\n"

    if state.sensitivity_results:
        output += f"\nSensitivity Analysis ({len(state.sensitivity_results)}):\n"
        for sens in state.sensitivity_results:
            output += f"  {sens.method}: {sens.interpretation[:80]}...\n"

    if state.critique_history:
        output += f"\nPrevious Critiques ({len(state.critique_history)}):\n"
        for fb in state.critique_history:
            output += f"  Iteration {fb.iteration}: {fb.decision.value}\n"

    agent._investigation_evidence.append("Reviewed analysis summary")
    return output

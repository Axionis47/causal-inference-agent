"""classify_variable_role - infer treatment/outcome/confounder/mediator from semantics."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "classify_variable_role",
    "description": (
        "Classify a variable's causal role based on domain knowledge "
        "and metadata. Returns: treatment, outcome, confounder, mediator, "
        "collider, instrument, or covariate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "variable": {"type": "string", "description": "Variable name to classify"},
        },
        "required": ["variable"],
    },
}


async def handle(agent, state: AnalysisState, variable: str = "", **kwargs) -> ToolResult:
    semantic_result = await agent._analyze_variable_semantics(state, variable)
    semantic = semantic_result.output if semantic_result.status == ToolResultStatus.SUCCESS else {}

    role = "covariate"
    confidence = "medium"
    reasoning: list[str] = []

    if state.treatment_variable and variable == state.treatment_variable:
        role = "treatment"
        confidence = "high"
        reasoning.append("Designated as treatment variable")
    elif state.outcome_variable and variable == state.outcome_variable:
        role = "outcome"
        confidence = "high"
        reasoning.append("Designated as outcome variable")
    elif semantic.get("is_likely_immutable"):
        role = "confounder"
        confidence = "high"
        reasoning.append(f"Immutable variable: {semantic.get('causal_constraints', [])}")
    elif "treatment" in semantic.get("likely_role", []):
        role = "treatment_candidate"
        confidence = "medium"
        reasoning.append("Name suggests treatment")
    elif "outcome" in semantic.get("likely_role", []):
        role = "outcome_candidate"
        confidence = "medium"
        reasoning.append("Name suggests outcome")

    temporal = semantic.get("temporal_position", "unknown")
    if temporal == "pre":
        if role == "covariate":
            role = "confounder"
        reasoning.append("Pre-treatment timing")
    elif temporal == "post":
        if role == "covariate":
            role = "potential_mediator"
        reasoning.append("Post-treatment timing")

    agent._variable_roles[variable] = role

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "variable": variable,
            "role": role,
            "confidence": confidence,
            "reasoning": reasoning,
            "semantic_analysis": semantic,
        },
    )

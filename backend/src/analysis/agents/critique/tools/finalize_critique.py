"""finalize_critique - schema only. The agentic loop short-circuits when the LLM calls this.

No `handle` function: the loop in agent.py detects this tool name in
pending_calls and returns the LLM's arguments directly as the final
result. Listed in SCHEMAS but excluded from BY_NAME.
"""

SCHEMA = {
    "name": "finalize_critique",
    "description": "Finalize your critique with scores and decision. Call this after investigation.",
    "parameters": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {
                    "statistical_validity": {"type": "integer", "minimum": 1, "maximum": 5},
                    "assumption_checking": {"type": "integer", "minimum": 1, "maximum": 5},
                    "method_selection": {"type": "integer", "minimum": 1, "maximum": 5},
                    "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
                    "reproducibility": {"type": "integer", "minimum": 1, "maximum": 5},
                    "interpretation": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "description": "Scores for each dimension (1-5)",
            },
            "decision": {
                "type": "string",
                "enum": ["APPROVE", "ITERATE", "REJECT"],
                "description": "Overall decision",
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific issues found",
            },
            "improvements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Actionable improvements",
            },
            "evidence_summary": {
                "type": "string",
                "description": "Summary of evidence gathered through investigation",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed reasoning for decision",
            },
        },
        "required": ["scores", "decision", "reasoning"],
    },
}

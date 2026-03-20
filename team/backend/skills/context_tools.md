# Skill: Context Tools

Load when: working with ContextTools mixin or pull-based context in agents.

## What ContextTools does

The ContextTools mixin registers pull-based query tools on a ReActAgent.
Instead of dumping all state into the prompt, agents call these tools
to fetch specific information on demand.

Source: backend/src/agents/base/context_tools.py

## Available context tools (12)

| Tool | Purpose | Token savings |
|------|---------|--------------|
| ask_domain_knowledge | Search domain knowledge by question | Avoids dumping all domain context |
| get_column_info | Stats for a specific column | ~100 tokens vs full profile |
| get_dataset_summary | Basic dataset stats | ~200 tokens summary |
| list_columns | All column names | Quick reference |
| get_eda_finding | Query EDA by topic | Targeted retrieval |
| get_previous_finding | Get prior agent's findings | Cross-agent context |
| get_treatment_outcome | Current T/Y variables | 2 variable names |
| get_confounder_analysis | Ranked confounders | Top-N only, ~150 tokens |
| get_profile_for_variables | Stats for specific columns | ~200 vs ~2000 for full profile |
| get_dataset_context | Semantic metadata | Domain, tags, descriptions |
| analyze_variable_semantics | Variable role analysis | Prevents misidentification |
| get_dag_adjustment_set | Adjustment set from DAG | Backdoor criterion result |

## How to add a new context tool

```python
# In context_tools.py

def _register_context_tools(self):
    # ... existing tools ...

    self.register_tool(
        name="my_context_query",
        description="Get [specific info] from state",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up"},
            },
            "required": ["query"],
        },
        handler=self._handle_my_context_query,
    )

async def _handle_my_context_query(self, state, query: str) -> ToolResult:
    # Pull specific info from state, not dump everything
    relevant_data = extract_what_agent_needs(state, query)
    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output=relevant_data,
    )
```

## Rules

1. Context tools return the MINIMUM information needed
2. Never return the full DataProfile — use get_profile_for_variables for specific columns
3. Never return all confounders — return top-N ranked by strength
4. Always handle missing state gracefully (return "not available yet")
5. Context tools must be fast — no computation, just state lookups
6. Every agent that inherits ContextTools must call self._register_context_tools() in __init__

# Skill: Agent Design

Load when: designing new agents, defining state contracts, tool schemas, or ReAct behavior.

## Agent design checklist

Before writing code, define these:

```
Agent name:        [registered name for @register_agent]
Base class:        ReActAgent + ContextTools (typical) or BaseAgent (simple)
Pipeline position: After [X], before [Y]
MAX_STEPS:         [10-20, based on task complexity]

State contract:
  REQUIRED_STATE_FIELDS: [what it reads]
  WRITES_STATE_FIELDS:   [what it produces — new field on AnalysisState]

Tools (domain-specific):
  1. [tool_name] — [what it does] — parameters: [list]
  2. [tool_name] — [what it does] — parameters: [list]
  3. finalize_[task] — [writes results to state] — parameters: [results dict]

Context tools needed:
  [which ContextTools does this agent need? all of them or a subset?]
```

## State contract rules

1. Every new agent adds at most ONE new field to AnalysisState
2. The field must have a default of None (optional)
3. The field should be a typed dict or Pydantic model, not a raw string
4. Two agents must NEVER write the same field (except proposed_dag which DAGExpert refines)
5. REQUIRED_STATE_FIELDS must match what upstream agents actually produce

## Tool design rules

1. Every tool returns ToolResult(status, output, error, metadata)
2. Tools that read data should be cheap (fast, no LLM call)
3. Tools that compute should handle errors gracefully (return ToolResult.ERROR, not raise)
4. Every agent needs a finalize tool that writes results to state
5. Tool parameter schemas use JSON Schema format
6. Tool descriptions should be clear enough for the LLM to choose correctly

## ReAct loop behavior

### Initial observation
Keep it lean. 150-200 tokens max. Tell the agent:
- What job this is
- What the key variables are (or "use tools to find out")
- What context tools are available
- What to do first

### Convergence
- Agent calls finish() when task is done
- Agent auto-stops at MAX_STEPS
- 3 consecutive errors trigger early stop
- reflect() tool available for self-assessment when stuck

### Common design mistakes

| Mistake | Consequence | Fix |
|---------|------------|-----|
| Too many tools (>10) | LLM confused about which to call | Merge related tools, remove redundant ones |
| No finalize tool | Agent loops without writing state | Always include a finalize tool |
| MAX_STEPS too low | Agent runs out of steps before finishing | Set to at least 12 for complex tasks |
| Initial observation too large | Token waste, irrelevant context | Use lean observation + ContextTools |
| Missing ContextTools | Agent can't access upstream results | Add ContextTools mixin + _register_context_tools() |

## Parallel safety

If the new agent could run in parallel with another:
- Their WRITES_STATE_FIELDS must not overlap
- Neither reads what the other writes
- The orchestrator's system prompt must mention the parallel group

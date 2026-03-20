# Skill: Write Agent System Prompts

Load when: creating or iterating system prompts for ReAct agents.

## Prompt location

System prompts are class attributes on each agent:

```python
class MyAgent(ReActAgent, ContextTools):
    SYSTEM_PROMPT = """You are an expert at [domain].
    Your task is to [specific objective]."""
```

Not external files. The prompt lives with the agent code.

## Prompt structure for ReAct agents

```
You are an expert [role] specializing in [domain].

Your task: [specific objective for this pipeline stage]

Available tools:
[tools are auto-injected by the ReAct loop — do NOT list them in the prompt]

Strategy:
1. [first thing to do — usually inspect data]
2. [analysis steps]
3. [what to check before finalizing]
4. Call finish() when done with a summary of findings.

Rules:
- [specific constraints]
- Use context query tools to pull information on demand
- Do not assume data characteristics — check with tools first
- Call reflect() if stuck to reassess your approach
```

## Prompt quality checklist for causal agents

- [ ] Role mentions specific causal expertise (not generic "data scientist")
- [ ] Task objective is precise: what estimand, what to produce, what to write to state
- [ ] Strategy mentions checking data characteristics BEFORE choosing methods
- [ ] Rules prevent common causal mistakes (conditioning on colliders, ignoring assumptions)
- [ ] Prompt does NOT dump data into the system instruction (use ContextTools instead)
- [ ] Prompt does NOT list tool schemas (the ReAct loop injects those)
- [ ] Prompt mentions when to call finish() and what summary to provide

## Common failure modes in causal agent prompts

| Failure | Cause | Fix |
|---------|-------|-----|
| Agent runs all methods blindly | Prompt says "run all available methods" | Say "select methods based on data characteristics" |
| Agent ignores assumptions | No instruction to check assumptions | Add "validate assumptions before interpreting results" |
| Agent conditions on mediators | No warning about mediator bias | Add "do not adjust for variables on the causal path" |
| Agent dumps too much context | Prompt tells agent to "summarize everything" | Say "report key findings, use finish() summary" |
| Agent loops without converging | No convergence criteria | Add "if 3 consecutive tools fail, call reflect()" |

## Orchestrator prompt — special rules

The orchestrator prompt is the most critical. It defines pipeline order and agent dispatch.
When updating it:
- List all registered agent names with one-line descriptions
- Specify which agents can run in parallel
- Specify the critique loop behavior
- Never hardcode data-dependent routing (let the LLM reason)

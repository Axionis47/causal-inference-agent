# Skill: Cross-Reference Checker

Load when: verifying docs match code after changes.

## What to cross-reference

### Agent catalog (docs/agents.md)
- Does the agent count match the registry? (`create_all_agents()` count)
- Are all registered agent names listed?
- Do REQUIRED_STATE_FIELDS and WRITES_STATE_FIELDS match the code?
- Are all tools listed for each agent?
- Are MAX_STEPS values correct?

### Pipeline flow (docs/pipeline.md)
- Does the pipeline order match the orchestrator's system prompt?
- Are parallel dispatch groups correct?
- Are conditional steps (DAGExpert if DAG exists) still accurate?

### State schema (docs/state-schema.md)
- Do all fields listed match AnalysisState class?
- Are there fields in the code not in the docs?
- Are there fields in the docs not in the code?

### Method catalog (docs/causal-methods.md)
- Does the method count match the registry?
- Are all registered method names listed?
- Are assumptions and estimands correct?

### API reference (docs/api-reference.md)
- Do all endpoints listed match the FastAPI routes?
- Are request/response schemas current?

### Configuration (docs/configuration.md)
- Do all env vars listed match settings.py fields?
- Are default values correct?

## How to check

```
1. Read the doc
2. Read the corresponding code
3. Diff: what's in doc but not in code? What's in code but not in doc?
4. Update the doc to match code (code is truth)
```

## Output format

```markdown
## Cross-Reference Report

### Mismatches Found
| Document | Section | Issue | Action |
|----------|---------|-------|--------|
| agents.md | Agent count | Says 13, registry has 12 | Update to 12 |
| pipeline.md | Step 5 | Says DAGExpert, code skips if no DAG | Add conditional |
```

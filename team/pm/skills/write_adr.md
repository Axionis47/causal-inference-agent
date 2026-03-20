# Skill: Architecture Decision Records

Load when: a decision affects system architecture, agent interfaces, or method selection.

## ADR format

```markdown
# ADR-XXX: [Decision Title]

Date: YYYY-MM-DD
Status: accepted | superseded by ADR-YYY
Agents affected: [list]

## Context
[Why this decision is needed. What problem are we solving.]

## Decision
[What we decided. Be specific.]

## Consequences
[What changes as a result. What tradeoffs we accept.]

## Alternatives Considered
[What we rejected and why.]
```

## When an ADR is required

- Adding or removing an agent from the pipeline
- Changing agent dispatch order or parallel grouping
- Changing the state contract (adding/removing AnalysisState fields)
- Adding or removing a causal estimation method
- Changing LLM provider or model
- Changing the storage backend
- Modifying the critique loop behavior
- Any change that affects 2+ agents' interfaces

## When an ADR is NOT required

- Bug fixes within a single agent
- Adding a tool to an existing agent
- Prompt iteration within an agent's SYSTEM_PROMPT
- Test additions
- Documentation updates

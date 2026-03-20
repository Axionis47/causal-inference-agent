# Skill: Living Documentation

Load when: maintaining documentation that reflects the current state of the system.

## The documentation map

```
docs/
  architecture.md      <- System design, component boundaries, class hierarchy
  agents.md            <- All 13 agents: tools, state contracts, ReAct pattern
  pipeline.md          <- Step-by-step pipeline flow, state transitions
  causal-methods.md    <- 12 estimation methods, 5 discovery algorithms
  api-reference.md     <- REST API endpoints, request/response schemas
  state-schema.md      <- AnalysisState fields, data models
  configuration.md     <- Environment variables, provider setup
  development.md       <- Adding agents, methods, providers; testing patterns
  CONTEXT_ENGINEERING.md <- Token optimization, pull-based context design
```

## When to update each document

| Document | Update when |
|----------|------------|
| architecture.md | Component boundaries change, new layers added |
| agents.md | Agent added/removed, tools changed, state contracts changed |
| pipeline.md | Pipeline order changes, new parallel opportunities, new error paths |
| causal-methods.md | Method added/removed, assumptions updated |
| api-reference.md | Endpoints added/changed, schemas updated |
| state-schema.md | AnalysisState fields added/changed |
| configuration.md | New env vars, provider changes |
| development.md | Patterns change, new extension points |
| CONTEXT_ENGINEERING.md | Token savings change, new context tools added |

## How to update docs

1. Read the actual code first. Never update docs from memory.
2. Check the agent registry: how many agents are registered now?
3. Check the method registry: how many methods are registered now?
4. Verify pipeline order matches the orchestrator's system prompt
5. Update docs to match code reality. Not the other way around.

## The 5-minute rule

Every document must be readable in under 5 minutes.
If it takes longer, it's too detailed. Move details to a linked document.

## Consistency checks

After any doc update, verify:
- [ ] Agent count in agents.md matches registry count
- [ ] Method count in causal-methods.md matches registry count
- [ ] Pipeline order in pipeline.md matches orchestrator system prompt
- [ ] State fields in state-schema.md match AnalysisState class
- [ ] API schemas in api-reference.md match Pydantic schemas

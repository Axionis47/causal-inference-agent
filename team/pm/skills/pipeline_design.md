# Skill: Pipeline Design

Load when: reviewing or redesigning agent pipeline flow and dispatch order.

## Current pipeline — 14 agents in sequence/parallel

```
1. DomainKnowledge (conditional — metadata available?)
2. DataProfiler (always)
3. DataRepair (conditional — quality issues found?)
4. EDA + CausalDiscovery (PARALLEL)
5. DAGExpert (conditional — DAG produced?)
6. ConfounderDiscovery
7. PSDiagnostics
8. EffectEstimator
9. SensitivityAnalyst
10. Critique -> APPROVE | ITERATE (max 3) | REJECT
11. NotebookGenerator
```

## Design principles for pipeline changes

### Data dependencies define order
An agent can only run after its REQUIRED_STATE_FIELDS are populated.
Never dispatch an agent before its upstream dependency has written state.

```
DataProfiler writes: data_profile, dataframe_path
  -> EDA reads: data_profile, dataframe_path
  -> CausalDiscovery reads: data_profile, dataframe_path
  -> These two are independent, so they run in PARALLEL
```

### Parallel dispatch rules
Two agents can run in parallel ONLY if:
1. Their REQUIRED_STATE_FIELDS are both already populated
2. Their WRITES_STATE_FIELDS do not overlap
3. Neither reads what the other writes

Merge strategy: deep copy per agent, gather, copy back by WRITES_STATE_FIELDS.

### Where new agents can be inserted

| Position | After | Before | Reads | Writes |
|----------|-------|--------|-------|--------|
| Post-profiling | DataProfiler | EDA/Discovery | data_profile | new_field |
| Post-discovery | DAGExpert | ConfounderDiscovery | proposed_dag | new_field |
| Post-estimation | EffectEstimator | SensitivityAnalyst | treatment_effects | new_field |
| Post-sensitivity | SensitivityAnalyst | Critique | sensitivity_results | new_field |

### What NOT to do

- Do not add agents that write the same state field as an existing agent
- Do not create circular dependencies between agents
- Do not bypass the orchestrator's LLM routing with hardcoded dispatch
- Do not add agents that require state fields from multiple parallel branches

## Pipeline critique checklist

- [ ] Every agent's REQUIRED_STATE_FIELDS are populated by an upstream agent
- [ ] Parallel agents have non-overlapping WRITES_STATE_FIELDS
- [ ] No circular dependencies
- [ ] Critique loop has bounded iterations (max 3)
- [ ] Every agent has a timeout (default 300s)
- [ ] Failure in one agent does not silently corrupt downstream agents
- [ ] Pipeline produces a notebook even on partial failure

## Known pipeline issues to watch

1. Orchestrator does not validate REQUIRED_STATE_FIELDS before dispatch
2. Treatment/outcome variable resolution is inconsistent across agents
3. Critique agent can loop without converging (auto-finalizes at MAX_STEPS)
4. Double finalization possible if orchestrator LLM loop doesn't terminate cleanly

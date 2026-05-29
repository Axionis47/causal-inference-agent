# Agent Contracts

Canonical reference for what each analysis-stage agent delivers, which
state it needs, when it refuses to run, and which flags it can raise.

Each entry mirrors the `AgentCapability` declaration on the agent's
Python class (see `backend/src/domain/briefs.py`). When the capability
declaration changes, this doc must change in the same PR.

See `docs/adr-002-orchestrator-worker-pattern.md` for the design
rationale and `CLAUDE.md` §1 for the working-mode rules.

## How to read an entry

- **Answers**: the one-line question the agent owns. The orchestrator
  uses this to assemble its capability menu.
- **Needs**: top-level state keys that must be present before the
  agent will run. If any are missing the agent returns a refusal
  brief with `Flag.NEEDS_NOT_MET`.
- **Delivers**: top-level state keys the agent writes on success.
  These appear in `brief.artifact_keys`.
- **Refuses when**: human-readable conditions that produce a refusal
  brief. Each entry should correspond to a testable criterion.
- **Flags raised**: the closed-enum kinds this agent can put on its
  brief. Workers may not invent flags outside this list.
- **Success criteria**: testable assertions on the brief, referenced
  by id from the agent's unit tests. A criterion without a test is a
  coverage gap.

## Status: which agents have been migrated

Phase 1 onward fills this table in as each agent gets its capability
declaration and AgentBrief return. The contract layer is additive:
no agent's internal reasoning pattern changes during migration. The
"Pattern (current)" column is informational only, so the brief-
production code knows where in the agent to plug in.

| Agent | Migrated | Pattern (current) | Phase |
|---|---|---|---|
| `data_profiler` | no | CoT | 3b |
| `data_repair` | no | Full ReAct | 3c |
| `eda_agent` | no | CoT | 3a |
| `confounder_discovery` | no | Full ReAct | 3c |
| `causal_discovery` | no | CoT | 3a |
| `dag_expert` | no | Full ReAct | 3c |
| `ps_diagnostics` | no | Full ReAct | 1 (reference) |
| `effect_estimator` | no | Full ReAct | 3d |
| `sensitivity_analyst` | no | CoT | 3a |
| `critique` | no | CoT | 3e |
| `domain_knowledge` | no | CoT | 3a |
| `notebook_generator` | no | Direct + per-section CoT | 3e |

---

## Template (one section per agent, copied at migration time)

```
## <agent_name>

**Answers:** <one-line question>

**Needs:**
- `<state_key_1>` (<short description of expected shape>)
- `<state_key_2>` (...)

**Delivers:**
- `<state_key_written>` (...)

**Refuses when:**
- <condition> -> `<FLAG_NAME>`

**Flags raised:**
- `<FLAG_NAME>` -- <when>

**Success criteria:**
- `<criterion.id>` -- <description>
```

---

<!-- Per-agent sections are added below as agents migrate. Order is
fixed (matches the table above) so diffs are easy to read. -->

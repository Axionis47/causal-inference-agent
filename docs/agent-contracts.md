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
| `eda_agent` | yes | Full ReAct | 3a |
| `confounder_discovery` | no | Full ReAct | 3c |
| `causal_discovery` | no | CoT | 3a |
| `dag_expert` | no | Full ReAct | 3c |
| `ps_diagnostics` | yes | Full ReAct | 1 (reference) |
| `effect_estimator` | no | Full ReAct | 3d |
| `sensitivity_analyst` | no | CoT | 3a |
| `critique` | no | CoT | 3e |
| `domain_knowledge` | yes | Full ReAct | 3a |
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

## ps_diagnostics

**Answers:** Are propensity scores well-overlapped, balanced, and calibrated?

**Needs:**
- `dataframe_path` (string path to the parquet/csv the profiler wrote)
- `data_profile` (typed `DataProfile`; profiler must have run)
- `treatment_variable` (column name)
- `outcome_variable` (column name)

**Delivers:**
- `ps_diagnostics` (dict; model_quality, recommended_method, trimming bounds, warnings)
- `agent_briefs["ps_diagnostics"]` (typed `AgentBrief`, always written)

**Refuses when:**
- `dataframe_path` is None -> `NEEDS_NOT_MET` (reroute to profiler)
- `data_profile` is None -> `NEEDS_NOT_MET` (reroute to profiler)
- `treatment_variable` is unset -> `PRECONDITION_FAILED` (escalate; only the user/router can set it)
- `outcome_variable` is unset -> `PRECONDITION_FAILED`

**Flags raised:**
- `POOR_OVERLAP` -- common support below 90%
- `SMD_HIGH` -- max weighted standardised mean difference exceeds 0.1
- `CALIBRATION_OFF` -- mean absolute calibration error exceeds 0.1

**Success criteria** (id -> test in `tests/test_brief.py`):
- `ps.brief.always_written` -- execute always writes a brief into `state.agent_briefs["ps_diagnostics"]`
- `ps.refusal.needs_missing` -- `NEEDS_NOT_MET` on missing dataframe_path / data_profile
- `ps.refusal.precondition` -- `PRECONDITION_FAILED` on missing treatment/outcome
- `ps.flag.poor_overlap` -- `POOR_OVERLAP` when overlap_pct < 90
- `ps.flag.smd_high` -- `SMD_HIGH` when max weighted SMD > 0.1
- `ps.flag.calibration_off` -- `CALIBRATION_OFF` when calibration MAE > 0.1

## domain_knowledge

**Answers:** What causal-role hypotheses do the dataset metadata support?

**Needs:**
- `dataset_info` (already required on the state; at least one of its metadata fields must be populated)

**Delivers:**
- `domain_knowledge` (typed `DomainKnowledge`; hypotheses, uncertainties, temporal understanding, immutables)
- `agent_briefs["domain_knowledge"]` (typed `AgentBrief`, always written)

**Refuses when:**
- every metadata source on the state is empty (no description, no column descriptions, no tags, no kaggle_domain, no `raw_metadata`) -> `NEEDS_NOT_MET` (reroute to the metadata-fetch stage)

**Flags raised:**
- `WEAK_CONFOUNDER_EVIDENCE` -- no confounder hypothesis formed at medium-or-better confidence (downstream specialists must lean on data-driven discovery)

**Success criteria** (id -> test in `tests/test_brief.py`):
- `dk.brief.always_written` -- execute always writes a brief into `state.agent_briefs["domain_knowledge"]`
- `dk.refusal.no_metadata` -- `NEEDS_NOT_MET` when every metadata source is empty
- `dk.flag.weak_confounders` -- `WEAK_CONFOUNDER_EVIDENCE` when no confounder hypothesis
- `dk.status.failed_when_incomplete` -- `status="failed"` when treatment or outcome hypothesis is missing
- `dk.status.done_when_complete` -- `status="done"` when at least one treatment and one outcome hypothesis exist at medium+ confidence

## eda_agent

**Answers:** What distributional, balance, correlation, and outlier issues does this dataset carry?

**Needs:**
- `dataframe_path` (string path to the parquet/csv the profiler wrote)
- `data_profile` (typed `DataProfile`; profiler must have run)

**Delivers:**
- `eda_result` (typed `EDAResult`; distribution stats, covariate balance, correlations, outliers, VIF, quality score)
- `agent_briefs["eda_agent"]` (typed `AgentBrief`, always written)

**Refuses when:**
- `dataframe_path` is None -> `NEEDS_NOT_MET` (reroute to profiler)
- `data_profile` is None -> `NEEDS_NOT_MET` (reroute to profiler)

Treatment / outcome unset is NOT a refusal: EDA can still report distributional and correlation findings without a primary pair declared; only the balance flag becomes inert.

**Flags raised:**
- `OUTCOME_SKEWED` -- the outcome variable's |skewness| exceeds 1.0
- `TC_IMBALANCE` -- the max covariate SMD exceeds 0.25 (severe-imbalance threshold; 0.1 is the noisier "imbalanced" floor used by check_balance but is too noisy for orchestrator-level routing)
- `SUSPECT_CORRELATION` -- at least one variable pair has |r| > 0.7 (filter already applied by the EDA tool)

**Success criteria** (id -> test in `tests/test_brief.py`):
- `eda.brief.always_written` -- execute always writes a brief into `state.agent_briefs["eda_agent"]`
- `eda.refusal.needs_missing` -- `NEEDS_NOT_MET` on missing dataframe_path / data_profile
- `eda.flag.outcome_skewed` -- `OUTCOME_SKEWED` when outcome |skewness| > 1.0
- `eda.flag.tc_imbalance` -- `TC_IMBALANCE` when max covariate SMD > 0.25
- `eda.flag.suspect_correlation` -- `SUSPECT_CORRELATION` when high_correlations is non-empty

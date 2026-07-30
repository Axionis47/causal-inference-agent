# Causal Studio MVP — architecture and tradeoffs

Date: 2026-07-29

## Product boundary

The studio answers one bounded question:

> Given an arbitrary uploaded Kaggle bundle, an uploader-confirmed causal context, and
> one of eight supported designs, produce a reproducible estimate, test its
> fragility, and publish it only under an executable policy.

It is not an autonomous data scientist. It cannot infer assignment mechanisms,
temporal order, valid instruments, or mediators from correlations. A person
must supply or confirm those facts.

## Separation of concerns

```text
Streamlit UI
  ├── context contract and approvals
  ├── EDA views
  └── method configuration and lane-specific assumption answers
        ↓
preparation_agent.py (bounded ReAct)
  ├── inventory and inspect tables/columns
  ├── investigate possible join keys
  ├── draft context and repair plan
  └── check eight-lane readiness
        ↓ human preparation approval
studio_prep.py
  ├── read upload
  ├── mechanical profile
  └── propose/apply reversible repairs
        ↓ human design review
studio_protocols.py
  ├── role ledger and proposed causal/design map
  ├── lane-specific pre-estimation checks
  ├── immutable contract hash and revision
  └── prespecified post-estimation check registry
        ↓ frozen contract
lanes.py
  └── one of eight estimators; numbers only
        ↓
checks.py
  └── protocol-selected deterministic post-estimation diagnostics
        ↓
studio_policy.py
  └── allow / review / block
        ↓
studio_workflow.py (LangGraph)
  └── checkpoint + human publication interrupt
        ↓
studio_export.py
  └── executable notebook + data + policy + trace bundle
```

The system prompt is not an authority in this MVP. There is no model in the
numerical path. The preparation model investigates and proposes through bounded
tools; policy is code, transformations are deterministic, causal estimates are
code, and human decisions are explicit graph state.

## Preparation ReAct agent

The agent receives a bundle, a vague description, and a causal question. Its
tool surface is intentionally asymmetric: it can gather evidence and write an
advisory plan, but it cannot mutate data or run an estimator.

| Tool | Authority |
|---|---|
| `list_tables` | Read table inventory, shapes, columns, and quality signals |
| `inspect_table` | Read schema, quality, and the deterministic repair catalogue |
| `inspect_column` | Read one column's profile; PII samples are redacted |
| `find_join_keys` | Compare possible keys without joining |
| `select_primary_table` | Propose the analysis table |
| `draft_context` | Draft semantic fields using exact existing columns |
| `propose_repair` | Add an allowlisted repair to the advisory plan |
| `check_lane_readiness` | Run structural contracts for all eight lanes |
| `add_human_question` | Escalate unresolved meaning, join, or design choices |
| `recommend_lane` | Recommend one lane without binding execution |
| `finalize_plan` | Seal the advisory plan for review |

When a model credential is unavailable or a model/tool call fails, a
deterministic fallback exercises the same contracts and records `mode =
deterministic`. This prevents model availability from becoming a hidden
pipeline dependency.

The preparation model provider is fixed to **Gemini on GCP Vertex AI** through
the Google Gen AI SDK's Vertex backend. Project, location, and model are read
from `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `PREPARATION_MODEL`.
Application Default Credentials provide identity. Gemini Developer API keys
are deliberately unsupported for this agent. Every tool-call trace includes
provider, project, location, model, prompt version, and prompt hash.

The active prompt lives at `prompts/preparation-agent/1.1.0.md`; `1.0.0` remains
immutable for comparison and replay. Each plan records its
prompt id, semantic version, SHA-256 hash, model, tool calls, arguments,
durations, outputs, and failures.

## Supported analysis designs

| Design | Estimand | Required human knowledge | Required diagnostic emphasis |
|---|---|---|---|
| Observational adjustment | ATE | defensible pre-treatment adjustment set | overlap, balance, specification spread |
| Propensity matching | ATT | confounders and treatment encoding | overlap, balance, subsample stability |
| Instrumental variables | LATE | a valid instrument and exclusion argument | first-stage strength; expert review |
| Survival | hazard ratio | duration, event, and treatment meaning | proportional-hazards stability |
| Difference in differences | ATT | treated group, intervention timing | pre-trend support or explicit untestability |
| Regression discontinuity | local effect | real assignment cutoff | placebo cutoffs |
| Mediation | indirect/direct effects | a defensible mediator | confounding and specification fragility |
| Interrupted time series | level change | intervention date | placebo dates |

The library signatures remain the capability contracts. The UI can recommend
or prefill arguments, but it never bypasses a lane's validation.

## Design contract and bounded agency

The model can propose a lane, roles, repairs, and unresolved questions. It
cannot freeze a design. After method fields are selected, the protocol runs
checks that do not depend on the treatment-effect estimate, renders a proposed
causal/design map, and asks two or three lane-specific mechanism questions. A
named reviewer confirms the role ledger and freezes an immutable contract.

Each contract records dataset/table fingerprint, cohort, estimand, method
arguments, role ledger, assumption answers, protocol version, content hash,
prepared-data version and fingerprint, revision, parent hash, and whether the
revision was created before or after a previous result. Revisions made after a result are labelled
`post_estimation_exploratory`; they cannot silently replace the earlier result.
Frozen revisions persist in local SQLite and are copied into the audit bundle.

“Best” means best-supported, not largest or most significant. The agent never
sees candidate treatment-effect results while proposing a primary design, and
the complete lane-specific post-estimation set runs without selecting a
favourable variant.

## Context gate

Analysis cannot start until these fields are present:

- causal question
- dataset description and source
- unit represented by one row
- treatment assignment mechanism
- treatment/outcome timing
- target population
- intended use
- outcome column

The file may suggest types and distributions. It cannot authoritatively supply
business meaning. The uploader confirms the contract.

## Repair agency

The preparation layer may only propose:

- normalized column names
- blank strings represented as missing
- infinity represented as missing
- exact duplicate removal
- reviewable parsing of numeric-looking strings

The raw file is preserved. Every accepted transformation creates an audit row.
The layer does not impute outcomes, delete outliers, choose favorable subsets,
or mutate the source upload.

Every approval also creates a content-addressed data version from the raw-data
fingerprint, ordered repair/cohort manifest, and exact prepared CSV fingerprint.
Changing either the cohort or repairs archives the visible run, clears the
active contract, and returns the UI to preparation → preflight → design freeze.
The old run remains immutable. A new run records its parent run ID and is
labelled exploratory when an earlier result was visible.
Minimal run lineage—run ID, parent ID, dataset/version ID, contract hash, and
status—is persisted in SQLite so that this post-result label survives an app
restart. No dataset rows are stored in the lineage table.

Immediately before dispatch, the workflow verifies the prepared file hash,
contract content hash, data-version binding, lane/role configuration, human
approval, and complete passing preflight registry. A mismatch blocks execution
before numerical code is called.

## The eight lanes are executors, not agent tools

The eight lane functions are a fixed internal executor registry selected by
validated application state. They are not exposed to the Vertex preparation
agent, and the model cannot invoke them while exploring repairs or candidate
roles. They could later be wrapped as MCP tools for another trusted client, but
the same execution guard must remain in front of them; direct estimator tools
must never bypass the data-version and design contracts.

## Publication policy and escalation

Policy is evaluated after estimation and sensitivity checks.

- **Block:** missing required context, no estimate, missing/unapproved design
  contract, or failed pre-estimation checks.
- **Review:** possible PII, unknown assignment, high-impact intended use, IV or
  mediation's untestable core assumption, review-bound preflight assumptions,
  a post-estimation contract revision, or any failed diagnostic.
- **Allow:** no rule above triggered.

A `review` decision pauses the LangGraph run. Streamlit presents the policy
findings and accepts an approve/reject decision plus reviewer note. A rejected
run has no report or download bundle. Human identity is a text field in this
MVP; production must bind it to authenticated identity and role.

## Why LangGraph, and why not everywhere

LangGraph is used for state checkpoints and the publication interrupt. Those
features would otherwise require custom pause/resume storage. Profiling,
repair, estimation, diagnostics, and policy stay as ordinary functions because
wrapping deterministic functions in agents would add failure modes without
adding judgment.

LangSmith is optional observability. When configured, the graph is traced. The
local event list remains the portable audit log and is included in the bundle.

## One-hour tradeoffs recorded

1. **Reuse the eight benchmarked estimator functions.** “From scratch” applies
   to the product workflow and UI, not to rewriting already-tested mathematics.
2. **ReAct is confined to preparation.** It investigates unknown tables and
   proposes a plan. Sensitivity checks remain a fixed method-specific protocol,
   so the model cannot choose reassuring checks or change the estimate.
3. **No MCP server yet.** The function boundaries are tool-ready, but adding a
   protocol process does not improve a local single-application MVP. Expose the
   same contracts over MCP when a second client needs them.
4. **No LLM dependency in execution.** The preparation model may draft context
   and method eligibility, but a person confirms them and an offline fallback
   preserves usability.
5. **Local storage only.** Raw/clean CSVs, context, repairs, and SQLite graph
   checkpoints live under the project. This is not suitable for regulated or
   multi-tenant production data without encryption, retention, authentication,
   and tenant isolation.
6. **Synchronous execution.** Streamlit waits for the estimator and checks.
   Large datasets need a job queue, resource quotas, cancellation, and progress
   streaming.
7. **Deterministic interactive EDA.** Schema, missingness, numeric and
   categorical distributions, relationships, time views, explicit cohort
   previews, and all eight lane-specific views run server-side. Charts can
   guide questions but cannot certify a causal assumption. Only explicit
   preparation actions invoke Vertex AI.
8. **Notebook portability is bundled.** The zip includes the cleaned data,
   minimal causal runtime, requirements, policy, report, and run state. The
   notebook recomputes and asserts the reported estimate.
9. **Multi-table joins are never automatic.** The agent can rank possible keys,
   but the row grain and join cardinality require explicit human approval and a
   later deterministic join-plan implementation.

## Monitoring and alerts

Every completed or blocked run records:

- preparation mode, prompt version/hash, table count, and tool-call count
- applied repairs and row-loss fraction
- remaining possible PII columns
- selected lane and treatment-assignment description
- failed and untestable diagnostics
- policy decision and analysis error

The UI also writes sanitized, append-only interaction events to
`studio_events.sqlite`. Events are chained by parent id and cover meaningful
state transitions such as EDA variable changes, cohort preview/commit,
preparation approval, lane selection, and publication review. Raw rows and
hover events are not stored. Events present when analysis starts are copied
into the portable run state.

Local alerts fire on row loss above 5%/20%, remaining possible PII, diagnostic
failure, or estimator failure. These records are shown in Streamlit and bundled
as `monitoring.json`. LangSmith tracing is additive observability, not the sole
audit store.

## Next increments, in order

1. Bind reviewers to authenticated identities and roles.
2. Add dataset size/resource limits and asynchronous jobs.
3. Add authenticated cohort-contract review and pre-analysis sensitivity to
   repair alternatives.
4. Add curated evaluation cases for context completeness, method eligibility,
   policy decisions, and report wording.
5. Add LangSmith evaluation sets for preparation-plan correctness, invented
   columns, wrong primary-table selection, missed escalation, and prompt-version
   regressions.
6. Expose the stable functions as MCP tools only when another agent/client needs
   to call them.

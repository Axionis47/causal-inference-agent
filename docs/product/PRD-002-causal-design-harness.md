# PRD-002 — Evidence-backed causal design compiler

Status: final for implementation  
Product stage: post-intake causal design  
Depends on: `SYSTEM-CONTRACT.md`; PRD-001 — Kaggle intake, semantic availability, and storage  
Unlocks: PRD-003 — data preparation and recoverable lineage

Shared identities, envelopes, context isolation, persistence, retries, approval binding,
observability, and operational events are governed by `SYSTEM-CONTRACT.md`.

## 1. Outcome

Given a valid PRD-001 handoff, a causal question, and exactly one selected CSV, this stage
produces exactly one of:

- an evidence-backed `CompiledDesign`, measured `DiagnosticReport`, passing `CapacityReport`,
  accessible `GraphViewSet`, exact-hash `DesignApproval`, and approved `DesignOutcome`;
- a bounded consolidated request for real-world context only the user can supply;
- `needs_data` when the selected CSV cannot support an otherwise valid design;
- `unsupported` when no registered method can answer the requested estimand; or
- `system_failure` when the product, registry, model response, storage, or observability fails.

The stage selects and proves a design. It never mutates data, estimates an effect, chooses a
method after observing an estimate, or lets a model-authored value become executable merely
because it is plausible.

## 2. Agency boundary

The product uses models where interpretation is required and deterministic code where
correctness can be compiled.

Models may:

- interpret bounded evidence about column meaning and measurement timing;
- propose causal relationships, explicit alternatives, assumptions, and risks;
- propose the assignment mechanism, estimand, comparator, unit, and method-specific facts;
- rank all methods that the compiler has already found structurally and empirically feasible;
- return `unknown`, `needs_context`, or `conflict`; and
- repair only the fields named by a safe validation response.

Models may not:

- decide whether a method is computable;
- invent columns, evidence IDs, graph nodes, diagnostics, registry entries, or method IDs;
- create executable preparation or estimator settings;
- run arbitrary code, SQL, estimators, or data mutations;
- inspect treatment-effect estimates before approval;
- approve a design; or
- classify or route their own failures.

The deterministic compiler owns evidence admissibility, fact resolution, role binding,
structural eligibility, diagnostic binding and execution, empirical eligibility, exact
contrast encoding, preparation policy, estimator parameters, capacity, and handoff.

## 3. Data boundary

V2 analyses exactly one admitted CSV. Other intake artifacts may provide evidence but cannot
supply rows or joined columns.

The selected bytes are immutable. The stage may read them through the shared frame store and may
compute bounded statistics. It may not delete rows, change cells, recode values, derive estimator
features, join tables, or persist a modified frame.

Every table-dependent result binds the selected table artifact and content hash. Changing the
table, parser profile, causal question, material fact, method pack, registry version, or approved
design starts a new revision.

## 4. Authoritative workflow

The graph executes these ordered responsibilities:

1. Validate the intake handoff and select one CSV.
2. Compile a bounded context manifest and a deterministic statistical profile.
3. Ask the intent task for the question framing and grain proposal.
4. Triage columns by compiler-known relevance and process semantic batches.
5. Compile the measurement map, collect role evidence, and synthesize causal alternatives.
6. Ask the proposal task for non-executable causal semantics and a complete method ranking.
7. Resolve evidence and user answers into one `DesignFactSet`.
8. Evaluate every method pack's structural requirements without method-specific orchestration.
9. Bind and run every required read-only diagnostic for surviving candidates.
10. Evaluate empirical support and select one feasible method deterministically, using the model
    ranking only when several feasible candidates remain.
11. Compile `CompiledDesign`, render every causal graph alternative, and prove delivery capacity.
12. Present one exact review bundle for human approval and open the PRD-003 handoff only after
    approval of that exact artifact hash and revision.

No standalone router agent exists. No model call produces the complete design.

## 5. Deterministic statistical inspection

Statistical behavior is measured before a design is approved. This is a harness-owned analysis
capability, not an unrestricted model tool.

The universal profile records bounded structural facts including:

- row and column counts;
- inferred data types;
- null count and rate;
- unique cardinality, constants, and all-null columns;
- duplicate-row count and unique-column candidates; and
- bounded numeric summaries and quantiles.

Method packs then bind registered diagnostics to compiled roles and facts. Examples include arm
support, assignment-unit uniqueness, rough overlap, cross-fitting feasibility, panel and cohort
support, pre/post-period placement, cutoff-side support, and sharp-assignment contradictions.

All diagnostic implementations are read-only, versioned, and deterministic. A diagnostic returns
typed inputs, columns read, row accounting, measured values, warnings, and a terminal status.
Missing bindings or inadequate support cannot be replaced by a pack default or model guess.

Design tasks have a zero callable-tool budget. Their bounded context already contains the
measurements and evidence they are allowed to interpret. Adding a model-facing tool requires a
funded handler, explicit per-task permission, deterministic receipts, denial tests, and a reason
the same result cannot be supplied safely in the context pack.

## 6. Evidence and executable facts

Every material proposal carries evidence IDs. Evidence is validated for existence, task scope,
source class, and relation to the claim.

`DesignFactSet` is the sole semantic input to method compilation. Each fact records its source
artifacts, evidence class, evidence relation, epistemic status, and whether it is executable.
An executable fact must have a value, non-model support, and no unresolved conflict.

The compiler never silently substitutes one estimand for another. ATT remains ATT, ITT remains
ITT, and unsupported estimands remain unsupported. Table grain and unit identity require measured
or user-supported evidence; a model label alone cannot determine independence or standard errors.

Each causal role has at most one compiled binding. Multi-column adjustment roles remain explicit
through deterministic suffixes downstream rather than being collapsed to one arbitrary column.

## 7. Model tasks and repair

The only design task kinds are:

- `intent`;
- `semantic_batch`;
- `role_evidence`;
- `causal_context`;
- `role_ledger`; and
- `method_design`, which emits `AgentDesignProposalV2`.

Every task has a forced schema, a fixed prompt version, a fixed token budget, zero callable tools,
and at most two targeted correction attempts after the initial response.

A correction includes only:

- the stable error code;
- the JSON path;
- a safe summary of the rejected value;
- the expected constraint;
- why it blocks progress;
- allowed actions;
- closed candidate values when applicable; and
- required input IDs.

Unknown fields, unsafe validation text, contradictory stopping states, missing requirements, and
unchanged issue fingerprints consume the bounded repair budget. Exhaustion becomes
`system_failure:agent_output_invalid`; it never masquerades as unsupported science.

## 8. Failure ownership and routing

Every `ValidationIssueV2` has one category and responsible actor:

| Category | Responsible actor | Meaning | Permitted next action |
|---|---|---|---|
| `model_fix` | model | The response violates a repairable output constraint | retry only the named fields within budget |
| `human_input` | user | A real-world fact may exist outside the CSV and evidence | ask one consolidated, decision-relevant question |
| `needs_data` | data owner | Required measured support is absent or insufficient | provide another eligible CSV or revise the data scope |
| `unsupported` | product | No registered method supports the requested design | add a reviewed method/diagnostic pack or refuse |
| `system_failure` | system | Code, registry, storage, model gateway, or observability failed | inspect and retry operationally |

Routing uses the typed category, never string matching on prose and never a model's preferred
recovery. A mixed issue set follows the most restrictive safe route. Identical issues are
deduplicated by stable fingerprint in the terminal outcome.

## 9. Human interaction

Only this stage may interrupt the user.

Clarification is used only when the missing fact is material to computability or causal meaning
and could reasonably be known by a person. The deterministic ask gate consolidates questions,
shows attempted evidence, uses closed choices when possible, and permits at most two rounds.

Approval is separate from clarification. The approval interrupt shows the selected method,
estimand, exact contrasts, diagnostic computability, capacity result, graph summary, assumptions,
identification risks, and sensitivities. Approval binds the `DesignReviewBundle` artifact ID,
content hash, design revision, and exact approved artifact references. A mismatch is
`stale_approval` and cannot resume the run.

## 10. Registry-driven method compilation

The shipped method packs are randomized experiment, observational AIPW,
difference-in-differences, and sharp regression discontinuity.

Each pack declares:

- compatible assignment mechanisms and supported estimands;
- required, optional, protected, and forbidden roles;
- structural predicates;
- required diagnostic recipes and their parameter sources;
- context and support requirements;
- preparation policy and estimator bindings;
- invalidation, sensitivity, and multiplicity policy;
- capacity derivation; and
- required visual evidence.

Startup verifies exact coverage between registry declarations and deterministic handlers. Runtime
orchestration does not branch on method IDs. Pack-specific behavior is selected by registered
predicates, diagnostic primitives, and estimator adapters.

For sharp RDD, the compiled structure includes the exact cutoff, assignment direction, treated
value, and comparator value. For DiD, it includes the approved adoption-time value or column and
the measured adoption profile. Downstream stages must use those values and may not infer a new
direction, cutoff, arm, or timing rule.

## 11. Exact contrasts and estimator boundary

Contrasts use one canonical reversible encoding shared by design and estimation. The compiler
derives contrast IDs from observed treatment levels and the approved comparator; estimators decode
that exact pair.

An approved contrast whose treated or comparator state is absent fails before fitting with a
typed not-estimable error. The estimator cannot choose the first two sorted arms, substitute a
different control, or commit a misleading partial primary result.

`CompiledDesign.preparation` contains only compiler-derived requirements: output grain, keys,
required roles, protected columns, permitted imputation targets, eligibility and unusable-row
rules, missingness indicators, method structure, deletion-impact dimensions, final diagnostics,
and estimator-input schema.

## 12. Capacity and reviewability

`CapacityReport` measures applicable dimensions from the compiled roles, statistical profile,
exact contrasts, and method pack. Dimensions are arms, contrasts, subgroups, cohorts, periods,
event times, cutoff sides, series, and evidence items.

Applicability is explicit: `applicable`, `not_applicable`, or `unknown`. Zero is a measured zero,
not a substitute for either of the latter states. Applicable unknown values block approval.

Every required visual-evidence ID must have at least one registered template whose panel, series,
label, annotation, and accessible-table bounds fit. Capacity failure is a product limitation, not
permission to omit evidence or invent a dynamic template.

## 13. Committed artifacts

The authoritative V2 surface is:

- `StatisticalProfile`;
- `AgentDesignProposal`;
- `DesignFactSet`;
- `DiagnosticPlan`;
- `DiagnosticReport`;
- `CompiledDesign`;
- `CausalGraphView` and `GraphViewSet`;
- `CapacityReport`;
- `DesignReviewBundle`;
- `DesignApproval`; and
- `DesignOutcome`.

`DesignOutcome.status` is exactly `approved`, `needs_context`, `needs_data`, `unsupported`,
`changes_requested`, `declined`, or `system_failure`. Only `approved` may carry the complete
cross-stage references.

The PRD-003 handoff contains exactly the selected table, compiled design, diagnostic report,
capacity report, review bundle, and approval. Receivers resolve every reference from storage and
fail closed on missing artifacts, changed hashes, wrong types, broken parents, non-passing
capacity, or approval mismatch.

There is no V1 executor, compatibility reader, artifact translator, alternate handoff, or
agent-authored executable design. An unfinished pre-cutover analysis restarts from its immutable
intake inputs and requires new approval.

## 14. Graph rendering

The renderer compiles a directed acyclic concept graph from the validated causal context,
measurement map, and role ledger. It renders the base graph plus every declared material
alternative. N alternatives therefore produce N+1 views.

Every view has an accessible summary and remains bound to the same compiled design. A graph
cannot substitute a new node, role, edge, or method fact.

## 15. Persistence, observability, and security

Graph checkpoints contain only allowlisted identifiers, hashes, statuses, counters, and pending
interrupt metadata. Dataframes, raw documents, model transcripts, and growing chat histories do
not enter checkpoint state.

Artifacts are immutable, content-addressed, parent-bound, and committed through the flush-gated
path. Replays with identical inputs and versions reproduce the same compiler-owned payloads.

Prompts and events contain bounded evidence, safe issue summaries, and artifact IDs. They do not
contain credentials, raw provider captures, unrestricted row samples, stack traces, or secrets.
An observability failure is a system failure and cannot be reclassified as a data or causal
limitation.

## 16. Acceptance criteria

The stage is complete only when all of the following hold:

1. Different valid single-CSV schemas can be profiled and compiled through role bindings rather
   than hard-coded column names.
2. Every material executable fact has admissible provenance.
3. Assignment-incompatible methods never reach model ranking.
4. Every required diagnostic has exact typed bindings and terminal measured output.
5. Missing context, insufficient data, unsupported capability, invalid model output, and system
   failure remain distinct through the CLI.
6. RDD and DiD structures reach their estimators unchanged.
7. Exact contrasts cannot be silently changed or partially estimated.
8. Capacity uses measured applicable dimensions and blocks unreviewable designs.
9. The human approves a legible, exact-hash review bundle.
10. PRD-003 receives exactly the six approved V2 handoff entries.
11. All four shipped method journeys, ambiguity and failure routes, mutation tests, schema tests,
    lint, strict typing, budget checks, and the full repository suite pass.
12. Superseded executable contracts, prompts, registry rows, migrations, readers, adapters, and
    fixtures have no remaining references outside append-only historical records.

## 17. Complexity limits

The approved ceilings for this cutover are:

- design production: 4,300 significant lines;
- total production: 16,500 significant lines;
- tests and evaluation tooling: 13,500 significant lines;
- declarative assets: 4,900 significant lines;
- repository total: 35,000 significant lines;
- production modules: 86;
- largest production module: 350 significant lines; and
- largest function: 75 significant lines.

Crossing a ceiling requires a new explicit user decision. Near-cap warnings are expected and do
not authorize compatibility layers or duplicate authorities.

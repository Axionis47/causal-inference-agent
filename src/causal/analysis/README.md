# Analysis capability library

This package exposes a navigable graph of supported scientific decisions, evaluates
partial candidates, accepts a complete fixed design, and executes an immutable
approved plan. It does not run an agent,
ask human questions, infer missing scientific facts, or select a different analysis
after inspecting effects. The caller owns scientific approval and human interaction.

## Public interface

Import operations from `causal.analysis.interface` and artifacts from
`causal.analysis.contracts`.

| Operation | Result |
| --- | --- |
| `list_methods()` | Four method summaries with capability versions and graph entry identifiers |
| `explore_capabilities(draft=None, at=None, relations=None, limit=30, cursor=None)` | A bounded graph neighborhood, candidate-conditioned choices, global blockers and continuation information |
| `evaluate_candidate(draft)` | Pure, deterministic `CandidateEvaluation` with `design_ready`, `needs_information` or `rejected` status |
| `retrieve_guidance(method, topic, draft=None, diagnostic_id=None)` | Targeted reasoning, generated input schema, applicable choices, exclusions, missing prerequisites and follow-up topics |
| `assess_specification(fixed_design, draft)` | Compare the complete accepted candidate with the proposed executable settings; expose a specification only when `ready` |
| `identify_data(data, name)` | Content identity for a Polars frame |
| `preflight(specification, data)` | Observable prerequisites and readiness to attempt estimation |
| `compile_plan(specification, preflight)` | Immutable, fully resolved `CompiledPlan` |
| `execute(approved_plan, data)` | Versioned `AnalysisEvidence`, including explicit failures |

Start at `analysis` before nominating a method. Method entries are `method:rdd`,
`method:randomized`, `method:aipw` and `method:did`. Stable identifiers then address
specific context, such as `rdd:decision:cutoff`, `rdd:role:running_variable`,
`rdd:requirement:sharp_assignment` and `rdd:sensitivity:half_bandwidth`. Links use
`offers`, `requires`, `reveals`, `excludes` and `checked_by`. Browsing is read only:
inspecting another method does not nominate it or alter a candidate.

A `CandidateDraft` can omit the method, population, outcome, bindings and scientific
facts. Resolve its independent requirements in any order and reevaluate after each
material change. `available` describes a local supported choice; it does not imply
that the whole candidate is ready. Missing prerequisites make dependent options
`conditional`, and contradicted prerequisites make them `blocked`. A revision can
reopen a branch. Invalidated selections remain visible rather than disappearing.

Every exploration includes whole-candidate status and blocker references even when
the returned neighborhood is small. `coverage` reports the total and returned
neighbors, offset, truncation and a continuation cursor. Follow that cursor with
the same candidate and query. Candidate, graph or query changes invalidate it.
Unknown nodes, relations and invalid requests raise `CapabilityRequestError` with
a typed request target. An absent node in a truncated response is not an exclusion.

Retrieval topics remain `overview`, `configuration`, `diagnostics` and
`diagnostic_details`; the last requires a diagnostic ID. These are compatibility
views over the graph and its evaluation, with the generated method input schema,
role slots, explanations and current option states. They have no independent
scientific acceptance rules. Inspect scientific assertions, fixed policies and
mechanical defaults separately through the evaluation's selection origins.
Fixed execution settings also have retrievable policy nodes, such as
`aipw:policy:nuisance.regularization_c`. These expose the exact method-owned values
used by execution; they are context, not additional configurable options.

The current typed configuration schemas are owned by
[randomized](methods/randomized/specification.py),
[AIPW](methods/aipw/specification.py), [DiD](methods/did/specification.py), and
[sharp RDD](methods/rdd/specification.py). Inspect the retrieved schema for the
exact supported variations. A configuration accepts no arbitrary extra parameters.
Unimplemented subgroup branches, E-values, general attrition bounds, arbitrary
placebo/donut settings, misleading CR2 inference and unsupported dependence
structures are excluded from this public boundary.

## A complete journey without an agent

```python
import polars as pl
from causal.analysis import interface as analysis
from causal.analysis.contracts import (
    ApprovedPlan, CandidateAssertion, CandidateDraft, CandidateOutcome, FixedCandidate,
)

# The notebook/caller supplies these study facts and actual column meanings.
data = pl.DataFrame({
    "person": list(range(40)),
    "assigned": [str(i % 2) for i in range(40)],
    "score": [2.0 + 3.0 * (i % 2) + 0.1 * (i // 2) for i in range(40)],
})
root = analysis.explore_capabilities()
proposal = CandidateDraft(
    method="randomized",
    population="All randomized participants in the prespecified study population",
    outcome=CandidateOutcome(
        column="score", kind="continuous", units="score points",
        meaning="The protocol's post-assignment score measurement",
    ),
    unit_grain="One row per randomized participant",
    missingness_policy="Require observed outcome and assignment for the declared population",
    facts=(CandidateAssertion(
        name="assignment_mechanism", value="individual_randomized",
        support="fact", evidence=("study-protocol:assignment-section",),
        original_answer="Each participant was individually assigned at random.",
    ),),
    configuration={
        "method": "randomized", "treatment_column": "assigned",
        "unit_column": "person", "treated_value": "1", "comparator_value": "0",
        "estimator": "difference_in_means",
    },
    seed=17,
    candidate_reference="study-design:revision-1",
    source_dataset_reference="source-study:revision-1",
)
evaluation = analysis.evaluate_candidate(proposal)
assert evaluation.status == "design_ready", evaluation.issues
menu = analysis.explore_capabilities(proposal, at="method:randomized")

# The caller has completed scientific review. This helper hashes the complete
# candidate under its real reference; constructing a snapshot grants no approval.
fixed = FixedCandidate.from_candidate(proposal, reference="accepted-study-design:revision-1")
assessment = analysis.assess_specification(fixed, {
    "dataset": analysis.identify_data(data, "prepared-trial:revision-1"),
})
assert assessment.status == "ready", assessment.issues
specification = assessment.specification
assert specification is not None
readiness = analysis.preflight(specification, data)
assert readiness.ready, readiness.issues
plan = analysis.compile_plan(specification, readiness)

# Obtain real approval of this exact plan before constructing this attestation.
approved = ApprovedPlan(
    plan=plan, approved_hash=plan.plan_hash,
    approved_by="study analyst", approval_reference="review-record:revision-1",
)
evidence = analysis.execute(approved, data)
assert evidence.primary.status == "computed"
assert abs(evidence.primary.estimates[0].estimate - 3.0) < 1e-10
```

The complete snapshot includes the scientific frame, treatment/comparator states,
estimand, unit grain, adjustment bindings, assertions and their evidence/scope,
configuration, selected diagnostics/sensitivities, seed and caller restrictions on
population or missingness. Optional `VariableBinding` entries connect graph roles
to notebook columns and source references; an expected prepared alias must be
declared before acceptance. The source snapshot reference stays distinct from the
exact prepared-frame identity bound after preparation.

A changed scientific selection or setting requires a new candidate revision and
accepted snapshot. Prepared-data binding can change without changing the accepted
candidate. Supplied executable settings are mechanically checked against the fixed
snapshot, so an alternative treatment state, adjustment variable or sensitivity
cannot be smuggled through an unchanged accepted reference. `FixedDesign` and
historical specifications/plans remain readable but cannot authorize a new run.

## Feedback and scientific context

An issue names its `category`, affected `field`, `finding`, `requirement`,
`explanation` and possible `resolutions`. The categories are `missing_context`,
`missing_data`, `unsupported_capability`, `contradictory_configuration` and
`incompatible_data`. The caller decides whether to ask a human, inspect a source,
repair data or propose a supported configuration. Missing context is an expected
assessment result, not a model-formatting failure.

A `CandidateAssertion` preserves the original answer independently of its normalized
value, supporting references, support kind and scope. An unknown value or missing evidence stays
unresolved. The library does not turn ambiguous prose into a fact. Fact references
record the caller's evidence; this package does not independently authenticate
sources or prove identification assumptions. An assertion declared as an assumption
can satisfy only a requirement that explicitly permits one. In particular, RDD
assignment rules require facts; a selected adjustment requires timing evidence
scoped to the selected column(s). False, unknown and absent remain distinct.

Requirement results include typed `resolution_targets`: `study_fact`,
`scientific_assumption`, `scientific_frame`, `configuration`, `role_binding`,
`data_preparation`, `execution` and `request`. Callers route these to their notebook,
design reasoning, repair or operational component. Analysis does not select the
responsible human or component, fetch sources, repair a frame or reinterpret facts.

Diagnostic obligation (`required`/`optional`) is separate from applicability
(`applicable`/`inapplicable`/`unresolved`). The catalog separately records when applicability resolves (design) and when the
measurement runs (execution). Data preflight has its own observable obligations. Required applicable checks and selected optional checks enter
the plan. Unresolved required or selected prerequisites block compilation.
Inapplicability and optional non-selection remain visible in evidence. Diagnostics
cannot be removed from an approved plan by editing its serialized contents.

## Identity, execution and reporting

Data identity hashes the complete frame, including types, row order, nulls and
unused columns. The plan freezes the resolved specification, diagnostics,
sensitivities, seed, capability version and preflight identity. New specification v2 and plan v3 records bind the complete fixed candidate; plans
contain no plot prescriptions. Historical specification v1 and plan v1/v2 records
remain readable, but execution requires a complete current candidate, recompilation
and fresh approval against the current implementation. Its
implementation hash also binds the executable method sources, shared calculation
code, retained numerical profile and installed numerical package versions. A
change invalidates earlier readiness and approval. Technical retry means executing
the same approved artifact with the same data and seed; this library has no
adaptive estimator-selection or correction loop.

Preflight does not fit effects. It checks such things as required columns, finite
values, assignment consistency, independent-unit structure, adoption schedules,
period support and cutoff support. A pass means ready to attempt estimation;
convergence, selected local bandwidth support, model diagnostics and scientific
interpretation can still fail or require qualifications.

Evidence separates primary estimates, diagnostics, sensitivities and raw scientific
`supporting_data`. Every planned numerical computation has an explicit status.
Failures preserve a successful primary fit. Each sensitivity retains its own
estimate and interval. Supporting calculation failures and unavailable values are
reported independently; no visual coordinates or plot choices are produced.
The overall status is `completed`, `completed_with_limitations`, `incomplete` or
`failed`; completion is not a declaration of causal validity. Post-analysis owns
interpretation, visual decisions and report composition. It must disclose failed
checks and cannot select a favorable sensitivity as a replacement primary result.

## Code ownership and migration

`methods/<family>/specification.py` owns typed choices, shared requirement
predicates, role definitions and data checks; `diagnostic_catalog.py` owns
obligations, dependency declarations and registered settings; `diagnostics.py` owns numerical measurements; `estimation.py` owns
estimator calls. Guidance and numerical tests are colocated with their method.
`common` contains shared boundary and numerical functions, not an agent workflow.

`integration` retains the old persistence/reporting contracts and pipeline adapter.
Its version-one resources describe historical estimation artifacts and are not
retrieval options for the new API. The in-memory numerical bridge reuses verified
calculation implementations without creating database records or implying legacy
approvals. Historical schema names and stored artifacts remain unchanged. The
existing agent has not been redesigned or silently switched to this new protocol.

Run `uv run pytest src/causal/analysis` for library and legacy integration tests;
the shared infrastructure fixtures may require Docker. Run `uv run pytest` for
repository regression checks, `uv run ruff check src tests tools conftest.py` and
`uv run mypy src/causal` for lint/source typing. Colocated tests use local support
modules and are discovered explicitly; they are excluded from production typing
and counted as tests in the unchanged complexity budgets. Runtime guidance counts
as declarative content. Existing budget debt remains a failing gate, not a release
pass.

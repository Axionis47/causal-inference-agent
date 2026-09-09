# T-038 — Isolate the runnable analysis module

Status: implementation in progress.

The user authorized the analysis refactor first on 2026-09-09. Keep the four
analysis families fixed, make runnable specifications discoverable upstream,
and put analysis code, diagnostics policy, resources, and tests under one owner.

## Boundary

- Move `src/causal/estimation` to `src/causal/analysis`; preserve persisted
  estimation stage names, artifact schemas, and database identities.
- Keep the four numerical adapters and their fit-dependent measurements in
  this package. Extract common diagnostic execution, applicability, and claim
  ceilings from the overloaded engine into `diagnostics.py`.
- Put estimation registry, validation registry, and claim-review prompt in
  `analysis/resources`. Keep global artifact and cross-stage registries global.
- Move the estimation tests to `analysis/tests` and extract reusable builders
  into test support instead of importing setup from collected test modules.
- Expose a small pure capability API. Design and execution consult the same
  supported methods, estimands, profiles, roles, and sensitivity definitions.
- A supported specification is not a guarantee of an estimable or causally
  valid result. Data-dependent support checks and scientific diagnostics still
  run and can block, invalidate, or qualify the result.
- Remove advertised branches that silently ignore their requested variation
  from the runnable default catalog. Existing requests for them must be rejected.
- Keep mandatory diagnostics and planned conditional checks deterministic.
  This task adds no model that selects diagnostics after seeing effect results.

## Ownership and scope

Analysis owns post-fit diagnostics. Intake profiling and design/preparation
feasibility checks stay in their stages. Method-dependent measurements remain
close to the fit that supplies them; common reporting and policy have one owner.
The claim-review model and persistence coordinator remain separate from the
numerical adapters; importing capability checks must not start either.

Planned production edits: all former estimation modules; new capabilities and
diagnostics modules; design compiler and review-policy import/resource seams;
runtime composition/dispatch imports; presentation contract imports; budget
path ownership. Planned test edits: relocate estimation tests, shared support,
capability-boundary tests, dependent import/resource paths, discovery settings.
Relocations delete their old paths; no second estimator implementation or
permanent compatibility package is created.

Budget baseline: `/tmp/causal-t038-budget-before.json`, captured before code
changes. Existing budget excess is not repaired by hiding relocated tests or
resources under a production path. Update accounting for the new ownership
before measuring the result. Target incremental growth is bounded to the
capability checks and focused boundary tests (approximately 400 production and
250 test lines before deletion/deduplication), not a new workflow framework.
The broader pre-existing budget debt remains visible and is not a reason to
abandon the user-authorized extraction or raise limits silently.

## Acceptance

1. All four estimator adapters and their numerical regression tests remain
   available from the analysis package; old estimation source paths disappear.
2. Unknown/unsupported specifications are rejected before fitting and design
   eligibility receives structured analysis capability blockers.
3. Diagnostic execution and applicability have a clear common entry point;
   prescribed sensitivity execution preserves results and explicit failures.
4. Run local analysis tests without a model or external account; integration
   tests use the existing local container fixtures and remain separately marked.
5. Run affected design/runtime/presentation checks, lint, strict source typing,
   and the complexity report. Record failures and remaining scope explicitly.
6. Document runnable capabilities and implementation limitations beside code.

No live model runs, new scientific estimators, data migration, or deployment
are part of this first extraction.

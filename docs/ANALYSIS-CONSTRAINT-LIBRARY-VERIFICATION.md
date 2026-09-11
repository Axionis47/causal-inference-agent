# Analysis capability graph verification

Implementation date: 2026-09-10/11. Scope: `src/causal/analysis` and the analysis
integration documentation. Existing unrelated working-tree changes were preserved.

## Baseline established before cutover

| Check | Observed result |
| --- | --- |
| Analysis, method and retained integration tests | 356 passed, 69 warnings in 40.46 seconds |
| Repository tests plus post-analysis tests | 1456 passed, 3 skipped, 3 warnings in 188.32 seconds |
| Analysis lint | Passed |
| Analysis strict typing | Passed; 43 source files |

The regression logs are `/tmp/causal-analysis-baseline.log`,
`/tmp/causal-repository-baseline.log`, `/tmp/causal-ruff-baseline.log` and
`/tmp/causal-mypy-baseline.log`. The full repository lint/typing snapshots were
taken after concurrent implementation began; their findings were confined to the
in-progress analysis files, so they are not described as pre-change baselines.

Read-only RDD probes are recorded in `/tmp/causal-guidance-baseline.json`. The
configuration list contained the same 16 entries for an empty proposal, running
variable only, running variable plus cutoff, a supplied covariate, and contradicted
sharp assignment. Adding a cutoff exposed four sensitivity comparisons despite
unresolved scientific context; adding a covariate exposed a fifth. Eight diagnostic
entries and four sensitivities could be locally applicable with
`sharp_assignment=False`. These were guidance defects, not numerical computations.

## Contract and acceptance changes

The public graph, partial candidate and evaluation are versioned contracts. All
four method definitions provide requirements, role slots and option/check
dependencies to the same evaluator. The tests exercise root discovery, method
navigation, real explanation targets, whole-candidate blocker visibility,
candidate-conditioned options, source support and scope, value origins,
deterministic evaluation, bounded traversal and invalid cursor handling.

`FixedCandidate` (`analysis-fixed-design.v2`) hashes the complete accepted candidate
and its real caller reference. New specifications are `analysis-specification.v2`
and new plans are `analysis-plan.v3`. Historical readers remain available, but the
narrow `FixedDesign` no longer certifies an executable candidate. Regression
fixtures now declare their synthetic scientific context, obtain a complete fixed
snapshot and bind the prepared frame before preflight. Invalid and incomplete
proposal tests use candidate evaluation before freezing. Numerical assertions were
retained.

New scoped timing requirements make an RDD predetermined covariate, randomized
baseline covariate, or AIPW adjustment set conditional until the supplied timing
assertion covers the selected column(s). Factual requirements cannot be replaced
by assumptions. DiD's explicitly declared identification assumptions remain
supported. Diagnostic applicability is resolved during design; its measurement
runs during execution.

Previously private execution constants now belong to method-owned fixed policy
definitions, with retrievable policy nodes and fixed-policy origins. The exact
dispatch settings were captured before that move in
`src/causal/analysis/tests/support/fixed_execution_settings.json`; regression tests
compare the full settings for all four methods and the AIPW nuisance/randomness
profile. The numerical values and supported menu were not expanded.

## Scenario coverage

| Requirement | Verification |
| --- | --- |
| Discover before method or columns exist | Empty `CandidateDraft`, root methods, unresolved method target |
| Every method shares the contract | Parameterized root/method/requirement/option navigation and reachability |
| Definitions are internally valid | Dangling dependency, duplicate ID and prerequisite-cycle rejection; all graph edges resolve |
| Execution policies are inspectable and exact | Graph node values and selection origins equal method policy declarations; complete dispatch settings equal the captured numerical baseline |
| Local eligibility differs from global readiness | Focused RDD sensitivity keeps global blocker count/references and rejection |
| Unknown, false and true differ | Three sharp-assignment states produce conditional, blocked and available branches |
| Revision and backtracking | Corrected assignment reopens branches; removal of a selected covariate and changed cutoff invalidate comparisons |
| Scope follows selected variables | Missing or wrong-column timing evidence does not satisfy adjustment prerequisites |
| Randomized choices | Unadjusted versus ANCOVA; conditional baseline check and timing evidence |
| AIPW choices and scale | Missing estimand; 1, 17 and 101 arbitrary Unicode-named covariates bound to one stable graph role |
| DiD choices | Simultaneous versus staggered adoption and conflicting/common/column schedules |
| Bounded graph requests | Pagination coverage, disjoint pages, stale candidate/graph cursors, typed unknown-node/relation/request errors |
| Deterministic purity | Exact repeated evaluations, order-independent mappings, changed revision fingerprint, preserved caller input |
| No effect fitting while designing | Fresh subprocess checks imports; only explicit preflight/execution obligations returned |
| Extensibility | A synthetic fifth capability uses the common navigation/evaluation path without a caller method branch |
| Complete fixed boundary | Dedicated fixed-candidate tests cover tampering, changed settings, historical readers and prepared-data binding |
| Repair does not hide deeper data failures | Restoring a missing treatment column exposes repeated-unit failure with data-preparation targets and the same fixed design |
| Numerical behavior and exact approvals | Existing method/public-boundary execution, data identity, implementation identity and tampering regression tests |

The documented example in `src/causal/analysis/README.md` was executed successfully
from exploration through fixed acceptance, preflight, compilation, exact approval
and the original expected estimate of 3.0. Its output is recorded in
`/tmp/causal-readme-example.log`.

Current read-only probes are in `/tmp/causal-guidance-current.json`: missing RDD
assignment facts keep sensitivities conditional, `sharp_assignment=False` blocks
all sensitivity branches and rejects the candidate, and a complete supported
candidate is design-ready with its four unadjusted comparisons available. The
unbound covariate comparison remains inapplicable and inspectable.

## Final checks

| Check | Observed result |
| --- | --- |
| `uv run pytest -q` | **1924 passed, 3 skipped, 71 warnings in 248.85 seconds**; includes all analysis, numerical, retained integration, repository and post-analysis tests |
| Final evaluator/graph regression | **72 passed in 14.42 seconds** after the final linear-time role-binding lookup and absent-selection projection changes |
| `uv run ruff check src tests tools conftest.py` | **Passed** on final source |
| `uv run mypy --strict src/causal` | **Passed**; 132 source files on final source |
| README executable example | **Passed** from exploration to the approved numerical result |

Final logs: `/tmp/causal-full-final.log`,
`/tmp/causal-final-evaluator-regression.log`, `/tmp/causal-lint-final.log`,
`/tmp/causal-typing-final.log` and `/tmp/causal-readme-example.log`.

The intermediate analysis run had 422 passes and two obsolete test expectations
for the former plan version and historical-plan refusal text. Both expectations
were updated for the versioned boundary; refusal and numerical assertions were
preserved. They pass in the final run. The three skips and warning categories were
present in the baseline. No test failures, lint failures or strict typing failures
remain.

Discovery/design checks do not authenticate source evidence, prove causal
assumptions, or make a prepared frame runnable. Those remain the documented caller,
scientific-review and data-preflight responsibilities. The dataset notebook,
replacement design workflow, human-question interface and repair agent remain
outside this analysis-library implementation, as specified by the plan.

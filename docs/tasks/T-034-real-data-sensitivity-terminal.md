# T-034 — Real-data sensitivity terminal correction

Status: accepted
Owning PRD: PRD-004 §15, §18, §26
Depends on: T-024, T-028, T-033

## 1. Finding

An actual simultaneous-adoption DiD run computes its primary estimate and every diagnostic, but
the `not_yet_treated_comparison` sensitivity has no eligible later-adopting cohort. The engine
correctly commits a visible typed `failed` sensitivity result, as PRD-004 §15 requires. Estimation
wall 10 then incorrectly treats that terminal failure as if the branch were absent and stops the
run with `sensitivity_not_terminal`.

The actual Groupon AIPW preparation stop was not a product defect: the run fixture named the
approved key `unit_id` while the source column remained `deal_id`. The rerun must align that name
without changing source values.

## 2. Deliverables

1. Make estimation wall 10 require one visible result for every planned sensitivity branch while
   accepting every typed `ExecutionStatus`, including `failed`, as terminal.
2. Keep the failed branch's warning, qualification, null result, and audit artifact unchanged.
3. Add a wall regression proving that a present failed sensitivity is terminal; an omitted branch
   must continue to fail closed.
4. Rerun the actual Groupon AIPW and state minimum-wage DiD inputs through preparation,
   estimation, and presentation.

## 3. Constraints and verification

No estimator, diagnostic, sensitivity calculation, or registry changes. Production code net
non-positive where practical; tests no more than +30 logical lines; no new production module.
Run the focused estimation tests, budget checker, ruff, mypy, and the full suite.

## Amendment 1 — composed-runtime fixture

Once wall 10 admitted the visible failed branch, two small-fixture runtime tests reached the claim
receiver for the first time. Their design-only scripted gateway cannot answer claim-review or
curator tasks and the boundary correctly converts its `KeyError` to `internal_error`. Give this
shared test fixture the same task-aware claim and curator delegates already used by the composed
close test. The five-row fixture must then reach its pre-existing diagnostic ceiling and return a
typed `invalidated` claim, with the failed sensitivity visibly preserved; realistic inputs may
continue to qualified delivery.

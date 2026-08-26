# T-028 — DiD and sharp-RDD adapters

Status: frozen for implementation
Owning PRD: PRD-004 §11, §12, §26
Depends on: T-026

## 1. Deliverables

1. `src/causal/estimation/did.py` (≤ 300 logical) — `EstimatorAdapter`, two
   registered profiles selected by the frozen plan (never by results):
   - simultaneous: pyfixest `feols` with approved group/post/fixed-effects/cluster
     specification.
   - staggered: pyfixest Sun–Abraham (`sunab`) cohort-relative event study with the
     approved comparison cohort (never/not-yet-treated), reference period, event
     window, and registered aggregation to the one primary item. A staggered
     profile requiring a panel refuses repeated cross-section (`not_estimable`).
   - event-cell contribution masks (§11.2): cells identified, never deleted;
     unsupported approved cells → `not_estimable`/`design_conflict`, never a
     different comparison.
   - §11.3 diagnostics harvested from the fit + one polars pivot: schema
     reconciliation, group-time/cohort support, pre/post placement, lead estimates
     + joint prespecified pre-period Wald test (from the sunab fit), composition
     change, cluster adequacy, aggregate-weight sensitivity, reference-period
     integrity; PRD-002 concurrent-event qualifications carried through.
   - §11.4 sensitivities via the engine loop: anticipation window, comparison-cohort
     alternative (when both approved), alternative aggregation, balanced-panel
     mask, leave-one-cohort-out helper.
   - figure payloads: group-time means, event-time estimates + intervals,
     support/composition counts, primary aggregate (§17 row 3).
2. `src/causal/estimation/rdd.py` (≤ 240) — `EstimatorAdapter`:
   - versioned `rdrobust` call with exact approved cutoff/direction, local linear
     default, approved kernel (triangular default), registered bandwidth rule,
     robust bias-corrected inference (pack names which quantity is primary);
     covariates/cluster per plan. One primary item.
   - bandwidth contribution mask generated only by the approved rule; assignment
     contradiction → invalidation, never row removal (§12.2).
   - §12.3 diagnostics: harvest of rdrobust output (bandwidths, effective N,
     mass-point handling, BC integrity), `rddensity` manipulation test, covariate
     continuity (loop re-calling rdrobust per approved covariate), support/heaping
     counts; §12.4 sensitivities via the engine loop (bandwidth multiples,
     polynomial order, kernel, donut, placebo cutoffs — all only when approved).
   - figure payloads: fixed binned outcome summaries (rdrobust/rdplot-quantities
     computed at fixed registered binning, no result-dependent bins), fitted-curve
     points, cutoff/bandwidth metadata, density summary, primary item (§17 row 4).

## 2. Tests (≤ 400 logical)

DiD: simultaneous parity vs hand-computed 2×2 fixture; staggered fixture with known
cohort effects → sunab aggregate within tolerance and TWFE-distinctness pinned
(assert the unqualified TWFE coefficient is NOT the committed primary on a
heterogeneous fixture); lead/pre-trend test surfaces; unsupported-cell refusal;
repeated-cross-section refusal for panel profile. RDD: rdrobust parity on the
package's canonical fixture (pinned estimate + robust CI to tolerance); rddensity
runs and maps to the registered diagnostic; cutoff immutability (plan cutoff ≠ data
tampering); donut/bandwidth sensitivities only when approved (unapproved branch id
refused). Both: e2e through `run_estimation` to `complete` with handoff accepted
(reusing the T-026 fixture machinery with did/rdd packs).

## 3. Constraints

estimation ≤ 3,130 total after (+540); tests ≤ 400; modules +2 (12 estimation
modules total, 78/86). Never reimplement estimator math the libraries provide. Wave
rethink shared. Committed code wins; record deviations.

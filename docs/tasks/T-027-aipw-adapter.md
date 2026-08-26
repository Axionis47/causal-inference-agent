# T-027 — AIPW adapter: cross-fitting, nuisance profiles, score, overlap diagnostics

Status: frozen for implementation
Owning PRD: PRD-004 §10, §26
Depends on: T-026

## 1. Deliverables

`src/causal/estimation/aipw.py` (≤ 310 logical) — the `EstimatorAdapter` for the
observational pack:

1. `CrossFitAssignmentV1` built deterministically from the plan seed + frozen
   source-row ids (treatment-stratified K folds; mapping as restricted object;
   counts by fold×treatment recorded).
2. Fold loop (§10.2, the leakage contract): per fold — fit the PRD-003
   `cross_fit_training_fold` preprocessing recipe on training rows only; transform
   both with training-fitted params; fit registered propensity + outcome learners
   (sklearn, from the pack's nuisance profile: primary = fixed regularized GLM);
   predict held-out rows; commit bounded fold diagnostics + restricted prediction
   object. No validation-row outcome ever trains its own prediction.
3. The AIPW score (~30 lines numpy, the ONE hand-written formula): ATE/ATT per the
   approved estimand from out-of-fold predictions with influence-function variance;
   versioned numerical propensity bound (division-by-zero guard only, reported
   separately from trimming, §10.4). One `PrimaryContrastResultV1`.
4. §10.5 diagnostics harvested from in-memory arrays: fold convergence/support,
   calibration + registered performance (sklearn metrics on held predictions),
   propensity distribution + common support, weight tails + ESS, balance before/
   after implied weighting (shared engine helper, weighted), influence
   distribution + influential-unit warnings, missingness-indicator usage;
   unmeasured-confounding qualification carried from the approved design.
5. §10.6 sensitivities via the engine loop: alternative registered profile
   (histogram-GB), alternative seed/fold count, propensity-bound sensitivity.
6. Figure payloads: overlap bins, weight summaries, balance measures, primary item
   + interval (§17 row 2).

## 2. Tests (≤ 300 logical)

Score parity: synthetic fixture with hand-computed AIPW ATE + IF-variance pinned to
tolerance (both ATE and ATT). Leakage trap: a poisoned recipe/learner that peeks at
validation outcomes must change held-out predictions and be caught by wall 6's
receipts (assert the honest path passes, the poisoned path fails). Fold
reproducibility: same seed → identical assignment artifact hash; different seed →
different folds, same-tolerance estimate on a null-effect fixture. Overlap fixture
with mass near 0/1 → warning/invalidation per pack rule. Predictions/weights never
appear in any committed envelope payload or event (scan test, the fitted-params
idiom from D-074).

## 3. Constraints

estimation ≤ 2,590 total after (+310); tests ≤ 300; modules +1. sklearn objects
never serialized (no pickle; predictions stored as typed arrays in restricted
objects). Wave rethink shared. Committed code wins; record deviations.

# T-026 — RCT adapter + coordinator nodes: first end-to-end `complete` bundle

Status: frozen for implementation
Owning PRD: PRD-004 §7, §9, §17, §19 (per §26.1), §22
Depends on: T-023, T-024, T-025

## 1. Deliverables

1. `src/causal/estimation/rct.py` (≤ 260 logical) — the `EstimatorAdapter` for the
   randomized pack:
   - ITT per approved contrast via pyfixest `feols` (difference-in-means or ANCOVA
     per plan; approved strata as fixed effects; cluster-robust vcov per plan;
     binary outcomes as risk difference on the probability scale via linear
     probability with robust SEs — the registered §9.1 reading).
   - multi-arm: one `PrimaryContrastResultV1` per approved contrast inside the one
     `PrimaryAnalysisResultV1`; Holm `MultiplicityResultV1` when >1 confirmatory.
   - outcome-observed contribution mask; §9.2 attrition visible by arm (never
     silently dropped from denominators).
   - §9.3 diagnostics HARVESTED from the single fit + mask counts (unit
     reconciliation, arm/cluster/stratum counts, attrition by arm, covariance/
     cluster adequacy, convergence); baseline balance via the shared engine helper
     (descriptive). §9.4 sensitivities via the engine branch loop (unadjusted vs
     adjusted ITT; sensitivity vcov profile; leave-one-cluster-out loop helper
     ≤12 lines).
   - figure payload functions: assignment/attrition counts, arm summaries, balance
     measures, ordered contrasts + intervals (§17 table row 1).
2. `src/causal/estimation/nodes.py` (≤ 330) — six node bodies over `HarnessBase`
   (entry → plan+capacity → estimate → evidence → judge → close),
   `run_estimation(deps, *, analysis_id, prepared handoff ids, stage_run_id) ->
   EstimationRunResult` as a plain sequential loop with terminal-status
   short-circuit (the `run_preparation` idiom), and `open_presentation_handoff`
   building the five-id §22 manifest through the shared handoff store/T-006 gate.
   Every §20.5 boundary event emitted; commits flush-gated; conflict/not_estimable/
   invalidated/failed routing per §5.2.

## 2. Tests (≤ 550 logical)

Module-scoped fixture: approved-design → prepared-frame chain reusing the T-019 e2e
fixture helpers (RCT pack). E2E: `run_estimation` → `complete` outcome, committed
EstimationBundle, PRD-005 handoff ACCEPTED by the T-006 gate; ZERO gateway
constructions except the one claim task (assert). pyfixest parity: fixture dataset
with hand-computed diff-in-means + robust SE pinned to tolerance; cluster fixture
parity. Multi-arm: two contrasts + Holm result; atomicity (one poisoned contrast →
not_estimable, computed item still recoverable). Rerun replay: second run, new
stage_run_id, no duplicate artifacts, same terminal outcome (EV-P4-010 reading).
Attrition never mutates denominators. Judgment ceiling cap honored e2e (invalidating
diagnostic fixture → `invalidated`, handoff blocked).

## 3. Constraints

estimation ≤ 2,280 total after (+590); tests ≤ 550; modules +2. No model call in any
node except the judge node's TaskRunner path. Wave rethink shared. Committed code
wins; record deviations.

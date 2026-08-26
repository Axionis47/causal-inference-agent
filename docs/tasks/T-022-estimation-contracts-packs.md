# T-022 — Estimation substrate: contracts, packs, registries

Status: frozen for implementation
Owning PRD: PRD-004 §5, §6, §8, §26 (Amendment 1 / D-083 governs)
Depends on: T-002..T-006 (shared kernel), T-019 (preparation handoff shape)

## 1. Deliverables

1. `src/causal/estimation/contracts.py` (≤ 320 logical, module cap 350) — pydantic v2
   strict models mirroring `preparation/contracts.py` idiom:
   - `EstimationContextManifestV1` (§19.1: refs + hashes + typed plan facts; no rows)
   - `EstimationPlanV1` (§6.1 fields; deterministic content hash = plan identity)
   - `AnalysisContributionMaskV1` (§6.2: counts, reason counts, mask hash, builder
     version; the bit vector itself is a restricted object payload, NOT in the model)
   - `PrimaryContrastResultV1` (§6.3 with uncertainty INLINED per §26.2: estimate,
     units, comparator/direction, SE, confidence level, interval bounds, optional
     p-value, uncertainty method, finite-sample correction, contributing counts,
     mask ref, estimator id/version/params, convergence status, method-specific map)
   - `PrimaryAnalysisResultV1` (non-empty ordered `primary_items`, atomic status)
   - `MultiplicityResultV1` (policy id, adjusted quantities per contrast)
   - `CrossFitAssignmentV1` (§10.2 counts + mapping-object ref + hash + profile id)
   - `DiagnosticResultV1` / `SensitivityResultV1` (§14.1 result shape; §15 branch
     fields: execution status, policy result, values map, denominators, mask hash,
     interpreting rule id, versions)
   - `FigureDataArtifactV1` (§17 fields; typed series/points as bounded lists)
   - `EvidenceBundleV1` (§26.2: kind in {diagnostic, sensitivity, figure_data},
     ordered result refs, terminal-status counts, parents)
   - `JudgmentCeilingV1` (per-item rows: ceiling, triggering rule ids, evidence
     refs; overall = most restrictive)
   - `NumericalEnvironmentManifestV1` (§6.4, flat)
   - `EstimationBundleV1` (§5.1 minus the consolidated bundles), `EstimationOutcomeV1`
     (§5.2 closed status set), status/severity/ceiling enums (§14.2, §16.1, §16.3)
   - one `EstimationError` class (PreparationError idiom)
   ClaimJudgment/ClaimItem/ClaimReviewContext belong to T-025's `judge.py`, NOT here.
2. `src/causal/estimation/packs.py` (≤ 140) — `EstimationPackV1` row model +
   fail-closed loader for `registries/method-pack-estimation.v1.json` + typed
   accessors (estimator params, required diagnostic rows with severity+thresholds,
   sensitivity branch rows with param deltas + comparison rule, figure-builder ids,
   estimator-input schema, allowed mask rules, not-estimable/invalidation rule ids).
   Loader idiom = `preparation/plans.py::load_preparation_packs`.
3. `registries/method-pack-estimation.v1.json` (~280 declarative) — four packs (rct,
   aipw, did, rdd) per PRD-004 §8..§12, Amendment-1 readings. DiD carries both
   `simultaneous` and `staggered` profiles. AIPW carries ≥1 registered nuisance
   profile (fixed regularized GLM primary; histogram-GB as registered sensitivity
   profile). Every id referenced must exist wherever it points (registered
   operations, schema ids); invented ids fail the loader.
4. `registries/artifact-types.v1.json` — append ~14 estimation rows (producer
   estimation-harness; readers per SC §3.1 table incl. PRD-005 for bundle/judgment/
   figure rows; restricted classes for mask/fold/prediction payload carriers).

## 2. Tests (≤ 300 logical)

Strict round-trips + rejection cases per model (extra field, wrong enum, empty
primary_items, self-inconsistent counts); loader fail-closed (unknown key, missing
severity, duplicate pack id, unregistered reference); accessor correctness for all
four packs; EvidenceBundle kind/status-count invariants; canonical-hash stability
fixtures for plan and mask models.

## 3. Constraints

estimation ≤ 460 total this task; declarative ≤ +410; tests ≤ 300; modules +2 (68/86
used after). No imports from `causal.design` or `causal.preparation` internals —
shared kernel only (canonical, envelope, validation, frames). Pre-code projection per
SC §14.1.2; one rethink for the WAVE (T-022..T-029 share it). Committed code wins over
this spec; record deviations for the ledger.

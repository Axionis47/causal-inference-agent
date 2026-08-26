# T-025 — Claim judgment: contracts, bounded claim-review call, deterministic validator

Status: frozen for implementation
Owning PRD: PRD-004 §16, §20.3 (claim-task tracing), SC §5.4 row PRD-004
Depends on: T-022, T-024

## 1. Deliverables

1. `src/causal/estimation/judge.py` (≤ 250 logical):
   - `ClaimReviewContextV1` (§16.2 receives-list ONLY: approved question/estimand/
     method/population/timeframe, assumptions + alternative graphs, typed primary
     and sensitivity SUMMARIES, diagnostic statuses + bounded values, missingness/
     contribution summaries, ceilings + required qualifications, artifact ids per
     statement — no rows, arrays, predictions, weights, or figure payloads),
     `ClaimItemV1`, `ClaimJudgmentV1` (§16.3 fields; closed status set; no
     model-confidence field).
   - context assembly from frozen artifacts (allowlist-driven; anything outside the
     receives-list is structurally absent from the model).
   - the ONE model call through shared `TaskRunner` (`shared/agenttask.py`) with
     `result_schema`, empty tool allowlist, one initial response + at most two
     targeted corrections for the same stable validation code (§16.2, D-067 detail
     discipline); exhaustion → `EstimationOutcome failed` with blocker.
   - deterministic claim validator (= wall 13): one ordered item per primary
     contrast; per-item and overall status ≤ ceiling; every substantive statement
     cites a committed design claim or frozen result artifact id (shared
     `collect_ids` resolution); required qualifications present; unsupported or
     ceiling-exceeding drafts → targeted correction, then fail.
   - deterministic `not_estimable` judgment path with NO model call when no
     complete primary result exists (§16.3 last rule).
2. `prompts/estimation/claim-review.v1.txt` (~50 declarative) — hydrated-context
   template per the D-064..D-071 prompt lessons: renders the closed vocabularies it
   must echo (ceilings, statuses, artifact ids, parent ids), the §16.2 may/may-not
   lists, and cite-or-omit discipline. JSON response contract via result_schema.

## 2. Tests (≤ 250 logical)

Fake-gateway TaskRunner: golden draft → committed judgment; ceiling-exceeding draft
→ correction with the exact stable code → compliant retry accepted; two-correction
exhaustion → failed + one blocker; citation of an uncommitted id rejected; context
assembly excludes forbidden classes (assert absent keys against a poisoned frozen
fixture); deterministic not_estimable path commits without gateway construction;
status↔outcome mapping table (§16.3) pinned.

## 3. Constraints

estimation ≤ 1,690 total after (+250); declarative ≤ +50; tests ≤ 250; modules +1.
Model profile = frozen shared gemini-2.5-flash profile (SC §10.4); no new gateway
code. Wave rethink shared. Committed code wins; record deviations.

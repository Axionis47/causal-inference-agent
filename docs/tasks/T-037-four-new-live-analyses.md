# T-037 — Four new live analyses

Status: in progress

Frozen acceptance build: `2b08b914c2f9acb4a84b32abb36cb197f440456772114bb45b10ea1b4b47678c`.
That build passes 1,759 tests (3 skipped); Ruff and strict typing pass. The focused
live method replay now passes task validation and requests two in-scope source
clarifications; it is not a compiled or approved final analysis.
The complexity check remains `blocked_complexity_budget`. Exact check logs and
budget dimensions are retained under
`output/four-verified-20260909/verification/build-9/`. Fresh live acceptance runs use
`output/four-verified-20260909/`; the older directory remains historical evidence.

The first fresh RCT attempt exposed an additional reference-repair defect: omitting
a declared schema default was incorrectly treated as a semantic change. Its exact
three responses are retained under `verification/rct-reference-reproduction/`.
The narrow fix compares strictly parsed defaults, retaining schema, scientific and
valid-reference guards. The reproduction failed before the fix and passes after;
the failed attempt is retained under `attempts/reference-default-guard/`.

The second attempt stopped on an HTTP429 with no retained provider message. Its
specific subtype remains unknown; a later availability probe succeeded. The
gateway now recognizes only Google's exact documented temporary-capacity response
for bounded same-identity retries (one then two seconds, three total attempts).
Hard quota and unknown429 responses remain terminal. Before/after regressions,
the official documentation, and the original evidence are retained under
`verification/` and `attempts/provider-http429/`; no model/profile substitution was
made.

The third attempt reached clarification but its resumed causal-context response
hit the fixed output-token limit. Lossless grouping of identical source text and
concise graph guidance reduced the exact prompt by 27.7%, retaining all evidence
IDs, text, hypotheses and accepted facts. A focused live replay of that one request
then completed and passed walls1–5 before another full run. Its response is
verification evidence only and is not reused as a final analysis decision. The
historical failure and exact replay are under `attempts/causal-output-truncation/`
and `verification/causal-context-truncation/`.

The fourth attempt exposed model-authored prerequisite requests for lower-ranked,
incompatible methods. The model preferred RCT but asked for an RDD cutoff; the
truthful unknown answer stopped the design before compilation. Wall6 now returns
such requests for targeted correction using registry method scopes, preserving
the ranking and all necessary preferred-method requirements. A focused live
method replay passed walls1–6. An additional saved-artifact RCT replay passed
compilation, preparation, primary estimation, eight diagnostics, five sensitivities
and all figure builders, with independent numerical agreement. These are
verification copies, not approved final analyses. The original attempt remains at
`attempts/irrelevant-method-context/`.

The fifth fresh attempt exhausted its diagnostic budget and then stopped on a
truncated method response. It is retained under `attempts/method-output-truncation/`.
Two focused probes reproduced additional
correction defects: original diagnostic feedback was lost after a later validation
error; semantic correction omitted the prior status and missing-requirement list;
one requirement error hid another until a later wall; and the response schema still
allowed diagnostic requests when none remained. The corrections preserve the fixed
model/profile and budgets, full prior model-owned decisions, source/accepted-fact
separation and strict reference guards. A draft is rejected before commit when its
diagnostic request count exceeds the actual remaining budget.

Probe evidence is retained under `verification/model-output-correction/`, including
the original captures and both failed focused probes. An answer-only observer in
the standalone verification script records safe token counts and non-thought answer
parts for future truncations without changing gateway behavior. Private reasoning
is excluded, and truncated output remains rejected. These probes are diagnostic
evidence, not final analyses or design approvals.

After the third focused probe passed task validation, the integrated checks above
passed and the sixth fresh RCT attempt started on the frozen build. The final
observed-level comparator check now also runs before a complete RCT/AIPW proposal
is committed, using existing diagnostic observations and the unchanged compiler.

The sixth attempt reached a full compiled-design review. The independent reviewer
requested graph corrections: the control arm must not be a separate causal parent,
and the treatment effect must remain a hypothesis before estimation. Reading the
original QuestionRecord resolved a weighting disagreement: preserve the requested
unweighted stratum-fixed-effects estimator and disclose its implicit stratum weights,
approximate uncertainty policy and target limitations. Original and revised review
responses remain archived. The approval view now includes the exact original input
and registered policy with hashes, including all four methods' selected profiles.
Saved-packet replay and all-four plan-compiler parity tests passed. This display-only
change did not alter estimators, registry defaults, calculations or thresholds.
The sixth case entered its normal second design revision on build 8, retaining
the original design and accepted source facts. That revision then failed because
an invalid source citation required changing both its evidence ID and its exact
quote, while reference-only correction froze the quote. The saved second model
attempt changed only that citation pair; all scientific fields stayed identical.
The full attempt remains under `attempts/citation-pair-repair/`, with exact
reproduction under `verification/rct-revision-intent/`. No final report was
accepted. The runtime has no supported terminal-design recovery command; a fresh
run will follow the focused regression and integrated checks for this fix.

## Requested outcome

Finish the current implementation and execute four complete public-data analyses,
one each for randomized experiments, AIPW, difference-in-differences, and sharp RDD.
Each run uses fresh analysis state, live model decisions, the production runtime,
an explicitly reviewed design approval, and a verified exported presentation.

## Scope

- Repair the observed contradiction between reference-only corrections and later
  semantic corrections; preserve the reference guard while its repair is active.
- Select source-documented datasets that satisfy the existing method requirements.
- Add a resumable four-case execution command with source/input fingerprints,
  review packets, terminal results, and exported artifacts.
- Preserve scientific qualifications and failures. A safe refusal is useful
  evidence but does not count as one of the four delivered analyses.

## Acceptance

The source and transformed input are fingerprinted, all five stages actually run,
the exact design bundle is reviewed before approval, and each completed result
exports its frozen charts and summary with hash verification. The result records
the model calls and dataset-specific limitations. Focused regressions, the full
test suite, lint, strict source typing, and complexity checks are recorded.
This execution exercise does not constitute approval of the separate human-gold
release gate.

## Approved correctness sweep — 2026-09-09

The user approved a focused implementation and fresh acceptance batch. Before
more live calls, reproduce the saved failures and repair graph edge identities,
revision feedback inheritance, measurement units, RDD diagnostic applicability,
and sensitivity disclosure. An absent approved baseline covariate makes only the
RDD covariate-continuity check inapplicable, with a mandatory qualification; it
does not relax required diagnostic failures in other circumstances.

Preserve estimator calculations and registered sensitivity thresholds. Display
the actual comparison criterion and every branch interval, including failures.
Count execution and reviewed acceptance separately. Acceptance binds numerical
verification and chart/report review to exact input and exported bundle hashes.
Run Rock the Vote first through all stages, followed by the other three in
parallel. A new failure needs an offline reproduction and regression before a
further live attempt. Existing complexity-budget breaches remain recorded release
limitations; this sweep does not change their ceilings.

## Change log: artifact versus context engineering

This task includes both categories of change. Artifact-contract changes alter what
is persisted, rendered, cited or accepted: graph identity validation; measurement
unit/provenance fields; categorical balance and RDD figure payloads; unavailable
diagnostic and failed-sensitivity states; criterion-specific sensitivity evidence;
truthful RCT covariance metadata; and hash-bound execution/acceptance records.

Context-engineering changes alter the bounded information and feedback supplied to
model tasks: compact evidence catalogs; exact semantic-batch scopes; immutable full
correction baselines; safe reference-array repair; inline settlement for supporting
unknowns; prior-review feedback in later revisions; source-evidence separation; and
prompt language for NHEFS timing, Head Start units and sensitivity criteria.

The distinction matters for audit: an artifact change is checked through schema,
lineage, rendering, numerical and export tests, while a context change is checked
through prompt-isolation, correction-loop, revision-inheritance and source-only
review tests. These changes did not authorize model decisions or human approval by
themselves. The separate human-gold release gate remains open.

The scoped citation-pair repair reproduces the original three-response failure and
accepts the exact valid second response with one commit, while adversarial scientific
and valid-citation changes still fail. Accepted-fact provenance is now visible without
granting citation authority; intent reuses an accepted grain without a redundant
interpretation. Build 9 passes 1,759 tests plus lint and typing; complexity remains
blocked. The seventh fresh RCT attempt follows these checks.

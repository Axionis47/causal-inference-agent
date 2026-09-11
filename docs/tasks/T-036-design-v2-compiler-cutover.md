# T-036 — Design V2 compiler cutover

Status: complete
Owning PRDs: PRD-002; PRD-003 §4; PRD-004 §4; PRD-005 §5
Depends on: T-014, T-019, T-029, T-032, T-035

## 1. Decision

Replace the superseded PRD-002 execution path in one pre-production cutover. The design stage
may use models to propose meanings, causal structure, missing context, and a ranking among
compiler-proven feasible alternatives. Models never decide computability, produce executable
preparation/estimation parameters, approve a design, or route their own failures.

There is no V1 design executor, compatibility reader, or artifact translator after this task.
Unfinished analyses restart from the immutable intake CSV and causal question and require a new
approval. The repository has no valuable deployed V1 data, so the design migration is rewritten
as the clean baseline and development/test databases are recreated.

## 2. Authoritative flow

1. Profile the selected CSV deterministically.
2. Collect bounded semantic and causal proposals from the model using closed identifiers.
3. Compile measurements, evidence, proposals, and user answers into provenance-bearing facts.
4. Enforce support requirements and ask only for missing facts a human may know.
5. Derive structural method candidates, bind and run their registered diagnostics, then derive
   the empirically feasible set.
6. Select one candidate deterministically or use one bounded model ranking when several survive.
7. Compile one immutable `CompiledDesignV2`; all preparation and estimator settings are typed
   projections of it.
8. Compute statistical feasibility and actual presentation cardinalities, render the base graph
   and all alternatives, and obtain human approval over one exact review-bundle hash.
9. Hand off only an explicitly approved bundle.

## 3. Contracts and routing

The committed V2 surface is `StatisticalProfileV2`, `AgentDesignProposalV2`, `DesignFactSetV2`,
`DiagnosticPlanV2`, `DiagnosticReportV2`, `CompiledDesignV2`, `CapacityReportV2`,
`GraphViewSetV2`, `DesignReviewBundleV2`, `DesignApprovalV2`, and `DesignOutcomeV2`.

`DesignOutcomeV2.status` is exactly `approved`, `needs_context`, `needs_data`, `unsupported`,
`changes_requested`, `declined`, or `system_failure`; absence never means approval.
`ValidationIssueV2.category` is exactly `model_fix`, `human_input`, `needs_data`, `unsupported`,
or `system_failure`. Only `model_fix` is retried, for one initial response plus at most two
targeted corrections. A repeated unchanged issue fingerprint consumes an attempt. Exhaustion is
`system_failure:agent_output_invalid`, never unsupported identification. Human questions remain
limited to two consolidated rounds.

The approved PRD-003 handoff contains exactly the selected table, compiled design, diagnostic
report, capacity report, review bundle, and approval. No agent-authored runnable-frame contract
exists. The approved estimand is copied explicitly into the estimation plan and estimator
parameters.

## 4. Registries and tools

Every method pack registers assignment mechanisms, estimands, roles, structural predicates,
diagnostic recipes, context/support requirements, preparation policy, estimator parameter
bindings, invalidation/multiplicity policy, capacity derivation, and visual evidence. Startup
fails unless every declaration and handler has exact total coverage. Runtime orchestration may
not branch on a method id.

The universal profiler and required candidate diagnostics run automatically. Design tasks have
no callable tools: the harness supplies the bounded measured results they may interpret. The
statistical inspection surface cannot write data, execute arbitrary Python/SQL, run an estimator,
or inspect a treatment-effect estimate before approval.

## 5. Acceptance

- ATT cannot become ATE; assignment-incompatible methods never reach the ranking task.
- Every required diagnostic has resolved typed bindings; missing variation or structure becomes
  `needs_data`, missing real-world context becomes `needs_context`, and no applicable registered
  method becomes `unsupported`.
- RDD uses the approved running column and cutoff; DiD uses the approved unit/group and time
  structure; row-as-unit is derived before validation.
- Capacity uses measured arms, contrasts, cohorts, periods, event times, cutoff sides, and series;
  not-applicable is not encoded as zero.
- N causal alternatives produce N+1 graph views and the human approves their containing bundle.
- Every executable contract field is live under mutation tests; identical frozen inputs and
  registry versions produce the same compiled hash.
- The superseded design contracts, runtime branches, prompts, registrations, fixtures, and
  downstream entry adapters have zero remaining references.
- RCT, AIPW, DiD, RDD, ambiguous, needs-context, needs-data, unsupported, and system-failure
  end-to-end fixtures pass together with schema-perturbation tests, lint, strict typing, and the
  repository complexity checker.

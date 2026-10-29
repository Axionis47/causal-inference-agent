# 11. Adjustment workflow

[Component map](README.md) · [Terms explained](terms.md) · [Source ownership](source-index.md) · [Evidence and discrepancies](evidence.md)

## Purpose and boundary

Analyzes supported causal targets using admitted structural accounts and explicit identification roads. It owns graph reasoning, identification, estimator eligibility, frozen execution and interpretation. Its scope includes observed adjustment and specific IV, front-door, mediation, modifier and structural-zero endpoints; the package name alone does not describe that scope.

## Internal parts

| Part | Current implementation | Responsibility |
|---|---|---|
| Desk-facing adapter | [family package](../../causal_agent/families/adjustment): `family.yaml`, `design.py`, `handoff.py`, `probes.py`, `overlap.py` | Requirements, family context block and pre-analysis probes |
| Policy | [knowledge](../../causal_agent/families/adjustment/lane/knowledge) | Endpoints, rungs, estimator/check capabilities, graph motifs, rivals and diagnostics |
| Reasoning | [reason](../../causal_agent/families/adjustment/lane/reason.py), [views](../../causal_agent/families/adjustment/lane/views.py), [selection](../../causal_agent/families/adjustment/lane/selection.py), [support](../../causal_agent/families/adjustment/lane/support.py), [checks](../../causal_agent/families/adjustment/lane/checks.py) | Ledger, graphs, identification candidates, support, estimator selection and freeze |
| Execution | [execute](../../causal_agent/families/adjustment/lane/execute.py), [engine](../../causal_agent/families/adjustment/lane/engine) | DoWhy/estimator adapters, declared fallback, falsifications, sensitivity |
| Artifacts | [artifacts](../../causal_agent/families/adjustment/lane/artifacts.py), [postviz](../../causal_agent/families/adjustment/postviz.py) | Target-aware interpretation, comparison and figures |

`graph.py`, `nodes.py`, `state.py`, `contracts.py`, and `prompts.py` connect these parts to the shared runtime.

## Inputs and outputs

Consumes an admitted `Handoff` and its CSV. Local [contracts](../../causal_agent/families/adjustment/lane/contracts.py) include `Graph`, `Estimand`, `Candidate`, `EstimatorChoice`, relationship records and frozen `Design`. Output includes candidate/ledger history, designs, fixed lead selection, estimates, diagnostic results, interpretations and terminal limitations.

The endpoint catalogue covers unadjusted ATE; backdoor ATE/ATT/ATC; general adjustment; binary Wald LATE; scalar, multiple and conditional homogeneous IV; linear front-door; linear natural direct/indirect effects; baseline modifiers; and positively supported structural zero. Exact endpoint IDs and obligations are authoritative in [the generated ladder](../../docs/ladders/adjustment.md).

## Contracts

| Boundary | Enforcement | Failure |
|---|---|---|
| Context → reasoning | Shared admission and restricted views/tools | Context deficiency or intake stop |
| Relationships → graphs | Sourced ledger, graph validity and bounded rival enumeration | Unidentified target or explicit uncertainty limitation |
| Candidates → estimator | Code computes allowed methods; singleton chosen by code; multi-option choice constrained to offered IDs/names | Gate rejection or selection exhaustion |
| Diagnostic code → choice | At most three calls; projected treatment/adjustment data excludes outcome; each pick cites successful finite numerical evidence | No usable evidence stops commitment |
| Frozen `Design` → fit | Seal, support/input checks, predeclared estimator and equivalent-target fallback | Failed attempts retained; no return to effect-informed redesign |

Frozen design holds context and ledger identity, graph/certificate, target, support, weights, contrasts, estimator/inference parameters and permitted diagnostics. Missing requested subgroup results cannot be replaced by the overall reference estimate.

## Decision authority

Human context establishes the study account; model judgments propose relationships and choose within offered methods. Code identifies admissible roads, checks support and seals execution. Lead ranking precedes target-effect fitting. An association cannot confirm a causal arrow, and a missing arrow cannot establish structural zero.

## Dependencies

Uses common/profile/memory/viz/lane/explore layers. DoWhy imports belong in its engine. The reasoning, execution and artifact import contracts are separate; family-specific method knowledge remains in this package.

## State and storage

Outer snapshot belongs to the desk. The lane writes under `.artifacts/runs/<dataset>-dowhy-<id>/`, including table, result, artifacts, figures, report and diagnostic attempt files. Each frozen candidate has its own design ID within one authorized run.

## Worked walkthrough

Given an admitted backdoor question → validate context and pair → seed supplied relations → judge mechanism/relationships → enumerate identified candidates → check support → choose an eligible estimator → freeze target, graph, rows and inference → fit each design/contrast → retain fallback failures and diagnostics → interpret the recorded target and expose figures. A successful companion does not silently replace a failed lead.

## Verification

**Code-verified:** current rung declaration, choice gate, frozen contract and execution/assembly. **Test-supported:** [lane](../../causal_agent/families/adjustment/tests/test_lane.py), [selection](../../causal_agent/families/adjustment/tests/test_selection.py), [motifs](../../causal_agent/families/adjustment/tests/test_motifs.py), [space](../../causal_agent/families/adjustment/tests/test_space.py). **Runtime-observed (retained report only):** [students selector readback](../../docs/acceptance/code-assisted-estimator-selection.md) reports repaired numerical evidence and a propensity-weighting choice under an explicitly conditional design. It is not full live-data endpoint coverage or a new run performed for this handbook.

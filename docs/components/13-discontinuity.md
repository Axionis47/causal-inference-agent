# 13. Discontinuity workflow

[Component map](README.md) · [Terms explained](terms.md) · [Source ownership](source-index.md) · [Evidence and discrepancies](evidence.md)

## Purpose and boundary

Analyzes admitted threshold designs. It keeps cutoff effects distinct from effects inside a fixed window, and receipt effects distinct from eligibility or sharp-assignment effects. It owns score normalization, local support, candidate policy, frozen execution and RD-specific inference.

## Internal parts

| Part | Current implementation | Responsibility |
|---|---|---|
| Desk-facing adapter | [family package](../../causal_agent/families/discontinuity) | Family requirements, design/handoff and probes |
| Reasoning | [reason](../../causal_agent/families/discontinuity/lane/reason) | Validate context, score, covariates, line/assumptions, candidates and freeze |
| Canonical data/diagnostics | [shape](../../causal_agent/families/discontinuity/lane/engine/shape.py), [diagnostics](../../causal_agent/families/discontinuity/lane/engine/diagnostics.py) | Score orientation, support, balance/density and selectors |
| Execution | [execute](../../causal_agent/families/discontinuity/lane/execute.py), [fit](../../causal_agent/families/discontinuity/lane/engine/fit.py), [postchecks](../../causal_agent/families/discontinuity/lane/engine/postchecks.py) | Fit frozen design, falsifications, sensitivities and modifier records |
| Artifacts/policy | [artifacts](../../causal_agent/families/discontinuity/lane/artifacts.py), [knowledge](../../causal_agent/families/discontinuity/lane/knowledge) | Target-aware explanation, comparison groups, figures and declarations |

## Inputs and outputs

Consumes admitted context, CSV, score/cutoff/side/equality/timing and requested locality/comparison. [Contracts](../../causal_agent/families/discontinuity/lane/contracts.py) include score/shape, covariate roles, line and assumption judgments, bandwidths, candidates, frozen `Design` and `RDInterpretation`.

Six endpoints combine sharp, fuzzy receipt, or eligibility effects with local-polynomial or local-randomisation identification. [The ladder](../../docs/ladders/discontinuity.md) defines exact obligations. Regression kink and multiple cutoffs are not supported endpoints.

## Contracts

| Boundary | Enforcement | Failure |
|---|---|---|
| Account → score/table | Admitted rule, score timing, side/equality and source-role validation | Essential omissions return deficiency; invalid roles stop |
| Line judgment → code | Up to three optional scripts; score/eligibility, optional receipt/cluster and approved predetermined covariates only | Outcome/proxy leakage or failed numerical provenance refuses the record |
| Reasoning → selector | Selector policy fixed before deterministic bandwidth/window computation | Unsupported policy or inadequate support remains explicit |
| Frozen design → execute | Integrity seal, canonical-table SHA256 and frozen row membership | Input change returns failed design records |
| Result → interpretation | Design/endpoint/target identity, exact recorded effect and citations | Wrong target or unsupported statement is gated |

`Design` includes score, target, rows, canonical fingerprint, certificate, support, weighting and selector policy, estimator/inference and declared diagnostics. Polynomial selectors may use outcomes deterministically under that fixed policy; free-form reasoning does not receive target outcomes. Randomisation confidence sets retain disconnected regions and unresolved tails.

## Decision authority

Human context fixes the threshold rule and causal target. Models classify permitted covariates and judge assumptions with cited evidence. Code performs density/balance/support checks and selector operations. Passing balance or observing unique scores does not establish continuity or random assignment.

## Dependencies

Uses rdrobust/rddensity and the implemented randomisation adapters, shared runtime, and isolated diagnostic coding. Effect execution is separated from reasoning. Family artifacts build RD figures directly; there is no required separate `postviz.py` file in this family.

## State and storage

The run folder contains canonical tables, score/binned products when produced, design records, results, report, figures and diagnostic visits. Frozen design IDs and target-qualified addresses distinguish receipt and eligibility records even when they share a dataset.

## Worked walkthrough

Admit a sharp cutoff question → validate score/cutoff/side → canonicalize → classify predetermined covariates → compute density/balance → judge line and assumptions → enumerate supported candidates → fix selector policy and freeze → verify table fingerprint and fit → interpret cutoff assignment effect. If fuzzy receipt fails, a separately valid eligibility result remains a distinct result.

## Verification

**Code-verified:** current rungs, frozen contract, fingerprint/row checks and interpretation contract. **Test-supported:** [lane](../../causal_agent/families/discontinuity/tests/test_lane.py), [local randomisation](../../causal_agent/families/discontinuity/tests/test_local_random.py), [restricted diagnostics](../../causal_agent/families/discontinuity/tests/test_diagnostics.py). **Runtime-observed (retained report):** [Senate readbacks](../../docs/acceptance/lane-diagnostic-code.md) describe diagnostic and ordinary sharp-polynomial runs. Full endpoint coverage and cross-platform sandbox behavior remain unestablished by this pass.

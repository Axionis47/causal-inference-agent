# 12. Difference-in-differences workflow

[Component map](README.md) · [Terms explained](terms.md) · [Source ownership](source-index.md) · [Evidence and discrepancies](evidence.md)

## Purpose and boundary

Analyzes admitted before/after and panel designs with explicit adoption, comparison support and averaging policy. It owns canonical panel shape, assumption reasoning, candidate targets, estimator/inference selection and frozen execution. A treated-cell average and a projection coefficient remain different targets.

## Internal parts

| Part | Current implementation | Responsibility |
|---|---|---|
| Desk-facing adapter | [family package](../../causal_agent/families/diff_in_diff): declarations, design/handoff and probes | Gather assignment/time/target context and pre-analysis support |
| Panel and targets | [shape](../../causal_agent/families/diff_in_diff/lane/shape.py), [targets](../../causal_agent/families/diff_in_diff/lane/targets.py), [history](../../causal_agent/families/diff_in_diff/lane/history.py) | Canonical panels, target identity and revision events |
| Reasoning | [reason](../../causal_agent/families/diff_in_diff/lane/reason) | Mechanism, comparison, controls, diagnostics, rival support and lead selection |
| Execution | [execute](../../causal_agent/families/diff_in_diff/lane/execute.py), [engine](../../causal_agent/families/diff_in_diff/lane/engine) | Frozen panel fit, inference, falsifications and sensitivity |
| Artifacts/policy | [artifacts](../../causal_agent/families/diff_in_diff/lane/artifacts.py), [postviz](../../causal_agent/families/diff_in_diff/postviz.py), [knowledge](../../causal_agent/families/diff_in_diff/lane/knowledge) | Interpretation, figures, declarations and capability limits |

## Inputs and outputs

Consumes admitted `Handoff` plus family block and CSV. [Local contracts](../../causal_agent/families/diff_in_diff/lane/contracts.py) define groups, periods, shape, ledger operations/readings, estimator choices, candidates and frozen `Design`.

Six structural endpoints cover paired two-wave ATT, common-adoption projection, untreated-cell imputation, imputation on an accepted prefix, never-treated cohort comparisons, and clean-control local projections. Exact configurations/estimators are in [the generated ladder](../../docs/ladders/diff_in_diff.md). Results retain the requested averaging policy and fixed lead choice.

## Contracts

| Boundary | Enforcement | Failure |
|---|---|---|
| Handoff → panel | Context validation, metadata mappings, layout and assignment checks | Typed deficiency or shape refusal |
| Reasoning → evidence | Source-classified ledger and decision-specific views | Diagnostic association cannot confirm parallel trends or another causal assumption |
| Probe → code | Three optional calls; all metadata rows remain, but outcome/safe covariate values are exposed only before earliest adoption | Source-role violations or unusable attempted evidence fail gates |
| Candidate → freeze | Support and target policies, estimator gate and lead ranking | Unsupported target/support produces a recorded stop |
| Frozen design → execution | `assert_integrity`, saved panel and exact row-ID order | Mismatch stops execution; no redesigned retry |

The frozen contract names formula, first/second stages when applicable, controls, weights, coefficients, clustering/variance policy, row IDs, panel path and target. Fewer than three assignment clusters marks inference unavailable. Profile p-values use the declared multiplicity adjustment.

## Decision authority

The person supplies the assignment account, requested averaging and any accepted prefix restriction. Model judgments propose assumptions/control classifications and choose eligible estimators. Code computes support and preserves target identity. Successful alternative weighting does not authorize a replacement target.

## Dependencies

Uses pyfixest adapters in its engine, shared ledger/episode/result machinery, and exploration's isolated diagnostic tool. Family reasoning cannot import effect-fit modules; execution cannot reopen reasoning.

## State and storage

Run output includes canonical panel data and frozen per-design panels, results, report, figures and diagnostic files beneath the configured run root. Candidate/history records preserve alternative decisions; the desk owns the immutable input snapshot.

## Worked walkthrough

For paired before/after data → verify the two-wave layout and comparison → admit the requested ATT → seed the assumption ledger → settle mechanism/comparison/controls → skip probe coding when evidence suffices → freeze panel, formula and inference → execute → report the estimate with unavailable inference if only two assignment clusters exist.

## Verification

**Code-verified:** rungs, local Design, diagnostic projection and execution gates. **Test-supported:** [policy](../../causal_agent/families/diff_in_diff/tests/test_policy.py), [shape](../../causal_agent/families/diff_in_diff/tests/test_shape.py), [diagnostic code](../../causal_agent/families/diff_in_diff/tests/test_diagnostic_code.py), [probes](../../causal_agent/families/diff_in_diff/tests/test_probes.py). **Runtime-observed (retained reports):** [paired run](../../docs/acceptance/readbacks/did-paired.md), [Cigar stop](../../docs/acceptance/readbacks/did-cigar.md), and [diagnostic readbacks](../../docs/acceptance/lane-diagnostic-code.md). Those observations are narrower than complete endpoint acceptance.

# Actual producer coverage

The live runtime uses analysis/integration, not the separately exposed
ApprovedPlan/AnalysisEvidence API. The numerical execution path has been preserved
so the post-analysis refactor does not silently substitute estimators or RNG settings.

| Required fact | Exact source | Receiver treatment |
| --- | --- | --- |
| Analysis identity and hashes | Shared artifact envelopes and handoff | Check bytes, type, analysis ID and references |
| Approved question, targets, units and roles | CompiledDesign reached from numerical bundle | Immutable; known legacy role names normalized only at the reader boundary |
| Approval and causal DAG | PreparedFrameBundle → DesignApproval → DesignReviewBundle → GraphViewSet → base CausalGraphView → CausalContext | Follow this exact chain; never select latest |
| Attempted request | EstimationPlan, context and prepared frame references | Compare actual methods, roles, columns, row-set and contrast bindings |
| Estimates and uncertainty | PrimaryAnalysisResult | Preserve values, units, order, identities and counts |
| Expected diagnostics | EstimationPlan.required_diagnostics | Reject missing/extra/duplicate records |
| Diagnostic status/evidence | DiagnosticResult | Explicit failure is accounted-for evidence; never a pass |
| Expected sensitivity branches | EstimationPlan.required_sensitivity_ids | Match exact branches and preserve their results |
| Branch diagnostic coverage | Not supplied by current producer | Disclose lack of branch-specific checks |
| Scientific supporting measurements | AnalysisSupportingData | Raw finite values preserved; unavailable nonfinite values explicitly listed |
| Full machine-readable descriptions/units for every support series | Incomplete | Tables can disclose raw evidence; refuse unsupported quantitative encodings |
| Failure before an analysis response | New failure outcome envelope pins attempted EstimationPlan; old receipts may lack it | Report the exact failed attempt when bound; otherwise block with analysis-owned attempt_request_unavailable |

New NumericalBundle and AnalysisSupportingData artifact types replace the old
claim-judgment/figure-dependent bundle on the live route. Primary, diagnostic and
sensitivity computation remains upstream. Comparison tables can combine compatible
primary and sensitivity values with exact per-row lineage.

A receiver schema/encoding gap names post_analysis as owner. A source retrieval
failure names storage. Scientific mismatches name analysis or design, with source,
path, expected value, received value and required action. No issue asks the LLM to
repair source artifacts.

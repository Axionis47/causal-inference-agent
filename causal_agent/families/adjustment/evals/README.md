# Adjustment endpoint validation

The endpoint catalogue is `../lane/knowledge/endpoints.yaml`. [motifs.yaml](../lane/knowledge/motifs.yaml) records the ten requested illustrative graph motifs plus natural mediation, modifier and blocked-trap examples; pure tests validate their actual identification certificates. The endpoints are finite semantic analysis classes, not a claim that all causal graphs fall into exactly ten DAGs. Each class can admit many graph structures. The common desk admission reads the same requirements the lane checks.

## Executable scope

| Endpoint | Target and executable method | Additional context and structural conditions |
| --- | --- | --- |
| `total.unadjusted` | ATE, linear or logistic outcome regression | Empty valid adjustment set; both treatment arms observed |
| `total.backdoor` | ATE, ATT or ATC; regression for ATE, propensity stratification/weighting/matching for all three targets | Observed valid backdoor set, covariate support and declared graph assumptions |
| `total.general_adjustment` | ATE, linear or logistic general-adjustment regression | DoWhy general-adjustment certificate with its exact observed set |
| `iv.wald_late` | Complier LATE, binary Wald ratio | Binary 0/1 instrument, exclusion, exogeneity and positively supplied monotonicity |
| `iv.linear_single` | ATE, DoWhy scalar IV | One numeric instrument, exclusion/exogeneity, homogeneous additive effect |
| `iv.linear_multiple` | ATE, statsmodels 2SLS | Multiple numeric instruments, exclusion/exogeneity, homogeneous additive effect |
| `iv.linear_conditional` | ATE, statsmodels 2SLS conditional on baseline controls | Controls block every non-treatment instrument-outcome route, controls are not treatment/instrument descendants, and homogeneous additive effect |
| `frontdoor.linear_total` | ATE, DoWhy linear two-stage regression | Singleton numeric mediator, front-door certificate and explicit additive linear-stage assumption |
| `mediation.linear_nie` | Natural indirect effect, DoWhy linear two-stage mediation | Singleton mediator, observed first/second/total adjustment certificates, sequential ignorability, no treatment-induced mediator-outcome confounder, additive linear stages |
| `mediation.linear_nde` | Natural direct effect, DoWhy linear two-stage mediation | The same natural-effect assumptions and certificates, with the direct-effect target |
| `adjustment.linear_conditional` | Baseline-modifier conditional effects, linear interaction regression | Named baseline modifiers and valid adjustment including modifier main effects; subgroup definitions frozen before outcomes are fitted |
| `structural.zero` | Structural implication of zero, with no sampling interval | Positive supplied `claim:change.no_causal_path=true` and a graph with no directed treatment-outcome path |

`identification.id_expression` is identification-only. The installed ID interface ignores observability for explicit latent-node graphs, so those graphs are blocked. Fully observed symbolic expressions are checked for observable variable references and returned as diagnostics. There is no arbitrary-expression estimator.

Continuous-dose treatment models, nonlinear/interactive natural mediation, multiple mediators, heterogeneous-effect IV without a local identification target, weak-IV robust confidence sets, and generative GCM intervention/counterfactual tasks are unsupported. Finite multivalued treatment with 2 to 12 observed levels is supported only through explicitly chosen pairwise contrasts. Numeric outcomes and explicitly encoded 0/1 binary outcomes are supported. No additional analytical libraries were added.

## Reasoning and execution policy

Every executable endpoint follows admission, fixed pair, source-preserving ledger, mechanism and relationships, bounded probe, graph/identification rivals, support checks, eligible estimator selection, integrity-sealed freeze, fit and evidence-gated interpretation. At most four unsettled structural entries produce at most sixteen rival graphs; at most eight distinct executable designs may survive. A budget excess is a visible refusal, not an omitted graph. Unspecified edges cannot establish structural zero.

The lane never interviews the person. Missing or changed admitted context returns a typed deficiency. Unknown causal relationships remain hypotheses or explicit graph assumptions. Drafted model context cannot become source-supported causal evidence. Pre-freeze outcome access is univariate scale/shape description only. Every relational tool involving the outcome, including outcome-covariate association or redundancy, is refused.

A run applies one common complete-case policy across the admitted loaded columns. Freeze records original source row IDs, contrast-specific data fingerprints and analysis-population identities. Execution verifies these before fitting. Comparison groups require identical complete targets, contrasts, scales and frozen analysis populations. Weight recipes, input rows, clipping parameters, fallback and inference are frozen. Percentile bootstrap resamples rows and refits the complete estimator, including propensity scores. DoWhy's generated propensity columns are isolated from raw refutation/sensitivity inputs.

Falsifications and sensitivities are separate records. Graph independence diagnostics run only after freeze, on a deterministic sample of at most 300 rows and at most sixteen graph-implied constraints with at most one conditioning variable, with Holm correction. Their results do not select another graph or implicate a particular causal edge. Sensitivity preserves the simulated strength grid and has no pass verdict.

## Real-source evaluation

Evaluation uses real-source datasets and their documented study accounts. The retained cases cover the existing students and marketing datasets plus a forced adjustment handoff for the government-transfer dataset. Generated protocol datasets, their generator, synthetic cases and their stored acceptance evidence have been removed. Future evaluation datasets come from Kaggle and must carry source provenance and an admitted study account.

```bash
GRPC_DNS_RESOLVER=native uv run python -m causal_agent.evals.lane adjustment students "Did completing the prep course raise math scores?"
GRPC_DNS_RESOLVER=native uv run python -m causal_agent.evals.lane adjustment --handoff causal_agent/families/adjustment/evals/handoffs/gov_transfers_forced.json
uv run python -m causal_agent.evals.trace --trace <trace-id> --thoughts
```

Pure tests still check graph identification, numerical adapters, raw-data isolation, frozen-support integrity, source provenance and comparison grouping. Small in-memory numerical inputs are unit tests, not evaluation dataset generators. These checks do not establish real-data trajectory coverage for every endpoint. New endpoint acceptance requires a documented real-source run and trace inspection; the removed synthetic evidence is no longer counted as acceptance coverage.

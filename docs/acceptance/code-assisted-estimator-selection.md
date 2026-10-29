# Code-assisted estimator selection

Validated on 14 October 2026. This increment changes adjustment's existing multi-option `pick_estimator` episode. It does not change inference, refuter or sensitivity selection.

## Scope

The retained 1,000-row students CSV was used directly, without a data generator. The selector was exercised after fixing an explicitly conditional `total.backdoor` design: lunch and parental education affect course uptake and math score, and uptake may affect math score. Those outcome arrows are validation assumptions, not established study facts. This is a selector-level test, not a new Kaggle intake or full API-conversation acceptance.

Three methods were eligible: linear regression, propensity weighting and propensity matching. The code received the admitted rows with treatment and the two adjustment columns. Outcome values were withheld. The exact treatment levels and frozen method parameters were supplied separately.

## Observed run

Run `students-dowhy-51761fe8`, trace `01a13c5d-34e1-7020-bf41-2a1b6d9e6347`, completed in 41 seconds:

1. Gemini fitted an auxiliary propensity model and wrote mixed numeric/text facts. The tool reported that these were inadmissible selection evidence.
2. Gemini repaired the output to numerical facts and recorded an inverse-probability-weight ESS of 908.3189799815319. Independent recomputation reproduced this value.
3. A later script failed because it omitted an import. The failure remained recorded and was not cited.
4. Gemini selected the offered propensity-weighting method, cited the successful second step and retained propensity-model correctness as unverified.
5. The method was frozen before target-effect fitting. The existing engine completed using the unchanged 100-draw percentile-bootstrap policy.

The remote trace was read back. It contains all three scripts, their outputs, numerical evidence, the accepted choice, freeze and execution. Empty output artifacts no longer cause rejected trace uploads.

## Limits

ESS describes concentration of the auxiliary fitted weights. It does not establish overlap, effect precision, literal rows retained, model correctness or an optimal estimator. The model's phrase “suggests good overlap” must be read with that limitation; registered overlap checks remain separate evidence. The auxiliary logistic fit is not certified as the exact production propensity fit. This run validates code-assisted conditional choice and provenance, not automatic statistical-method optimality or the unchanged inference policy.

Earlier real runs exposed schema-only output, malformed scripts, invented diagnostic citations and overstated explanations. Those findings drove the numerical-evidence gate and immediate feedback. One repeated model-output run was interrupted. If the code budget ends without usable evidence, the current harness stops before requesting a final choice; final-answer retries cannot create missing tool evidence.

## Checks

`make check` passed: 425 Python tests, 49 web tests, five platform-dependent skips, lint, import contracts, types, schema and ladder freshness. Native sandbox checks exercised real pandas and logistic-regression code while blocking access to the full source table, source writes, links and child processes. Linux and Docker runtime coverage remains platform-dependent.

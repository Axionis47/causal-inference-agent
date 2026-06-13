# Live audit of the 16 representative question types (2026-06-13)

Every taxonomy type in `representative_cases.yaml` was run through the full
S1..S12 spine against the live Vertex provider (no stubs), via
`python -m src.analysis_v2.evals.live_runner <case_id>`. Result: **16/16
completed, every estimate inside its manifest truth band, every notebook
verified-running.** This file records what the runs proved and where the
analysis quality still has room to grow. It is the qualitative companion to
the hermetic gate in `workflow_evals/`.

## Scoreboard

| type | case | lane | estimate (truth) | claim | robustness |
|---|---|---|---|---|---|
| simple_effect | lalonde-nsw-experimental | observational | 1676 (~1794) | moderate | robust |
| binary_treatment | lalonde-observational | observational | 1548 (flip from -635) | exploratory | fragile |
| dose_response | advertising | observational | 0.0475 (0.046) | moderate | robust |
| multi_factor | student-multi-factor | observational | -2.20 failures | exploratory | robust |
| interaction | insurance | interaction | 1390/BMI-unit (~1434) | moderate | robust |
| mediation | synthetic-mediation | mediation | indirect 0.30 | exploratory | fragile |
| did | synthetic-did-panel | did | 1.83 (2.0) | strong | robust |
| rdd | synthetic-scholarship | rdd | 7.37 jump (8.0) | strong | robust |
| iv | synthetic-iv-late | iv | 2.05 LATE (2.0) | strong | robust |
| time_series | website-visitors-its | time_series | +411 level shift | exploratory | fragile |
| survival | heart-failure | survival | HR 0.95 | moderate | robust |
| before_after | website-before-after | time_series | +411 | exploratory | fragile |
| heterogeneous | hillstrom-hte | interaction | 0.063 interaction | moderate | robust |
| driver_analysis | telco-churn | joint regression | 0.174 contract | exploratory | robust |
| no_effect | hillstrom-null | observational | 0.0006, CI spans 0 | moderate | robust |
| mechanism_search | student-por | mediation | indirect 0.83 | exploratory | fragile |

Claim-strength calibration reads correctly: the three synthetic designs with
clean identification earn `strong`; messy observational and single-series
cases land at `moderate` or `exploratory`. No case over-claims.

## What the live runs fixed (each its own commit)

1. **Tool-result batching** (`fix(llm)`): Vertex rejected the investigator's
   first multi-call turn. One results message per turn.
2. **Survival outcome** (`fix(design)`): live intake resolved duration+event
   but left `outcome` null; the plan gate failed a well-specified question.
   Time-to-event is now promoted as the outcome.
3. **Dossier completes the adjustment set** (`feat(plan)`): lalonde ran a raw
   comparison (-$635, the famous wrong sign) while the investigator's dossier
   held all eight confounder roles unread. Pre-treatment roles now fill an
   empty covariate set; post-treatment / mediator / leakage / id roles are
   vetoed even when intake offered them.
4. **E-value vs continuous treatment** (`fix(diagnostics)`): advertising's
   textbook-correct dose-response was declared `not_supported` because
   `|slope|/sd(outcome)` collapsed the e-value to 1.10. The check now skips a
   continuous treatment, and the claim upgraded to `moderate`/`robust`.

## Room for improvement (ranked by impact on analysis quality)

1. **Estimator upgrade for the observational lane.** Several cases lean on
   plain `regression_adjustment`. lalonde-observational shows regression
   ($1548) and IPW ($445) disagreeing, correctly flagged fragile; the
   experimental truth (~$1794) sits nearer the regression. AIPW with
   cross-fitting would be doubly robust under the same assumptions and would
   narrow that gap. Pure deterministic lane work, no agency change.
2. **Overlap / estimand policy.** No case trims to the common-support region
   or switches to ATT under heavy imbalance; the report states overlap but
   the estimate still spans the whole sample. A deterministic trim-and-report
   policy in the observational lane would make the imbalanced cases honest
   about which population the number covers.
3. **EDA checks skipped under the hermetic-style stub leak into live.** The
   lalonde report says "covariate balance and propensity overlap were
   skipped." The targeted-EDA stage is not always running its design-specific
   checks on the live path; wiring those in would feed diagnostics richer
   signal and remove the "skipped" caveat from real runs.
4. **Interpretation depth in the report.** Reports state the number and the
   limitations well but rarely translate scale ("$1548 on a mean of $6793",
   "what an E-value of 1.5 would require of a hidden confounder vs the
   strongest measured one"). This is the gap between a correct answer and an
   analyst-grade one. The claim-language guardrails already exist to keep it
   honest.
5. **The investigator is read-only.** Card-Krueger reshape, the cigarette
   derived instrument, and the Rossmann store.csv join (manifest cases not yet
   in the live set) still need the sandboxed frame-preparation half of the
   investigator. The dossier (understanding) shipped; the FrameRecipe
   (preparation) is the next build and is what unlocks the wide-format and
   multi-file cases.

## How to reproduce

```bash
cd backend
python -m src.analysis_v2.evals.fixtures.cache   # hillstrom + telco, once
LOCAL_STORAGE_PATH=/tmp/live/<case> \
  python -m src.analysis_v2.evals.live_runner <case_id> --out /tmp/live/results
```

The hermetic gate (`pytest src/analysis_v2/evals/workflow_evals`) covers the
same 16 types with the LLM stubbed and the investigator on its degraded path,
so CI stays green offline.

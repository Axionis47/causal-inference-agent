# Skill: Notebook Report Standards

Load when: defining what a good causal analysis notebook looks like.

## Notebook sections — 14 sections in order

| # | Section | Source Agent | Required? | Contains executable code? |
|---|---------|-------------|-----------|--------------------------|
| 1 | Introduction | LLM-generated | Yes | No |
| 2 | Domain Knowledge | DomainKnowledgeAgent | If metadata exists | No |
| 3 | Setup | Static | Yes | Yes — imports, config |
| 4 | Data Loading | DataProfilerAgent | Yes | Yes — read parquet |
| 5 | Data Profile | DataProfilerAgent | If profile exists | Yes — summary stats |
| 6 | Data Repairs | DataRepairAgent | If repairs made | Yes — repair verification |
| 7 | EDA | EDAAgent | Yes | Yes — correlation, VIF |
| 8 | Causal Structure | CausalDiscoveryAgent + DAGExpert | If DAG produced | No — graph display |
| 9 | Confounder Analysis | ConfounderDiscoveryAgent | If confounders found | No |
| 10 | PS Diagnostics | PSDiagnosticsAgent | If PS computed | Yes — overlap, balance |
| 11 | Treatment Effects | EffectEstimatorAgent | Yes | Yes — OLS verification |
| 12 | Sensitivity | SensitivityAnalystAgent | If sensitivity run | Yes — E-value, placebo |
| 13 | Critique | CritiqueAgent | If critique exists | No |
| 14 | Conclusions | LLM-generated | Yes | No |

## Quality standards for each section

### Treatment Effects (most critical section)
- Must show at least 2 estimation methods
- Must include point estimates, standard errors, confidence intervals, p-values
- Must include OLS as a baseline for comparison
- Must state which assumptions were checked and which were violated
- Must include executable verification code that reproduces at least one estimate

### Sensitivity Analysis
- Must include E-value (how strong unmeasured confounding would need to be)
- Should include at least one of: Rosenbaum bounds, placebo test, specification curve
- Must interpret results in plain English: "the estimate is robust to [X level of confounding]"

### Conclusions
- Must state the causal claim clearly with appropriate hedging
- Must list the key assumptions underlying the claim
- Must list limitations and threats to validity
- Must not overstate certainty when assumptions are untestable

## What makes a bad notebook

- Runs all 12 methods without justifying which ones apply
- Reports p-values without effect sizes or confidence intervals
- Claims causation without discussing identification strategy
- No sensitivity analysis or dismisses robustness concerns
- Executable cells that error when run
- No verification code — results are not reproducible
- Generic conclusions that could apply to any dataset

## Review checklist

- [ ] Introduction names the treatment, outcome, and research question
- [ ] At least 2 estimation methods with justification for selection
- [ ] OLS baseline included
- [ ] Assumptions stated and testable ones checked
- [ ] Sensitivity analysis interprets robustness
- [ ] Executable cells run without errors
- [ ] Conclusions appropriately hedged based on identification strength
- [ ] Limitations section is honest about untestable assumptions

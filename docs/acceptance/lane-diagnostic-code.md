# Restricted diagnostic coding in DiD and discontinuity

Validated on 14 October 2026 (America/New_York), using retained sourced datasets. No dataset generator was added.

## Increment

The existing DiD `probe` and discontinuity `line` steps optionally offer three arbitrary Python calls. They reuse `explore/diagnostics.py`, including the isolated sandbox, finite numerical-facts checks, citation gate, immediate error feedback and bounded stop before final commitment when no usable evidence exists. Adjustment uses the same shared implementation.

Existing evidence can support a judgment without coding. After code is attempted, a successful numerical probe must be cited. The trace retains every attempt and artifact. DiD diagnostic notes can cite evidence with `operations=[]`; a computation need not manufacture a causal-ledger update.

## Data restrictions

| Lane | Permitted code input |
|---|---|
| DiD | All assignment/time metadata rows; outcomes and confirmed safe covariates only before the earliest adoption anywhere. Later outcomes and covariate values are masked, including never-treated and not-yet-treated rows. |
| Discontinuity | All admitted rows with oriented score, exact eligibility, optional receipt/cluster and approved predetermined covariates. No target outcomes or known outcome proxies. Outcome missingness cannot change projected rows. |

Source-role checks run before reasoning tools. Outcomes cannot enter through score, receipt, cluster or other metadata aliases. Candidate collisions with canonical DiD names are rejected; valid declared unit/time/cluster mappings remain usable. Direct diagnostic callers repeat these checks. Standard endpoint, target, estimator, inference, bandwidth/window and execution policies retain their authority.

## Real Gemini runs and trace inspection

| Run | Trace | Observed behavior |
|---|---|---|
| Cigar diagnostic probe | `01a13d9e-5e4f-7e42-be41-0deca3d3c07f` | One script computed pre-period means and comparison support, cited its numerical probe and returned no causal operations. All 1,380 metadata rows remained. The ledger stayed at revision 21 and the explicitly proposed validation anticipation question remained proposed. |
| Senate diagnostic line | `01a13d9e-a031-7383-aa08-b05e8347831d` | Three scripts computed score support and uniqueness. A missing diagnostic citation was rejected; the corrected record cited a successful probe and continued to assumptions. All 1,390 admitted rows were retained without outcomes. |
| Ordinary Card–Krueger analysis | `01a13da0-28a1-72d2-a0bc-57e9dc6fe1f9` | Completed `paired_two_wave_att`, skipped unnecessary probe coding, and retained unavailable inference with only two assignment clusters. The interpretation gate required that limitation to be explicit. |
| Ordinary Senate analysis | `01a13da0-307f-79c0-ba1d-7bb8ef619b03` | Completed `sharp_local_polynomial` through the optional no-code path, including frozen execution, falsifications and sensitivities. |

These remote traces were read back, including the generated code and gate transitions. Earlier attempts with script errors, invalid numerical output or omitted citations stopped visibly.

After placing the source-role guards at their final early workflow boundaries, both diagnostic paths passed again: Cigar trace `01a13da5-e3a5-7590-8c93-fdb5dcbaa3a3` and Senate trace `01a13da5-e8f0-7431-b9c4-e538f74c0f36`. The Cigar record explicitly retained causal uncertainty and no ledger operations. Senate corrected an omitted numerical citation after the gate supplied the exact permitted probe addresses. Both final traces were read back.

Independent recomputation reproduced the Cigar pre-period means (121.11923076923077 for California and 127.34247863247862 for controls) and 45 control units. The cited Senate descriptive window of ±3.97 contained 108 observations below and 96 above, each with a unique score, and none exactly at the cutoff. This rounded diagnostic window did not change the engine's window policy.

## Limits and regression coverage

These diagnostics do not establish causal assumptions. The Cigar code computed pre-period levels, not trend slopes, and did not settle anticipation. Senate score uniqueness does not establish continuity or absence of manipulation. The diagnostic requests were explicitly labeled validation requests; no new study facts were confirmed. These runs do not claim new Kaggle intake, full API-conversation acceptance or coverage of every endpoint.

Pure-code and native-sandbox checks cover outcome-value and missingness perturbations, source-role and reserved-name aliases, cutoff orientation/equality, numerical provenance, no-code acceptance and denied access to full source tables. Existing data and generated API schemas remain unchanged.

Final `make check` passed: 477 Python tests, 49 web tests, five platform-dependent skips, lint, import contracts, types, schema and ladder freshness. The web production build also passed.

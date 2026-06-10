# Start-analysis payload (what the analysis receives)

**Status: draft, planning. Pairs with the agreed upstream contract
`docs/input-slice/confirmed-dataset-format.md`.**

This defines exactly what the analysis is handed the moment **start analysis**
is triggered, and what it is allowed to rely on. It is the boundary between the
input slice (which ends at a `CONFIRMED` record) and the analysis. We design
every analysis agent against this fixed payload, one agent at a time.

The rule that governs the whole payload: **nothing here is guessed.** Every
input that defines the question came from the user and was validated at confirm.
The analysis never infers treatment, outcome, contrast, table, or time. If a
required field is missing or invalid, that is a gate bug, and the analysis
refuses; it does not fall back to a guess.

---

## 0. When it is constructed, and the two layers

- Start analysis is a **separate explicit trigger** that reads a `CONFIRMED`
  record. It is not part of the input slice (lifecycle in the confirmed
  contract, section 8).
- The payload **is** the `CONFIRMED` record loaded into memory. The analysis
  tail receives exactly this, nothing more.
- In the state/memory split: this payload is the **initial shared STATE** the
  analysis reads. Per-agent **MEMORY** (the tool-call scratchpad shown in the
  right frame) starts empty and fills as each agent runs. Agents seal outputs
  back into STATE; they never put working memory there.

---

## 1. The payload at a glance

Five things travel together under one `job_id`:

```
THE DATA       one normalized parquet (the used table)     <- what we load
THE INDEX      manifest.json (typed description of bundle)  <- how to read it
THE ESTIMAND   user-locked treatment / outcome / time (+ contrast, open)
THE FACTS      deterministic profile (types, stats)         <- what the data is
IDENTITY       job_id, source, timestamps, run-config
```

---

## 2. Identity and run-config

| field | type | note |
|---|---|---|
| `job_id` | str (8 char) | key for everything |
| `kaggle_url` | str | validated source |
| `created_at` | datetime | job creation |
| `confirmed_at` | datetime | when the dataset was accepted |
| `status` | enum | `confirmed` on entry |
| `orchestrator_mode` | enum | which orchestrator runs it (run-config, not data) |

## 3. The data

- Format: **parquet**, one file per table under `normalized/`.
- Exactly **one table is marked `used`**: the single analysis frame. Everything
  downstream loads this one table.
- Multi-file **assembly (joining tables into one frame) is deferred** (upstream
  contract, section 4). The analysis input is the single `used` parquet.
- Other tables in the bundle exist for display only; they are not analysed.

## 4. The estimand (the locked human inputs, the no-guessing core)

From the **user**, validated at confirm against the columns of the `used` file
(upstream contract, section 6). This is the part that makes the question
well-posed.

| field | type | rule at confirm |
|---|---|---|
| `treatment_variable` | str | must be a column of the used file |
| `outcome_variable` | str | must be a column of the used file |
| `has_time_dimension` | bool | user-confirmed |
| `time_column` | str \| null | if `has_time_dimension`, must be a real column |
| `user_context` | str \| null | optional free prose, passed through verbatim |

**Open addition (section 8, needs your call):** `treatment_contrast` (what counts
as treated vs control / the comparison for a non-binary treatment). The current
system guesses a binarization threshold at estimation time; "no guessing" means
this should be locked here too.

## 5. The facts (deterministic profile, no LLM)

Computed before the gate, pure pandas/numpy, reproducible from the data alone
(upstream contract, section 7).

| field | type | meaning |
|---|---|---|
| `n_samples` | int | rows of the used file |
| `n_features` | int | columns |
| `feature_names` | list[str] | columns of the used file |
| `feature_types` | dict[str,str] | binary / ordinal / numeric / datetime / categorical / text |
| `missing_values` | dict[str,int] | per-column NaN count |
| `numeric_stats` | dict[str,dict] | per numeric column: mean, std, min, max, median |
| `categorical_stats` | dict[str,dict] | per categorical column: top value counts |
| `has_time_dimension` | bool | deterministic detection, then user-confirmed |
| `time_column` | str \| null | detected/confirmed time column |

Role-candidate guesses (`treatment_candidates`, `potential_confounders`, etc.)
are **intentionally absent**: roles come from the user, so no machine guess is
stored. They may exist as live UI hints only.

## 6. The index (manifest.json)

Per file: `name`, `format`, `size_bytes`, `n_rows`, `columns`, `tabular`,
`used`, normalized path, content hash. Plus the full raw Kaggle metadata dict
(description, column descriptions, tags, subtitle), so the metadata block
survives eviction of other artifacts.

---

## 7. Invariants the analysis may assume (no re-checking, no guessing)

Given this payload, every agent can assume, without re-validating:

1. The `used` parquet exists at the manifest path and is readable.
2. `treatment_variable` and `outcome_variable` are real columns of the used file.
3. If `has_time_dimension`, `time_column` is a real column of the used file.
4. `profile.feature_names` equals the used file's columns.
5. Everything in section 5 is reproducible from the data alone; no LLM ran.
6. The estimand inputs (T, Y, time, and contrast once added) are **human-locked**.
   The analysis must never infer, override, or "fix" them. A missing or invalid
   required input is a gate failure, and the analysis refuses rather than guesses.

If any invariant cannot hold, the dataset never reaches `CONFIRMED` and the
analysis never starts.

---

## 8. What is NOT in the payload (the analysis must produce it)

None of the following is present at start; each is an **output** the tail
computes. No agent may assume any of these exist on entry:

- EDA over the locked T/Y (correlations, balance, overlap, outliers, collinearity)
- causal structure: the DAG, variable roles, forbidden edges
- identifiability verdict and the adjustment set
- confounders
- treatment effect estimate(s) and method diagnostics
- sensitivity / robustness results
- critique / trustworthiness verdict
- the report / notebook

---

## 9. Open items to settle before designing the first agent

- **D-A. Treatment contrast.** Add a `treatment_contrast` to the locked estimand
  (section 4)? Options: (a) propose from the profile and have the human confirm
  at the gate, then lock it here; (b) leave it to the estimation agent.
  Recommendation: (a), so the estimand is fully specified with zero downstream
  guessing. Needs decision.
- **D-B. Used-file selection when multiple files.** Who marks the single `used`
  table when the bundle has more than one candidate: the human at the gate, or
  an upstream heuristic? Recommendation: human-confirmed when ambiguous, to hold
  the no-guessing line. Needs decision.

---

## 10. Plan from here (one agent at a time)

The analysis tail, in spine order, against this payload:

1. **EDA** (linear, computed, now T/Y-aware) <- first agent to design
2. **Causal structure** (DAG + adjustment set + identifiability) + DAG gate
3. **Estimation** (method + estimate + diagnostics) + possible downgrade
4. **Sensitivity** (robustness)
5. **Critique** (trustworthy enough to report) + bounded iterate loop
6. **Report** (notebook + summary)

Each gets its own doc (`01-eda.md`, `02-causal-structure.md`, ...): what it
reads from STATE, what it computes vs judges, its per-agent memory, the artifact
shown in its center pane, and its gate or loop.

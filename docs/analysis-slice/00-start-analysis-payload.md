# Start-analysis payload (what the analysis receives)

**Status: finalized (v1). All open items resolved (section 9). Pairs with the
agreed upstream contract `docs/input-slice/confirmed-dataset-format.md`.**

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
| `treatment_contrast` | obj \| null | the treated-vs-control comparison; proposed from the profile and human-confirmed at the gate (D-A, section 9). Null for a plain binary 0/1 treatment |
| `ignored_columns` | list[str] | columns dropped from analysis (e.g. a row-index); real columns, disjoint from T/Y/time; default `[]` (D-D, section 9) |
| `user_context` | str \| null | optional free prose, passed through verbatim |

### 4.1 The two context channels (separate, never merged)

Context reaches the analysis from two distinct sources. The first meaning-making
agent reads both, labeled, and treats their presence as a signal:

| channel | field(s) | provenance | when absent |
|---|---|---|---|
| analyst | `user_context` | what the human asserts / wants; authoritative | optional |
| source | `kaggle_description`, `column_descriptions` (manifest, section 6) | Kaggle's own description; informative, not authoritative | common (LaLonde: none) |

Never concatenated: the analysis weighs human intent against the source claim and
must know when either is missing. If **both** are empty, that is a low-context /
thin signal the analysis surfaces, not one it guesses past.

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

### 5.1 Ignore-candidates (proposed, not decided)

The profiler also flags likely non-features (a `Unnamed: *` column, a perfect
`0..n-1` sequence, an all-unique monotonic id) as **ignore-candidates**. These
only pre-check the gate's `ignored_columns` control; the human locks the set. By
the time analysis starts, `ignored_columns` is fixed and the analysis frame
already excludes it (section 7). No agent re-decides what to drop.

## 6. The index (manifest.json)

Per file: `name`, `format`, `size_bytes`, `n_rows`, `columns`, `tabular`,
`used`, normalized path, content hash. Plus the full raw Kaggle metadata dict
(`kaggle_description`, `column_descriptions`, tags, subtitle), so the metadata
block survives eviction of other artifacts.

This raw Kaggle metadata is the **source context channel** of section 4.1:
`kaggle_description` and `column_descriptions` are the source's claim about the
data, distinct from the analyst's `user_context`, and may be empty.

---

## 7. Invariants the analysis may assume (no re-checking, no guessing)

Given this payload, every agent can assume, without re-validating:

1. The `used` parquet exists at the manifest path and is readable.
2. `treatment_variable` and `outcome_variable` are real columns of the used file.
3. If `has_time_dimension`, `time_column` is a real column of the used file.
4. `profile.feature_names` equals the used file's columns.
5. Everything in section 5 is reproducible from the data alone; no LLM ran.
6. The estimand inputs (T, Y, time, contrast) are **human-locked**. The analysis
   must never infer, override, or "fix" them. A missing or invalid required input
   is a gate failure, and the analysis refuses rather than guesses.
7. `ignored_columns` are real columns, disjoint from T/Y/time. The **analysis
   frame** is the used table with `ignored_columns` dropped; the effective feature
   set is `feature_names` minus `ignored_columns`. No agent sees a dropped column.

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

## 9. Decisions (resolved)

- **D-A. Treatment contrast. RESOLVED: lock at the gate.** `treatment_contrast`
  is added to the estimand (section 4), proposed from the profile and
  human-confirmed at the gate, so a non-binary treatment's comparison is locked
  with zero downstream guessing. Null for a plain binary 0/1 treatment. The exact
  encoding (level pair vs threshold) is settled when the estimation agent is
  designed; the contract reserves the field now.
- **D-B. Used-file selection. RESOLVED: human-confirmed when ambiguous.** A single
  tabular file is auto-marked `used`; when the bundle has more than one candidate
  the human picks `used` at the gate. No upstream heuristic guesses.
- **D-C. Two context channels. RESOLVED: separate, never merged.** `user_context`
  (analyst) and `kaggle_description` / `column_descriptions` (source) travel as
  distinct fields with presence flags (section 4.1). Both empty raises a
  low-context flag rather than a guess.
- **D-D. Columns to ignore. RESOLVED: deterministic proposal, human lock.**
  `ignored_columns` (section 4) is pre-proposed by the profiler's index detector
  (section 5.1) and locked by the human at the gate, validated as real columns
  disjoint from T/Y/time. No LLM decides. Closes the `Unnamed: 0` leak.

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

---

## 11. Persistence and checkpointing (local-first)

The whole payload is **persisted locally**, which is exactly what makes
checkpoint-and-reload work.

On disk under `{local_storage_path}/{job_id}/` (default `./data`): `raw/`
(original Kaggle files), `normalized/<table>.parquet` (the analysis frame),
`manifest.json` (the typed index). See the upstream contract, section 2.

Run record / state:
- The job record (confirmed inputs + profile + status) persists to local JSON
  (or Firestore when enabled), and the confirmed `AnalysisState` is parked to
  `parked_states.json`. **`start analysis` reloads that parked record** to build
  this payload, so a crash or restart between confirm and run loses nothing.

Checkpointing the analysis tail (the STATE / MEMORY split, section 0):
- Each agent seals its output back into shared STATE; STATE is the durable,
  reloadable checkpoint. Per-agent MEMORY (the tool scratchpad) is ephemeral and
  is **not** persisted. So the rebuild checkpoints after every agent by persisting
  STATE and resumes mid-pipeline by reloading it, the same way the gate already
  reloads the parked `CONFIRMED` state today.

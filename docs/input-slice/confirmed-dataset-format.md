# Confirmed-dataset format (input slice output contract)

**Status: agreed. D1-D4 resolved (see end).**

This defines exactly what exists, and what an analysis is allowed to rely on,
the moment a user clicks **confirm** in the preview. It is the boundary between
the input slice and any analysis. The input slice ENDS by producing this record.
Analysis is a separate, later step that reads it. **Nothing in this contract
requires or runs an LLM.**

Grounded in `docs/input-slice/map.md` (sections 3, 5, 6).

---

## 1. The record at a glance

A confirmed dataset is four things that travel together under one `job_id`:

```
THE DATA        normalized parquet (one file per table)   <- what analysis loads
THE INDEX       manifest.json (typed description)          <- how to find/read it
THE INPUTS      user-confirmed treatment / outcome / time  <- what to analyse
THE FACTS       deterministic profile (types, stats)       <- what the data is
```

plus a `status: confirmed` marker that says "this dataset has been reviewed and
accepted; it is ready, and nothing is running."

## 2. On-disk layout

```
{local_storage_path}/{job_id}/
  raw/                         original Kaggle files, unzipped     [provenance]
  normalized/<table>.parquet   analysis-ready data, one per table  [THE DATA]
  manifest.json                typed description of the bundle     [THE INDEX]

job record (persisted state)   confirmed inputs + profile + status [THE CONTRACT]
```

Default `local_storage_path` is `./data` (`config/settings.py:44`). The job
record persists to local JSON, or Firestore when enabled. See map section 6.3.

## 3. Identity

| field | type | note |
|---|---|---|
| `job_id` | str (8 char) | the key for everything above |
| `kaggle_url` | str | validated source URL |
| `created_at` | datetime | job creation |
| `confirmed_at` | datetime | when the user accepted the dataset |
| `status` | enum | `confirmed` (terminal for this slice) |

### 3.1 Recorded run configuration (not part of the data contract)

Recorded on the job and kept flowing through the system, but it describes how to
*run* an analysis, not the data itself.

| field | type | note |
|---|---|---|
| `orchestrator_mode` | enum | which orchestrator runs the job (today `standard` / `react`). Recorded per job so different orchestrators can be compared across experiments; extensible as new ones are added. |

Rationale: orchestrators are an active area of experimentation, so every job
records which one it used.

## 4. The data

- Format: **parquet**, one file per table in the bundle, under `normalized/`.
- Exactly **one file is marked `used`**: the primary dataframe analysis runs on.
- Multi-file **assembly** (joining shards/related tables into one analysis frame)
  is **out of scope here** and deferred. The authoritative analysis input is the
  single `used` parquet.
- The preview **may show and profile every table** in the bundle (display only:
  rows per file, plus a per-table schema). Showing more tables does not change
  what analysis consumes. There is existing multi-file infrastructure
  (`dataset_inspector` -> `state.file_profiles`, the relational profile) to build
  on rather than duplicate.

## 5. The manifest (`manifest.json`)

Already produced today (`storage/job_data.py:78-140`). Per file it records:

| field | meaning |
|---|---|
| `name` | original file name |
| `format` | csv / parquet / etc. |
| `size_bytes` | size |
| `n_rows` | row count |
| `columns` | list of column names |
| `tabular` | is it a readable table |
| `used` | is this the primary dataframe |
| normalized path | where the parquet lives |
| hash | content hash |

It also carries the full raw Kaggle metadata dict, so the metadata block
survives even if other artifacts are evicted.

## 6. Confirmed inputs (validated at confirm)

These come from the **user**, not from a guess. Validation happens at the moment
of confirm, against the columns of the `used` file.

| field | type | rule enforced at confirm |
|---|---|---|
| `treatment_variable` | str | **must** be a column of the used file |
| `outcome_variable` | str | **must** be a column of the used file |
| `has_time_dimension` | bool | user-confirmed in the preview |
| `time_column` | str \| null | if `has_time_dimension`, **must** be set and be a real column; else null |
| `ignored_columns` | list[str] | columns to drop from analysis (e.g. a row-index like `Unnamed: 0`); each **must** be a real column and **must not** be treatment / outcome / time; default `[]` |
| `user_context` | str \| null | optional free prose; passed through verbatim |

`ignored_columns` is **deterministically proposed, then human-confirmed** (see
section 7.1): the profiler pre-selects obvious non-features and the analyst
confirms or edits at the gate. No LLM decides what to drop.

A confirm that violates any rule is rejected and the user fixes it in the
preview (the fix-in-preview behavior we agreed on). The record is never stored in
a broken state.

### 6.1 Two context channels (kept separate, never merged)

Context reaches the analysis from two distinct sources, each carried as its own
field with a present/absent signal:

| channel | field(s) | provenance | when absent |
|---|---|---|---|
| analyst | `user_context` (this section) | what the human asserts / wants analysed; authoritative | optional, may be empty |
| source | `kaggle_description`, `column_descriptions` (manifest, section 5) | Kaggle's own description; informative, not authoritative | common (LaLonde: "none provided by source") |

They are never concatenated into one blob: the analysis must be able to weigh the
human's intent against the source's claim, and to know when either is missing. If
**both** are empty, that is a low-context / thin signal the analysis surfaces
rather than papers over.

## 7. The deterministic profile (facts only, no LLM)

Computed by the deterministic structural profiler **before** the gate (the
keystone change). Pure pandas/numpy; fully reproducible from the data.

| field | type | meaning |
|---|---|---|
| `n_samples` | int | row count of the used file |
| `n_features` | int | column count |
| `feature_names` | list[str] | columns of the used file |
| `feature_types` | dict[str,str] | per column: binary / ordinal / numeric / datetime / categorical / text |
| `missing_values` | dict[str,int] | per column NaN count |
| `numeric_stats` | dict[str,dict] | per numeric column: mean, std, min, max, median |
| `categorical_stats` | dict[str,dict] | per categorical column: top value counts |
| `has_time_dimension` | bool | deterministic time detection (then user-confirmed, section 6) |
| `time_column` | str \| null | the detected/confirmed time column |

**Intentionally excluded from the stored contract:** the LLM/heuristic
role-candidate lists (`treatment_candidates`, `outcome_candidates`,
`potential_confounders`, `potential_instruments`, `discontinuity_candidates`).
Roles come from the user, so storing a machine guess alongside them is the
unnecessary fallback we are removing. (They may still be computed live as
non-authoritative UI hints; see open decision D2.)

### 7.1 Index / non-feature detection (proposes, does not decide)

The profiler flags likely non-features as **ignore-candidates**, purely
deterministically and reproducibly: a column named `Unnamed: *`, a column whose
values are a perfect `0..n-1` sequence, or an all-unique monotonic integer id.
These are **proposals only**: they pre-check the `ignored_columns` control at the
gate. The human confirms or overrides; the confirmed set is what locks. This is
the keystone fix for the row-index leak (`Unnamed: 0` entering balance and
confounders as a phantom variable).

## 8. Lifecycle

```
PENDING -> FETCHING_DATA -> (download + deterministic profile) -> AWAITING_CONFIRM
   -> user confirms  ->  CONFIRMED   == END OF SLICE
```

`CONFIRMED` is terminal for this slice. The slice **does not start analysis.**
This replaces today's behavior where approving the data-review gate resumes
straight into the pipeline (`manager.py:281-289`). Analysis becomes a separate,
explicit trigger that reads a `CONFIRMED` record.

## 9. Invariants an analysis may rely on

Given a `CONFIRMED` record, analysis can assume, without re-checking:

1. The `used` parquet exists at the manifest's normalized path and is readable.
2. `treatment_variable` and `outcome_variable` are real columns of the used file.
3. If `has_time_dimension`, `time_column` is a real column of the used file.
4. `profile.feature_names` equals the used file's columns.
5. Everything here is reproducible from the data alone; no LLM ran to produce it.
6. `ignored_columns` are real columns of the used file, disjoint from treatment /
   outcome / time. The analysis frame is the used table with `ignored_columns`
   dropped; the effective feature set is `feature_names` minus `ignored_columns`.

If any invariant cannot hold, the dataset never reaches `CONFIRMED`.

---

## Decisions (all resolved)

- **D1, orchestrator mode. RESOLVED: keep it, recorded on the job.**
  `orchestrator_mode` is recorded per job and kept flowing (section 3.1), because
  orchestrators are being actively experimented with. It is a run-config field,
  separate from the data-describing contract.
- **D2, role hints. RESOLVED: UI-only, never stored.** Any deterministic
  column-role suggestion is a live UI convenience at most; nothing of the kind is
  persisted. The stored contract stays facts plus user choices, no machine
  guesses.
- **D3, status name. RESOLVED: yes, explicit status.** Introduce an explicit
  `CONFIRMED` terminal status (with `AWAITING_CONFIRM` before it), distinct from
  the existing approval states, so analysis can query for ready datasets cleanly.
- **D4, where it lives. RESOLVED: reuse what exists.** Store the confirmed inputs
  + profile by extending the existing `manifest.json` and job record. No new
  storage mechanism.
- **D5, columns to ignore. RESOLVED: deterministic proposal, human lock.** A new
  `ignored_columns` field (section 6). The profiler proposes obvious non-features
  (section 7.1); the human confirms at the gate; the set is validated as real
  columns disjoint from T/Y/time, and stored. No LLM decides what to drop, holding
  the no-guessing line. Closes the `Unnamed: 0` row-index leak.
- **D6, two context channels. RESOLVED: separate, never merged.** `user_context`
  (analyst) and the source's `kaggle_description` / `column_descriptions` (in the
  manifest) are distinct fields with presence flags (section 6.1). Both empty is a
  low-context signal, not a prompt to guess.

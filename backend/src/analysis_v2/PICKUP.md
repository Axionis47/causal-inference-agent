# Pickup: where the analysis reads its data from

This doc answers one question only: **when the "run analysis" button is hit, where
is every piece of data the analysis can use?** It is a map of locations, not a
plan. What to use, and how, is decided later, agent by agent.

Pairs with `docs/input-slice/01-files-and-flow.md` (the flow that produces this
data).

---

## 1. The trigger and the pickup point

```
POST /jobs/{id}/run
  -> backend/src/jobs/manager.py :: JobManager.run_analysis(job_id)
       state = await self.firestore.load_parked_state(job_id)   # <- PICK UP HERE
```

`run_analysis` is currently a stub: it loads the CONFIRMED record and stops
(`{"resumed": false, "status": "confirmed"}`). The analysis rebuild replaces the
stub body with a hand-off to `analysis_v2/runner`, passing this `state`.

The object you pick up is one `AnalysisState`
(`backend/src/analysis_v2/state/state.py`). It is the durable, reloadable
confirmed record. Everything below is a field on it.

---

## 2. Where each piece lives (on the picked-up `state`)

| data | location | notes |
|---|---|---|
| job id | `state.job_id` | 8-char key |
| source url | `state.dataset_info.url` | the Kaggle dataset |
| status | `state.status` | `confirmed` on entry |
| created at | `state.created_at` | |
| approval record | `state.human_approval` | set at confirm (APPROVED) |
| run-config | `state.orchestrator_mode` | recorded, not data |

### The question (the human's input)
| data | location | rule |
|---|---|---|
| causal question | `state.causal_question` | required, free text; what the human wants to know ("Does job training increase income?"). Validated non-empty at submit and confirm |
| ignored columns | `state.ignored_columns` | columns to drop (e.g. `Unnamed: 0`); real |
| time dimension | `state.data_profile.has_time_dimension` | |
| time column | `state.data_profile.time_column` | real column when has_time_dimension |

The human no longer names treatment and outcome. `state.treatment_variable` and
`state.outcome_variable` are nullable **derived-later slots**: empty at pickup,
for the analysis to fill once it has read the question and the data. Do not
expect the human to have set them.

The first thing the analysis does with `state.causal_question` is recognize
which **kind** of causal question it is (a plain X to Y effect, a
difference-in-differences / policy effect, a regression-discontinuity cutoff, an
instrumental-variable question, mediation, survival, before-after, driver
discovery, and so on). That recognition is what routes the identification
strategy and, with it, which columns become treatment / outcome / running
variable / instrument / mediator. How that recognition and routing work is
decided when the first analysis agent is designed; this doc only fixes where the
question is read from.

### The facts (deterministic profile, no LLM)
| data | location |
|---|---|
| the whole profile | `state.data_profile` (`DataProfile`) |
| rows / cols | `state.data_profile.n_samples` / `n_features` |
| columns | `state.data_profile.feature_names` |
| column types | `state.data_profile.feature_types` |
| missingness | `state.data_profile.missing_values` |
| numeric stats | `state.data_profile.numeric_stats` |
| categorical stats | `state.data_profile.categorical_stats` |

### The data on disk
| data | location |
|---|---|
| loaded frame path | `state.dataframe_path` |
| file bundle | `state.dataset_info.files` (`FileEntry`: name, size_bytes, format, used) |
| used file marker | the `FileEntry` with `used == True` |
| local path | `state.dataset_info.local_path` |
| on-disk root | `./data/{job_id}/` (raw + normalized parquet + manifest) |

---

## 3. The two context channels (separate tags, both passed, never merged)

There are two distinct kinds of context. They are stored under different fields
and must be handed to the analysis **separately, each labeled**. Either may be
empty. If both are empty, that is a low-context signal to surface, not guess past.

| tag | channel | location | trust |
|---|---|---|---|
| `user_context` | analyst (what the user typed) | `state.dataset_info.user_provided_context` | authoritative: what they want analysed |
| `kaggle_description` | source (Kaggle's own) | `state.dataset_info.kaggle_description` | informative, not authoritative; may be empty |

Supporting source fields (same channel, also from Kaggle):
`state.dataset_info.kaggle_subtitle`, `state.dataset_info.kaggle_column_descriptions`,
`state.dataset_info.kaggle_tags`, `state.dataset_info.kaggle_keywords`.

Rule: pass both `user_context` and `kaggle_description` to whatever first reads
meaning, each under its own tag. Do not concatenate them into one blob; the
analysis must be able to tell the human's intent from the source's claim, and to
know when either is missing.

---

## 4. Explicitly out of scope (decided later)

Not in this doc, on purpose:
- what the analysis does with any of this
- how the two context channels are weighed
- the order of agents, the methods, the gates
- what `run_analysis` returns once it actually launches

Those get decided when the first analysis agent is designed. This doc only fixes
**where to read from**.

---

## 5. The one code change when analysis is built

Swap the stub body of `JobManager.run_analysis` (`backend/src/jobs/manager.py`):
it already loads `state` from `load_parked_state(job_id)`. Hand that `state` to
`analysis_v2/runner` instead of returning the stub. Nothing upstream changes.

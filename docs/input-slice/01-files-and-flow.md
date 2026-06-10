# Files and flow: download to launch (doc one)

**Scope.** Every file in the path from "submit a Kaggle dataset" to "wait for the
analyst to launch analysis", frontend and backend. The flow is: **download ->
review the data -> take the inputs -> confirm -> wait for launch**. The launch
(`POST /jobs/{id}/run`) is the **boundary**: what happens after it is the
analysis, which is out of scope here (deleted, being rebuilt against
`docs/analysis-slice/00-start-analysis-payload.md`).

This is the kept slice. Keep it clean; nothing analysis-related belongs in these
files.

---

## 1. The flow at a glance

```
 USER                FRONTEND                         BACKEND
  |  fill form         HomePage.tsx
  |  "Run analysis" -> api.createJob ---------------> POST /jobs
  |                    navigate /jobs/:id              manager.create_job
  |                                                      spawns worker
  |                                                    manager._download_and_gate:
  |                                                      download/  (pull Kaggle)
  |                                                      storage/normalize (parquet)
  |                                                      storage/job_data (manifest)
  |                                                      <profiler> deterministic facts
  |                                                      park at data-review gate
  |                    JobPage.tsx                        |  emits SSE + persists
  |   sees live   <-   useJob (SSE stream)  <----------- GET /jobs/:id/stream
  |   data        <-   useDatasetView (poll) <---------- GET /jobs/:id/dataset
  |                    DatasetView.tsx                    (api/utils/dataset_view)
  |   review:          SampleRowsView <---------------- GET /jobs/:id/dataset/rows
  |                    SchemaBlock / RelationalBlock
  |  set T/Y/time ->   InputsBlock -> api.updateInputs-> PATCH /jobs/:id/inputs
  |                                                      manager.set_dataset_inputs
  |  confirm      ->   ApprovalBar -> api.confirm -----> POST /jobs/:id/confirm
  |                                                      manager.confirm_dataset
  |                                                      status -> CONFIRMED  (slice ends)
  |  launch       ->   RunAnalysisBar -> api.run ------> POST /jobs/:id/run
  |                                                      manager.run_analysis  == BOUNDARY
  v                                                      (hands off to analysis)
```

---

## 2. Backend files

| file | role in the flow |
|---|---|
| `api/main.py` | FastAPI app, CORS, mounts the routers |
| `api/routes/jobs.py` | the endpoints (section 4). create / dataset / rows / inputs / confirm / run / stream |
| `api/schemas/job.py` | request + response models (`CreateJobRequest`, `UpdateInputsRequest`, dataset/confirm/run responses) |
| `api/utils/dataset_view.py` | builds the `GET /dataset` payload: download block + deterministic profile + kaggle metadata |
| `jobs/manager.py` | the orchestration: `create_job` -> `_download_and_gate` -> `set_dataset_inputs` -> `confirm_dataset` -> `run_analysis` |
| `jobs/gate_resume.py` | rehydrate the data-review gate snapshot after a refresh / SSE drop |
| `download/` | Kaggle acquisition: `client`, `pull`, `auth`, `url`, `transport`, `storage`, `download_store`, `profile_store`, `events`, `api` |
| `storage/normalize.py` | normalize raw files to one parquet per table |
| `storage/job_data.py` | write `manifest.json` (typed index of the bundle) |
| `storage/local_storage.py` / `firestore.py` | persist the job record + parked state (local JSON or Firestore) |
| `storage/serialize.py`, `storage/cleanup.py` | (de)serialize state; evict local artifacts |
| `domain/` | the state types: `job.py` (Job + status), `download.py`, `dataset_manifest.py`, `approval.py`, `relational.py` |
| `config/settings.py` | `local_storage_path` (default `./data`), CORS origins, instance id |

On-disk per job (local-first, the checkpoint): `./data/{job_id}/` with `raw/`,
`normalized/<table>.parquet`, `manifest.json`; job record + parked state in
`jobs.json` / `parked_states.json`.

---

## 3. Frontend files

| file | role in the flow |
|---|---|
| `pages/HomePage.tsx` | the submit form: Kaggle URL + treatment + outcome + context; calls `createJob`, navigates to `/jobs/:id` |
| `services/api.ts` | the HTTP client + types: `createJob`, `getJob`, `getDatasetView`, `getDatasetRows`, `updateDatasetInputs`, `confirmDataset`, `runAnalysis` |
| `pages/JobPage.tsx` | the job view orchestrator: drives polling + SSE, mounts `DatasetView`, renders `ApprovalBar` / `RunAnalysisBar` |
| `hooks/useJob.ts` | the SSE stream; lights up the dataset blocks live as events arrive |
| `hooks/useDatasetView.ts` | polls `GET /jobs/:id/dataset` until every block settles |
| `hooks/useDatasetRows.ts` | fetches raw sample rows on demand |
| `store/jobStore.ts` | the dataset-view store (`patchDownload` / `patchProfile` / `patchKaggleMeta`) |
| `components/job/terminal/DatasetView.tsx` | the data-review surface; composes the blocks below |
| `components/job/terminal/SampleRowsView.tsx` | raw rows |
| `components/job/terminal/SchemaBlock.tsx` | deterministic schema: column types, missingness, time tag |
| `components/job/terminal/InputsBlock.tsx` | treatment / outcome / time selectors; existence check; `updateDatasetInputs` |
| `components/job/terminal/RelationalBlock.tsx` | bundle structure for multi-file datasets |
| `components/job/terminal/ApprovalBar.tsx` | confirm dataset / reject (gated on valid inputs) |
| `components/job/terminal/RunAnalysisBar.tsx` | the launch control (`runAnalysis`) == boundary |
| `components/job/terminal/{atoms,format}.ts(x)`, `resolveGate.ts` | shared UI atoms, status formatting, gate-snapshot resolution |
| `App.tsx`, `components/common/Header.tsx`, `config/constants.ts`, `types/index.ts`, `utils/index.ts` | router, app header, constants, shared types/util |

**Out of scope (analysis view, not part of this slice):** `Tape`, `AgentsRail`,
`FocusPane`, `PhaseStrip`, `DagGate`, `DagSvg`, `BalancePlot`, `ForestPlot`,
`ResultsGate`, `ResultsView`, `TraceSteps`, `traceEvents`, `deriveJobView`,
`describe`, `preview`, and `hooks/useResults`. These render analysis output and
belong to the analysis rebuild, not here.

---

## 4. The wire contract

REST (frontend base `/api` -> vite proxy -> backend):

| method + path | handler | purpose |
|---|---|---|
| `POST /jobs` | `create_job` | create + start download (`CreateJobRequest`) |
| `GET /jobs/{id}` | `get_job` | poll job status |
| `GET /jobs/{id}/dataset` | `get_dataset_view` | the review payload (download + profile + metadata) |
| `GET /jobs/{id}/dataset/rows` | `get_dataset_rows` | raw sample rows |
| `PATCH /jobs/{id}/inputs` | `update_dataset_inputs` | set treatment / outcome / time |
| `POST /jobs/{id}/confirm` | `confirm_dataset` | accept the dataset; status -> CONFIRMED |
| `POST /jobs/{id}/run` | `run_analysis` | **launch (boundary)** |
| `GET /jobs/{id}/stream` | `stream_job_status` | SSE for live blocks |
| `GET /jobs/{id}/approval` | `get_approval_snapshot` | rehydrate the gate after refresh |

SSE events the data-review surface consumes (names are a contract, see
`CLAUDE.md` section 5): `dataset_download_started`, `dataset_download_complete`,
`dataset_load_failed`, `dataset_metadata_started`, `dataset_metadata_ready`,
`dataset_metadata_failed`, `data_profile_ready`.

---

## 5. Rebuild gaps on this branch (refactor/strip-analysis-backend)

The analysis package was deleted, so the shell here does not import yet. Two
things must be rebuilt for this slice to run, neither of them analysis:

1. **Minimal state model in `domain/`.** `AnalysisState`, `JobStatus`,
   `DatasetInfo`, `DataProfile`, `FileEntry` lived in the deleted package; the
   shell still imports them. Rebuild them trimmed to the download/gate stages
   (no analysis statuses), carrying the finalized `ignored_columns` and the two
   context channels (`user_context` + `kaggle_description`).
2. **Re-home the deterministic profiler.** `compute_deterministic_profile` /
   `detect_time_column` lived in the deleted `agents/data_profiler/helpers.py`.
   They are pure pandas/numpy (no LLM) and belong to this slice; move them into a
   kept module (a small profiling unit, or under `download/`).

After those two, `manager.run_analysis` stays as the launch stub (records the
start-analysis payload and stops) until the analysis slice is built.

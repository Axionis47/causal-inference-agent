# Input slice: file ledger and clean-code log

This is the living record for one section of the app: everything from the
landing page to the moment a user confirms a dataset and we persist it. The
rule for this section is strict: no dead code, no unnecessary fallbacks, one
clean tested path each. Every file is marked KEEP, DELETE, REVIVE, or REVIEW.

## Scope

The section **ends at confirm-and-store.** Running the analysis pipeline is a
separate concern and is not part of this section.

```
Input page  (Kaggle URL + treatment + outcome; read-only "Kaggle ok / LLM ok" status)
   -> Download dataset
   -> Preview  (schema + sample rows + stats;
                validate treatment/outcome against real columns, fix-in-preview;
                detect time column, user confirms -> sets has_time_dimension tag)
   -> Confirm  -> persist { dataset + inputs (treatment, outcome, time tag) }  -> END
```

Notes that shape the design:
- Credentials are **not** entered in the UI. They live in the backend `.env`.
  The UI only shows read-only "is Kaggle working / is the LLM working" status.
- Treatment and outcome are **user-given**, never guessed. The profiler's job
  is to validate them against real columns, not to choose them.
- The time column is **detected deterministically, then confirmed by the user**
  in the preview (option b). Confirming it sets the `has_time_dimension` tag,
  which is carried downstream on the shared state. The tag means "eligible for
  the time-series path," not "force a time-series analysis."

## Frontend

| File | Verdict | Note |
|---|---|---|
| `pages/HomePage.tsx` | KEEP | the input page; tested by `HomePage.test.tsx`. Will gain the read-only status indicator. |
| `services/api.ts` | KEEP | transport: `createJob`, `getDatasetView`, `getDatasetRows`. Will add a status client. |
| `store/jobStore.ts` | KEEP | shared client state (currentJob, datasetView, jobs). Creation action removed (step 1). |
| `hooks/useDatasetView.ts` | KEEP | polls `/dataset` for the preview. |
| `hooks/useDatasetRows.ts` | KEEP | paged sample rows. |
| `components/job/terminal/DatasetView.tsx` | KEEP | preview window root. Will gain the schema block + confirm action. |
| `components/job/terminal/SampleRowsView.tsx` | KEEP | sample-rows table. |
| `components/job/terminal/RelationalBlock.tsx` | KEEP | multi-file bundle view. |
| `hooks/useCreateJob.ts` | DELETED | dead duplicate 3-field creation path. Removed in step 1. |
| `store/jobStore.ts` createJob action + `isCreating` | DELETED | only the dead hook used them. Removed in step 1. |
| `components/job/terminal/SchemaBlock.tsx` | KEEP (new) | terminal-styled schema renderer for the preview (added in 1c-frontend). |
| `components/job/terminal/InputsBlock.tsx` | KEEP (new) | preview block: editable treatment/outcome/time selectors validated against real columns, persisted via PATCH (2a/2b). |
| `components/job/terminal/ApprovalBar.tsx` | KEEP / CHANGED | data gate: now "confirm dataset" (POST /confirm) + reject; notes field dropped (4c). |
| `components/job/terminal/RunAnalysisBar.tsx` | KEEP (new) | shown on a CONFIRMED job; launches the pipeline via POST /run (4c). |
| `components/dataset/SchemaSection.tsx` | DELETED | journal-themed orphan; replaced by `SchemaBlock`. Removed in 1c-frontend. |
| `components/dataset/useExpandable.ts` | DELETED | served only the deleted SchemaSection. Removed in 1c-frontend. |
| `store` action `fetchJobs` | KEEP | used by `hooks/useJobsList.ts`. |
| `store` actions `fetchResults`/`fetchTraces` | KEEP | used by `hooks/useResults.ts`. |
| `store` action `cancelJob` | DELETED | dead; pages call `api.cancelJob` via react-query, not the store. Removed in step A. |

Out of this section (results area; queued for the results pass, not touched here):
`components/results/ResultsDisplay.tsx` (no live importer) and
`components/results/CausalGraphView.tsx` (imported only by the dead ResultsDisplay).
Confirmed dead in step A: zero live importers, no test files. Deleting them cleanly
means also removing whatever only they import, which belongs to the results pass.

## Backend

| File | Verdict | Note |
|---|---|---|
| `api/routes/jobs.py` (`create_job`) | KEEP / WILL CHANGE | today it auto-starts the pipeline; the section ends at confirm-and-store, so the kickoff moves out of the create path. |
| `api/schemas/job.py` (`CreateJobRequest`) | KEEP | request validation. |
| `jobs/manager.py` (`create_job`, data-review gate) | KEEP / WILL CHANGE | the existing data-review gate becomes the confirm-and-store terminal of this section. |
| `PATCH /jobs/{id}/inputs` + `set_dataset_inputs` | KEEP (new) | persist analyst input corrections at the data gate, validated against real columns (2b-backend). |
| `analysis/agents/data_profiler/*` | KEEP / WILL CHANGE | deterministic descriptive profile stays; add time/panel detection and validate user treatment/outcome against real columns. |
| `storage/*` | KEEP | where `{ data + inputs }` persist. |
| status endpoint (Kaggle + LLM health) | ADD | read-only, for the input page. |

## Shared files (used by till-download AND the rest of the app)

The till-download components we built are self-contained. They sit on four shared
infrastructure files, used across the app, not owned by this slice. The
till-download usage of each is clean; the dead code listed is cruft serving other
(now dead) areas, queued for the separate cleanup pass.

| Shared file | What till-download uses it for | Dead code still inside (not this slice's) |
|---|---|---|
| `services/api.ts` | the single axios client (one `axios.create`, base `/api`) plus the dataset / confirm / run functions and their types. Imported by all 12 till-download files. | ~12 unused exported types (`CausalGraph`, `SensitivityResult`, `DagSummary`, ...), `API_BASE_URL`, an unused `default` export |
| `store/jobStore.ts` | the `datasetView` slot + `patchDownload` / `patchKaggleMeta` / `patchProfile` (used by `useDatasetView`, `useJob`) | unused selectors `selectCurrentJob` / `selectIsJobRunning`; dead actions `fetchJobs` / `fetchResults` / `fetchTraces` (the dead-hooks chain) |
| `config/constants.ts` | poll interval + rows-page-size + request timeout (used by `SampleRowsView`, `useDatasetView`, `useJob`) | clean (the dead `RESULTS_ANIMATION_DELAY_MS` was removed) |
| `utils/index.ts` | `validateKaggleUrl` only (used by `HomePage`) | ~20 unused functions (`formatPValue`, `debounce`, `clamp`, ...); only `validateKaggleUrl` is live |

Plus the app shell (`App.tsx`, `main.tsx`) that mounts every page.

**Not shared by till-download:** `types/index.ts` is imported by **no**
till-download file. It is standalone legacy (almost all dead: legacy types + label
maps replaced by the `api.ts` types), flagged for the separate cleanup pass.

### API architecture (one client, one backend app)

- Frontend: a single axios client in `services/api.ts` (one `axios.create`, base
  `/api`), one function per endpoint, imported by 31 files. The only non-axios
  path is the live job stream, one `EventSource` in `useJob.ts` for SSE.
- Backend: one FastAPI app, three routers, `/jobs` (this slice),
  `/api/v1/download` (Kaggle profile store, not wired to the UI), and unprefixed
  health. Dev: Vite proxies `/api/*` to the backend and strips `/api`.

**Guarantee status:** the till-download *component* files (HomePage, the preview
components, the dataset hooks) are clean, everything in them is used (the build
enforces `noUnusedLocals`; knip + grep find nothing dead inside them). The four
shared files above are used by till-download but are **not** internally
dead-code-free; the residual dead code is listed here and queued.

## Fallbacks to remove (queued, each with its reason)

- **Profiler LLM role-selection (`finalize_profile`) vs deterministic
  `auto_finalize`.** Treatment and outcome are now user-given, so the LLM
  guessing of roles is an unnecessary second path. Keep the deterministic
  structural profile; drop the LLM role-guess for this section.
- **Two Kaggle credential paths** (encrypted profile store vs `settings`/env).
  The UI no longer enters keys, so we keep one path and the status check reuses
  the existing validator. Decide which path survives, remove the other.

Out of section: silent discovery degradation (GES -> PC, LiNGAM -> NOTEARS)
lives in the discovery agent; flag for that section, not this one.

## Change log

- **Step 1 (done): removed the dead duplicate job-creation path.** Deleted
  `hooks/useCreateJob.ts`; removed the `store/jobStore.ts` `createJob` action,
  the `isCreating` flag, and the now-unused `apiCreateJob` import; removed the
  barrel export in `hooks/index.ts`; deleted one dead test and one dead
  assertion in `jobStore.test.ts`. Verified: `tsc --noEmit` exits 0 (nothing in
  the frontend referenced the removed symbols); `jobStore.test.ts` (14) and
  `HomePage.test.tsx` (7) pass. The live path (HomePage -> react-query ->
  `api.createJob`, 5 fields, validated) is untouched.
- **Step A (done): finished the store dead-code audit.** Kept `fetchJobs` (used
  by `useJobsList`), `fetchResults` and `fetchTraces` (used by `useResults`).
  Removed the dead `cancelJob` store action, its `apiCancelJob` import, and its
  test, because both pages call `api.cancelJob` via react-query, not the store.
  Verified: `tsc --noEmit` exits 0; jobStore (13), JobsListPage (5),
  JobPageSelection (1) tests pass. The `components/results` orphans are confirmed
  dead and queued for the results pass.
- **Step 1a (done): deterministic profile function, no wiring yet.** Added
  `compute_deterministic_profile(df)` (facts + deterministic time detection,
  role-candidate lists left empty per the contract) and `detect_time_column(...)`
  in `data_profiler/helpers.py`, and routed `auto_finalize`'s time logic through
  the shared `detect_time_column` (one implementation, not three). Added 7 tests
  in `data_profiler/tests/test_helpers.py`. Verified: all 12 helpers tests pass,
  including the pre-existing `auto_finalize` tests (behavior preserved). No
  runtime flow changed; the gate does not call it yet (that is step 1b).
- **Step 1b (done): the gate computes the deterministic profile.** In
  `_download_and_gate` (`manager.py`), after `load_dataset`, the gate now sets
  `state.data_profile = compute_deterministic_profile(df)` before parking, so the
  facts-only profile persists with the parked state and is available to the
  preview. Updated the gate docstring to match. Added
  `test_gate_attaches_deterministic_profile`. Verified: all 24 jobs tests pass,
  including the parked-snapshot tests (which now carry a profile). The
  post-approval profiler still overwrites `data_profile` later, so the analysis
  path is unchanged.
- **Step 1c-backend (done): surface the profile + time tag.** The dataset-view
  builder (`api/utils/dataset_view.py`) already mapped `state.data_profile` into
  the profile block; with 1b it now loads at the gate. Added `has_time_dimension`
  / `time_column` to the `build_from_state` profile dict so the preview can show
  the time tag. The gate endpoint reaches this via the parked state
  (`jobs.py:261-266`). Added `test_profile_surfaces_time_tag`. Verified: 20
  dataset-view tests pass.
- **Step 1c-frontend (done): the schema renders in the preview.** Added
  `components/job/terminal/SchemaBlock.tsx` (terminal theme: column / type /
  missing %, with a time tag) and wired it into `DatasetView` as a `[ schema ]`
  section; added `has_time_dimension?` / `time_column?` to the frontend
  `DataProfileSummary` type. Deleted the orphaned journal-themed
  `components/dataset/SchemaSection.tsx`, its `useExpandable.ts`, and
  `SchemaSection.test.tsx` (the `components/dataset/` folder is now gone). Added
  `SchemaBlock.test.tsx`. Verified: `tsc --noEmit` exits 0; the full frontend
  suite (107 tests) passes.
- **Step 1 complete:** the deterministic profile is computed before the gate and
  is now visible in the preview (schema facts + time tag), with no LLM in the
  slice and no dead schema component left behind.
- **Step 2a (done): read-only input validation in the preview.** Added
  `components/job/terminal/InputsBlock.tsx` (treatment / outcome each checked
  against the real columns, plus the detected time column), wired into
  `DatasetView` as an `[ inputs ]` section; `JobPage` threads
  `job.treatment_variable` / `outcome_variable` in. Added `InputsBlock.test.tsx`.
  Verified: `tsc` exits 0; 111 frontend tests pass.
- **Step 2b-backend (done): persist input corrections at the data gate.** Added
  `UpdateInputsRequest` / `DatasetInputsResponse` schemas, a guarded
  `JobManager.set_dataset_inputs` (data gate only via `_human_approved`; validates
  treatment / outcome / time against `data_profile.feature_names`; updates the
  parked state), and `PATCH /jobs/{id}/inputs` (404 / 409 / 422 mapping). Added
  `test_set_dataset_inputs.py` (valid update, clear time, unknown column rejected,
  past-gate rejected). Verified: 28 jobs tests pass; the API app imports and the
  route is registered.
- **Step 2b-frontend (done): editable inputs in the preview.** Added
  `updateDatasetInputs` (+ `UpdateInputsBody` / `DatasetInputs` types) to the api
  client. `InputsBlock` is now interactive: treatment / outcome / time are
  dropdowns of the real columns seeded with current values; a "save inputs" button
  (enabled only when changed and valid) persists via `PATCH /jobs/:id/inputs` and
  invalidates the job + dataset-view queries so corrected values flow back; a 422
  shows inline; read-only until the schema loads. `DatasetView` threads `jobId`.
  Rewrote `InputsBlock.test.tsx` (seeding, mismatch flag, save, clear-time,
  read-only). Verified: `tsc` exits 0; 112 frontend tests pass.
- **Step 2 complete:** the preview reconciles treatment / outcome / time against
  the real columns and persists corrections at the data gate (fix-in-preview).
- **Step 4a + 4b (done): confirm and run paths (backend).** Added
  `JobStatus.CONFIRMED`. `JobManager.confirm_dataset` (data gate only via the gate
  predicates; marks the data approved + status `CONFIRMED`; keeps the parked state
  as the confirmed record; no respawn) and `JobManager.run_analysis` (reloads the
  confirmed state, clears the parked record, respawns `_run_job` past the data
  gate). Routes `POST /jobs/{id}/confirm` and `POST /jobs/{id}/run` (404 / 409 /
  422). The dataset-view endpoint renders for `CONFIRMED` jobs; progress maps
  `confirmed`. Added 5 tests in `test_confirm_dataset.py`. Verified: 33 jobs tests
  pass; the app imports with both routes. (Fork resolved: analysis stays
  launchable via `/run`.)
- **Step 4c (done): confirm/run in the UI.** Added `confirmDataset` and
  `runAnalysis` to the api client. The data-gate `ApprovalBar` now confirms (mint
  "confirm dataset" -> POST /confirm; notes field dropped) and still rejects; a
  new `RunAnalysisBar` appears on a `CONFIRMED` job to launch the pipeline (POST
  /run). Taught the status helpers about `confirmed` (label, tone, pill,
  category). Rewrote `ApprovalBar.test.tsx`; added `RunAnalysisBar.test.tsx`.
  Verified: `tsc` exits 0; 114 frontend tests pass.
- **Step 4 complete:** confirm stores the dataset + inputs and ends the input
  slice (`CONFIRMED`); analysis is launchable separately via run. The
  confirmed-dataset boundary from the spec is now real.
- **Dead-code audit (done):** verified everything we built is wired, each new
  component has exactly one live importer; each new api function, manager method,
  and route has a live caller. The audit caught one issue: I had needlessly added
  a `confirmed` case to pre-existing dead status helpers. Removed the dead chain
  entirely: `STATUS_LABELS`, `getStatusCategory` (types + utils), `getStatusColor`,
  and the now-unused `StatusCategory` type. The live status display
  (`format.ts` -> `TopBar` / `FocusPane`) carries `confirmed`. Verified: `tsc`
  exits 0; 114 frontend tests pass; re-grep shows 0 references to the removed
  helpers.
- **Results-cluster deletion (done):** removed the dead journal-era results
  components, the whole `components/results/` folder (`ResultsDisplay`,
  `CausalGraphView`, and the journal `ForestPlot`, all only reachable from the
  dead `ResultsDisplay`), plus the orphaned `components/common/Tooltip.tsx` (0
  other users). The live results path (terminal `ForestPlot` / `ResultsView` /
  `ResultsGate`) is untouched. Verified: `tsc` exits 0; 114 tests pass; 0
  references to any deleted name.
- **Backend dead-code audit, input -> download -> confirm (done):** ran vulture
  on the slice's backend modules and vetted every hit. No genuine dead code: the
  flags were all false positives, FastAPI route handlers (registered via
  decorators), Pydantic model fields / validators, and helpers (`auto_finalize`,
  `download_to_workdir`, etc.) called by the profiler / inspector agents outside
  the scanned set. What we built is wired; nothing to remove. (vulture installed
  into the venv for the scan; not added to requirements.)
- **Till-download unused-export sweep (done):** ran ts-prune, vetted by grep
  (ts-prune false-flagged live exports like `validateKaggleUrl`, so it was not
  trusted raw). Removed three confirmed-dead till-download items:
  `RESULTS_ANIMATION_DELAY_MS` (`config/constants.ts`, orphaned by the
  `ResultsDisplay` deletion), and `getJobStatus` + `DatasetSseEvent`
  (`services/api.ts`, zero references). `DatasetEventType` is kept (used by
  `useJob`'s SSE dispatch). Verified: `tsc` exits 0; 114 tests pass.
- **Correction + out-of-scope flag:** the sweep showed `useResults` and
  `useJobsList` are themselves unused (only the dead `hooks/index.ts` barrel
  references them). So the store actions `fetchJobs` / `fetchResults` /
  `fetchTraces`, which step A called "live (used by the hooks)," are in fact dead
  via dead hooks. That plus app-wide journal cruft (legacy `types/index.ts`
  types, the `utils` grab-bag, `AgentTraces` / `ActivityFeed` / `JobProgress`) is
  the results / jobs-list area, not the till-download slice. Left for a separate
  store-and-legacy cleanup pass.

# Input Slice Architecture Map

**Scope:** the `INPUT -> DOWNLOAD -> PREVIEW -> CONFIRM/STORE` slice of the
Causal Orchestrator. The landing/input form, the dataset download, the data
preview overlay, and where the data + inputs get persisted. This map **ends at
confirm-and-store** (the data-review gate). It does not document the analysis
pipeline that runs after a human approves.

Every claim below is grounded in `file:line`. Line counts are `wc -l` (raw,
including blanks/comments).

---

## 1. Overview + end-to-end flow

A user submits a Kaggle dataset URL plus the treatment and outcome column names
on the landing page. The backend creates a job, immediately downloads the
dataset (no analysis), normalises every file to parquet, writes a durable
manifest, and **parks the job at a human data-review gate**. The frontend opens
the job page, mounts the dataset overlay, hydrates it over REST, and patches it
live over SSE as each download/metadata step completes. The user reviews the raw
rows + source metadata in the overlay and either approves (which resumes the job
into the analysis pipeline) or rejects.

Key design facts, verified:

- **Credentials are backend-only.** `KAGGLE_KEY` / `KAGGLE_USERNAME` and the LLM
  key (`ANTHROPIC_API_KEY` / `CLAUDE_API_KEY`) are read from the backend env /
  settings (`backend/src/config/settings.py:64,145,171`;
  `.env.example` Kaggle + LLM blocks). The Kaggle client authenticates from env
  inside the backend (`backend/src/analysis/agents/data_profiler/loading.py:331-332`).
  The frontend never sees them.
- **No "is Kaggle / is LLM working" status endpoint exists yet.** The only health
  routes are generic: `GET /health` and `GET /ready`
  (`backend/src/api/routes/health.py:10-26`). There is no read-only credential /
  liveness status surface for Kaggle or the LLM, and the frontend renders none.
  This is a gap to fill if the read-only status UI is desired.
- **Treatment and outcome are user-given, not guessed.** They are required form
  fields (`frontend/src/pages/HomePage.tsx:34-35`), required in the request body
  (`backend/src/api/schemas/job.py:31-42,77-95`), and stored verbatim on state
  (`backend/src/jobs/manager.py:220-221`). The profiler's `treatment_candidates`
  / `outcome_candidates` are advisory only and never overwrite the user's names.
- **The time column is detected deterministically, but is NOT user-confirmed in
  this slice.** Detection is a pure keyword/dtype scan
  (`backend/src/analysis/agents/data_profiler/tools/check_time_dimension.py:18,32-60`;
  `.../helpers.py:195-204`) that sets `has_time_dimension` + `time_column` on the
  profile (`.../output.py:32-33`), carried downstream
  (`backend/src/causal/estimators/effect_estimator.py:238-240`). **However, that
  step runs inside the data_profiler which only runs AFTER approval, and the
  data-review gate UI (`ApprovalBar`) has no time-column confirmation control**
  (it offers only approve / reject + free-text notes,
  `frontend/src/components/job/terminal/ApprovalBar.tsx:36-45`). So in the
  shipped code the time tag is deterministic-then-automatic, not
  deterministic-then-user-confirmed. Flag this as a divergence from the stated
  design intent.

### Sequence diagram

```
 USER                FRONTEND                       BACKEND                         STORAGE (disk)
  |                     |                              |                                |
  | fill form           |                              |                                |
  | (url, T, Y, ctx)    |                              |                                |
  |-------------------->| HomePage submit              |                                |
  |                     | POST /jobs ----------------->| create_job route               |
  |                     |                              |  manager.create_job:           |
  |                     |                              |   build AnalysisState(T,Y,ctx) |
  |                     |                              |   create_job_if_capacity ----->| job row (firestore/local)
  |                     |<-- {id} (201) --------------|  spawn _run_job task            |
  |                     | navigate /jobs/{id}          |                                |
  |                     |                              |  _download_and_gate:           |
  |                     |                              |   FETCHING_DATA                |
  | open JobPage        |                              |   fetch_kaggle_metadata        |
  |<--------------------| mount DatasetView overlay    |    ~SSE metadata_started/ready |
  |                     | open SSE /jobs/{id}/stream   |                                |
  |                     |<== agent_event: metadata_* ==|                                |
  |                     | GET /jobs/{id}/dataset ----->|   load_dataset:                |
  |                     |   (poll, 2s)                 |    ~SSE download_started        |
  |                     |<== agent_event: download_* ==|    kaggle download + unzip --->| raw/  (unzipped files)
  |                     |                              |    normalize_bundle ---------->| normalized/*.parquet
  |                     |                              |    write_manifest ------------>| manifest.json
  |                     |<-- DatasetView (files+meta) -|                                |
  |                     | GET .../files/{name}/rows -->|   read_file_page (manifest)    |
  |                     |<-- DatasetRowsPage ----------|<------------------------------ | normalized/*.parquet
  | scroll rows,        |                              |  park_for_approval:            |
  | read metadata       |                              |   AWAITING_APPROVAL            |
  |                     |<== agent_event: approval_req=|   save_parked_state ---------->| parked state
  |                     | render ApprovalBar           |                                |
  | click "approve"     |                              |                                |
  |-------------------->| POST /jobs/{id}/approval --->|  (resume -> analysis pipeline) |
  |                     |                              |       === END OF THIS SLICE === |
```

Note the ordering nuance: in the shipped `_download_and_gate` path the
`data_profiler` agent does **not** run before parking
(`backend/src/jobs/manager.py:379-420`), so `data_profile_ready` and the profile
block stay `pending` at the data gate. The profiler (and therefore the
`data_profile_ready` event and time-dimension tag) only fires once the job
resumes after approval, which is outside this slice.

---

## 2. Frontend components

All paths under `frontend/src/`. Line counts are `wc -l`.

| Component | Path | LOC | Role in slice |
|---|---|---|---|
| HomePage | `pages/HomePage.tsx` | 202 | The input/landing form |
| JobPage | `pages/JobPage.tsx` | 324 | Mounts the preview overlay + approval bar |
| DatasetView | `components/job/terminal/DatasetView.tsx` | 218 | The preview overlay shell + metadata/download renderers |
| SampleRowsView | `components/job/terminal/SampleRowsView.tsx` | 173 | Paged raw-row table (child of DatasetView) |
| RelationalBlock | `components/job/terminal/RelationalBlock.tsx` | 61 | Multi-file bundle structure (child of DatasetView) |
| ApprovalBar | `components/job/terminal/ApprovalBar.tsx` | 124 | Confirm/reject gate action bar |
| SchemaSection | `components/dataset/SchemaSection.tsx` | 218 | **ORPHANED** schema renderer (see flag below) |

### 2.1 HomePage (`pages/HomePage.tsx`, 202 LOC)

- **What the user sees** (`:50-201`): a title block; a "Dataset URL" text input
  (`:69-76`); a two-column grid with "Treatment variable" and "Outcome variable"
  inputs (`:82-109`); an optional "Context" textarea (`:115-122`); a collapsible
  "Advanced options" toggle exposing the orchestrator mode pills
  (standard/react) (`:128-169`); an inline error alert (`:171-179`); and the
  submit button labelled "Run Causal Analysis" / "Starting Analysis..."
  (`:181-187`).
- **Props:** none (route component).
- **Local state:** `kaggleUrl`, `treatmentVar`, `outcomeVar`, `causalQuestion`,
  `orchestratorMode`, `showAdvanced`, `error` (`:15-21`).
- **Data source / out:** validates the URL with `validateKaggleUrl`
  (`utils/index.ts:184`), requires T and Y (`:33-35`), then fires the
  `createJob` mutation (`:23-27,36-42`). On success it navigates to
  `/jobs/{id}` (`:25`). This is the only write into the slice.

### 2.2 JobPage (`pages/JobPage.tsx`, 324 LOC)

The terminal-layout orchestrator. Relevant to this slice:

- **Mounts the preview:** opens `DatasetView` by default on arrival
  (`showDataset` initialised to `!isPreview`, `:101`) and renders it at
  `:300-307`, passing `view={datasetView}`, `jobId`, `relational`, `onClose`.
- **Hydrates the preview:** `const { view: datasetView } = useDatasetView(...)`
  (`:102`) drives REST hydration + polling.
- **Drives SSE:** `useJob(...)` (`:54`) opens the SSE stream that patches the
  dataset blocks live.
- **Renders the confirm gate:** when `job.status === 'awaiting_approval'` and the
  active gate is the data gate, renders `ApprovalBar` (`:309-317`); the data gate
  keeps `DatasetView` open (`:180-184`).
- **Relational profile:** derived from the `dataset_inspection_complete` agent
  event or the gate snapshot (`:162-176`) and passed to `DatasetView`.
- **Props:** none (route component, reads `jobId` from the URL, `:35`).

### 2.3 DatasetView (`components/job/terminal/DatasetView.tsx`, 218 LOC)

The preview overlay shell. It also contains two inner block renderers as local
functions.

- **What the user sees** (`:179-217`): a full-screen overlay with a `[ dataset ]`
  header bar and an `esc · close` button (`:181-189`). The body stacks four
  captioned sections: `[ raw data ]` (`SampleRowsView`), `[ metadata ]`
  (`MetadataBlockView`), `[ download ]` (`DownloadBlockView`), and conditionally
  `[ bundle structure ]` (`RelationalBlock`, only when `relational.files.length > 1`,
  `:207-212`). Esc closes the overlay (`:171-177`).
- **Props** (`:160-170`): `view: DatasetView | null`, `jobId: string | null`,
  `relational?: RelationalProfilePayload | null`, `onClose: () => void`.
- **Data source:** all from the `view` prop (the zustand `datasetView`). No
  fetching of its own except via children.

**Inner block renderers (local functions in the same file):**

- **`MetadataBlockView`** (`:80-123`): renders `view.kaggle_meta`. Shows
  `pending` / `error` / data states (`:82-86`). On data it lays out label/value
  `MetaRow`s for title, subtitle, description (heading-aware `Description`,
  `:60-78`), tags + keywords (`Chips`, `:41-55`), license, source, size,
  downloads, votes, usability, and a per-column descriptions table
  (`:94-121`). Data comes from `view.kaggle_meta.data` (a `KaggleMeta`).
- **`DownloadBlockView`** (`:125-158`): renders `view.download`. Shows a
  `StatusLine`, optional URL + error, and a files table (File / Format / Size /
  Used) over `view.download.files` (`FileEntry[]`, `:143-152`). The `used` flag
  marks the file that feeds the working dataframe.

### 2.4 SampleRowsView (`components/job/terminal/SampleRowsView.tsx`, 173 LOC)

- **What the user sees** (`:93-172`): a `StatusLine` for rows; a file-picker
  `<select>` listing every downloaded file (marking the used one and non-tabular
  ones) (`:98-109`); a "rows X-Y of N" counter (`:110-118`); prev/next page
  buttons (`:119-136`); and a scrollable, sticky-header table of the current
  page's rows (`:147-170`). Non-tabular files show a "not previewable" notice
  (`:139-143`).
- **Props** (`:32-38`): `view: DatasetViewData`, `jobId: string | null`.
- **Data source:** the file list + column names come from
  `view.download.files` (manifest-backed) (`:39,85`). The rows are fetched a page
  at a time via `useDatasetRows(jobId, fileName, offset, limit)` (`:61-66`),
  which calls `GET /jobs/{id}/dataset/files/{name}/rows`. Page size is
  `DATASET_ROWS_PAGE_SIZE = 50` (`config/constants.ts:22`).

### 2.5 RelationalBlock (`components/job/terminal/RelationalBlock.tsx`, 61 LOC)

- **What the user sees** (`:18-60`): a one-line shape hint
  (`single` / `same_schema_shards` / `multiple_files` / `unrelated`, `:8-13`), a
  note, a per-file table (File / Rows / Cols / candidate id columns), and any
  shared-schema groups. **Renders nothing for single-file bundles** (`:16`).
- **Props** (`:15`): `profile: RelationalProfilePayload | null`.
- **Data source:** the `profile` prop, threaded down from JobPage's `relational`
  (derived from the `dataset_inspection_complete` event or the data-gate
  snapshot, `JobPage.tsx:162-176`).

### 2.6 SchemaSection (`components/dataset/SchemaSection.tsx`, 218 LOC) — ORPHANED

**This is the orphaned schema renderer. It is rendered nowhere in the live app.**
A repository search finds it imported only by its own test
(`__tests__/SchemaSection.test.tsx`); no page or live component imports it (its
only collaborator `useExpandable` is likewise used only here + the test).

It is a fully-built schema table: takes a `ProfileBlock` prop (`:17-19`), derives
per-column rows with dtype + missing % + a role badge
(treatment/outcome/confounder/instrument) from
`block.data.treatment_candidates` etc. (`:30-59`), and renders a sortable table
with a headline (`:97-181`). The profile data it needs IS fetched (it lives in
`datasetView.profile`, hydrated by `/jobs/{id}/dataset` and patched by the
`data_profile_ready` SSE event), but nothing mounts this component to display it.
**The fetched profile is rendered nowhere live in this slice.** This is dead UI
plus a live-but-unrendered data path; flag for either wiring up or removal.

---

## 3. Client state

### 3.1 The DatasetView shape

The preview's entire client state is one object, `DatasetView`, defined in
`frontend/src/services/api.ts:312-316`:

```ts
export interface DatasetView {
  download: DownloadBlock;     // api.ts:293-298
  kaggle_meta: KaggleMetaBlock; // api.ts:300-304
  profile: ProfileBlock;        // api.ts:306-310
}
```

Three independent blocks, each with its own status, so the UI lights each up as
its data arrives:

```ts
// api.ts:284-310
export type BlockStatus =
  | 'pending' | 'downloading' | 'downloaded'
  | 'loaded' | 'unavailable' | 'error' | 'failed';

export interface DownloadBlock   { status: BlockStatus; url: string|null; files: FileEntry[]; error: string|null; }
export interface KaggleMetaBlock { status: BlockStatus; data: KaggleMeta|null; error: string|null; }
export interface ProfileBlock    { status: BlockStatus; data: DataProfileSummary|null; error: string|null; }
```

Supporting record types:

```ts
// api.ts:237-248  FileEntry — one downloaded file (manifest-backed)
{ name; size_bytes; format; used; columns?; n_rows?; tabular? }

// api.ts:250-268  KaggleMeta — widened at the gate to the full Kaggle payload
{ description; column_descriptions; tags; domain; metadata_quality;
  title?; subtitle?; source?; license?; keywords?; total_size?;
  download_count?; vote_count?; usability_rating? }

// api.ts:270-282  DataProfileSummary — what the profiler writes
{ n_samples; n_features; feature_types?; missing_values?;
  treatment_candidates; outcome_candidates;
  potential_confounders?; potential_instruments? }

// api.ts:319-325  DatasetRowsPage — one page of raw rows (fetched on demand)
{ columns; rows; total_rows; offset; limit }
```

The `RelationalProfilePayload` (`api.ts:372-384`) is **not** part of
`DatasetView`; it rides a separate SSE event / gate snapshot and is threaded as a
prop, not stored in the dataset slot.

### 3.2 Where it lives (zustand store)

`frontend/src/store/jobStore.ts`:

| Slot / action | Location | Purpose |
|---|---|---|
| `datasetView: DatasetView \| null` | `jobStore.ts:38` | the single state slot for the preview; `null` until first hydrate |
| `setDatasetView(view)` | `jobStore.ts:153-155` | full replace, used by REST hydration |
| `patchDownload(partial)` | `jobStore.ts:157-167` | merge into `download` block |
| `patchKaggleMeta(partial)` | `jobStore.ts:169-179` | merge into `kaggle_meta` block |
| `patchProfile(partial)` | `jobStore.ts:181-191` | merge into `profile` block |
| `emptyDatasetView()` | `jobStore.ts:84-88` | seed all three blocks to `pending` so a patch arriving before hydrate has a base |

Switching jobs nulls the slot so the panel never shows stale data
(`setCurrentJob`, `jobStore.ts:143-151`). Only `currentJobId` is persisted to
localStorage; the dataset view is **not** (`partialize`, `jobStore.ts:203-206`).

### 3.3 Hydrated (REST) vs patched (SSE)

- **Hydrate (REST):** `useDatasetView` (`hooks/useDatasetView.ts:27-61`) runs a
  React Query against `getDatasetView` (`GET /jobs/{id}/dataset`). It **polls
  every 2s** (`DATASET_VIEW_POLL_INTERVAL_MS`, `constants.ts:12`) and freezes once
  `isSettled` is true (download `downloaded|failed`, profile `loaded|error`,
  kaggle_meta `loaded|unavailable|error`, `useDatasetView.ts:18-25`). Each
  successful fetch calls `setDatasetView` (`:48-52`). The file list with columns +
  row counts only comes over REST, so polling is the primary live path, not just a
  backstop.
- **Patch (SSE):** `useJob` (`hooks/useJob.ts`) opens the SSE channel and routes
  dataset events into the same store via the patch actions (see section 5). SSE
  updates statuses in place between polls.

`SampleRowsView`'s rows are a separate read path: `useDatasetRows`
(`hooks/useDatasetRows.ts:13-34`), `GET /jobs/{id}/dataset/files/{name}/rows`,
keyed by `(jobId, fileName, offset, limit)`, `keepPreviousData` so paging does
not flicker. Rows never enter the zustand store.

---

## 4. REST API

Frontend base URL is `/api` (`api.ts:11`). In dev, Vite proxies `/api/*` to the
backend at `http://localhost:8000` and **strips the `/api` prefix** via
`rewrite: (path) => path.replace(/^\/api/, '')` (`frontend/vite.config.ts:9-15`),
so frontend `POST /api/jobs` reaches backend `POST /jobs`.

| Method + path (frontend) | Client fn | Backend handler |
|---|---|---|
| `POST /jobs` | `createJob` (`api.ts:462-465`) | `create_job` (`backend/src/api/routes/jobs.py:61-144`) |
| `GET /jobs/{id}` | `getJob` (`api.ts:467-470`) | `get_job` (`jobs.py:211-240`) |
| `GET /jobs/{id}/dataset` | `getDatasetView` (`api.ts:506-509`) | `get_dataset_view` (`jobs.py:242-269`) |
| `GET /jobs/{id}/dataset/files/{name}/rows` | `getDatasetRows` (`api.ts:511-522`) | `get_dataset_rows` (`jobs.py:272-304`) |
| `GET /jobs/{id}/stream` (SSE) | `getStreamUrl` (`api.ts:553-555`) | `stream_job_status` (`jobs.py:328-...`) |
| `POST /jobs/{id}/approval` (confirm gate) | `submitApproval` (`api.ts:524-530`) | `submit_approval` (`jobs.py:436-...`) |
| `GET /jobs/{id}/approval` (gate snapshot) | `getGateSnapshot` (`api.ts:539-547`) | `get_approval_snapshot` (`jobs.py:416-...`) |

### 4.1 `POST /jobs` — create

- **Request body** (`CreateJobRequest`, `api.ts:74-80` / backend
  `api/schemas/job.py:22-62`):
  `{ kaggle_url, treatment_variable, outcome_variable, orchestrator_mode?, user_context? }`.
  `treatment_variable` and `outcome_variable` are **required** (min_length 1,
  validated, `job.py:31-42,77-95`); `kaggle_url` is pattern-validated
  (`job.py:64-75`).
- **Headers:** optional `Idempotency-Key` and `X-API-Key` (`jobs.py:66-67`); a
  repeat key within 24h returns the original job (`jobs.py:83-99`).
- **Response** (`Job`, 201): `{ id, kaggle_url, status, created_at, updated_at }`
  (`jobs.py:124-130`). Rate-limited 10/min (`jobs.py:62`). On capacity returns
  429 (`jobs.py:132-137`).

### 4.2 `GET /jobs/{id}/dataset`

- **Params:** path `job_id`. **Response:** `DatasetViewResponse` (the three-block
  `DatasetView`). The handler prefers live in-memory state, falls back to the
  parked state when `AWAITING_APPROVAL`, then to the persisted record
  (`jobs.py:252-269`). Block assembly is in
  `backend/src/api/utils/dataset_view.py:100-247`
  (`build_from_state` / `build_from_persisted`). Rate-limited 60/min.

### 4.3 `GET /jobs/{id}/dataset/files/{name}/rows`

- **Params:** path `job_id`, `file_name`; query `offset` (>=0) and `limit`
  (1..`_MAX_ROWS_PER_PAGE`), default 50 (`jobs.py:276-282`).
- **Response:** `DatasetRowsPage` `{ columns, rows, total_rows, offset, limit }`.
  Rows are read on demand from the file's normalised parquet, resolved via the
  manifest, with path-traversal guarded so `file_name` cannot escape the job's
  normalized dir (`read_file_page`, `backend/src/storage/job_data.py:156-203`).
  404 when the job, manifest, or named file is absent (`jobs.py:291-303`).
  Rate-limited 120/min.

---

## 5. SSE contract

This is the wire contract pinned by `CLAUDE.md` section 5 and the backend test
`tests/unit/test_data_profiler_sse_events.py`. Event names and payload keys must
not change without updating the frontend in the same commit.

**Channel:** a single `EventSource` to `GET /jobs/{id}/stream`
(`getStreamUrl`, `api.ts:553-555`), opened in `useJob`
(`hooks/useJob.ts:133-137`). The backend wraps every state SSE event in an
`agent_event` named SSE message; the frontend listens on the `agent_event`
listener (`useJob.ts:158-174`) and routes by `event_type`.

**Dispatch function:** `dispatchDatasetEvent` (`useJob.ts:52-108`). The
type-guard `isDatasetEventType` (`useJob.ts:110-117`) decides whether an
`agent_event` is a dataset event (routed into the store) or a generic agent
event (appended to the local event list). All emit sites call
`state.push_sse_event(...)`.

| Event name | Backend emit (file:line) | Frontend handler (file:line) | Patches |
|---|---|---|---|
| `dataset_metadata_started` | `data_profiler/loading.py:466` | `useJob.ts:55-57` | `patchKaggleMeta({status:'pending'})` |
| `dataset_metadata_ready` | `data_profiler/loading.py:498-509` | `useJob.ts:58-71` | `patchKaggleMeta({status:'loaded', data:{description, column_descriptions, tags, domain, metadata_quality}})` |
| `dataset_metadata_failed` | `data_profiler/loading.py:511,517,523` | `useJob.ts:72-77` | `patchKaggleMeta({status:'error', error})` |
| `dataset_download_started` | `data_profiler/loading.py:346-349` | `useJob.ts:78-84` | `patchDownload({status:'downloading', url, error:null})` |
| `dataset_download_complete` | `data_profiler/loading.py:387-394` | `useJob.ts:85-91` | `patchDownload({status:'downloaded', files, error:null})` |
| `dataset_load_failed` | `data_profiler/loading.py:343,379,411` | `useJob.ts:92-97` | `patchDownload({status:'failed', error})` |
| `data_profile_ready` | `data_profiler/agent.py:171-183` | `useJob.ts:98-104` | `patchProfile({status:'loaded', data})` |

**Payload keys, verified from the emit sites:**

- `dataset_metadata_started`: `{}` (empty, `loading.py:466`).
- `dataset_metadata_ready`: `description, subtitle, column_descriptions, tags,
  keywords, domain, metadata_quality` (`loading.py:500-508`). The frontend reads
  a subset (`description, column_descriptions, tags, domain, metadata_quality`,
  `useJob.ts:62-69`).
- `dataset_metadata_failed`: `{ error }` (`loading.py:512-523`).
- `dataset_download_started`: `{ url, dataset_id }` (`loading.py:347-348`);
  frontend uses `url`.
- `dataset_download_complete`: `{ rows, columns, files }` (`loading.py:389-393`);
  frontend uses `files`.
- `dataset_load_failed`: `{ error }` (`loading.py:343` / `379` / `411`).
- `data_profile_ready`: `{ n_samples, n_features, treatment_candidates,
  outcome_candidates, potential_confounders, potential_instruments,
  feature_types, missing_values }` (`agent.py:173-182`). Note it does **not**
  carry `has_time_dimension` / `time_column`.

Two adjacent events are **not** dataset events and ride the generic
`agent_event` path: `approval_required` (the data-gate event,
`orchestrator/base.py:152`) and `dataset_inspection_complete` (carries
`relational_profile`, consumed in `JobPage.tsx:162-167`). They are part of the
slice's flow but are routed via `addAgentEvent` (`useJob.ts:170`), not the
dataset patch actions.

**Lifecycle note:** in the shipped data-review gate path the `data_profiler`
agent does not run before parking, so only the `metadata_*` and `download_*` /
`load_failed` events fire in this slice; `data_profile_ready` fires later, after
approval resumes the job.

---

## 6. Backend path

```
POST /jobs  (create_job route, jobs.py:61-144)
   └─ manager.create_job (manager.py:167-250)
        ├─ build AnalysisState (manager.py:214-226)
        ├─ firestore.create_job_if_capacity  ──> job row persisted (manager.py:229-231)
        └─ asyncio.create_task(_run_job)      (manager.py:239)
              └─ _run_job_inner (manager.py:261-377)
                    └─ if not _human_approved:  _download_and_gate (manager.py:287-289)
                         └─ _download_and_gate (manager.py:379-420)   <-- THE DATA-REVIEW GATE
```

### 6.1 The AnalysisState it builds

`create_job` constructs the initial state (`manager.py:214-226`):

```python
AnalysisState(
    job_id=job_id,
    dataset_info=DatasetInfo(url=kaggle_url, user_provided_context=user_context),
    treatment_variable=treatment_variable,   # user-given, verbatim
    outcome_variable=outcome_variable,        # user-given, verbatim
    orchestrator_mode=orchestrator_mode or self._orchestrator_mode,
    status=JobStatus.PENDING,
    created_at=now, updated_at=now,
)
```

The user's `treatment_variable` / `outcome_variable` are stored as-is; the
`user_context` is stashed on `dataset_info.user_provided_context` for the
post-approval domain-knowledge agent (`manager.py:188-192`).

### 6.2 The data-review gate (park before any analysis)

`_download_and_gate` (`manager.py:379-420`) is where download + the inputs park
before any analysis agent runs:

1. `status = FETCHING_DATA`, persist (`manager.py:396-397`).
2. If a Kaggle URL, `fetch_kaggle_metadata(state, infer_domain_label=False)`
   (`manager.py:403-404`). `infer_domain_label=False` keeps the one derived field
   out: facts only, no inference, before approval
   (`data_profiler/loading.py:462-464,486-487`).
3. `load_dataset(state)` (`manager.py:406`): downloads + unzips the Kaggle bundle,
   normalises every file to parquet, and writes the manifest
   (`loading.py:359-394`). On failure the job is marked `FAILED`
   (`manager.py:407-410`).
4. `park_for_approval(state, persist_status)` (`manager.py:412`): sets
   `AWAITING_APPROVAL`, emits the `approval_required` SSE event with the data-gate
   payload, persists (`orchestrator/base.py:128-146`). The data-gate payload
   carries `treatment_variable, outcome_variable, data_summary, files, relational`
   (`orchestrator/base.py:_build_gate_payload`, around `base.py:25-44` of that
   function).
5. `save_parked_state`, `save_traces`, `update_job` (`manager.py:413-415`), then
   return. **No analysis agent runs.** Every label-producing step lives in the
   orchestrator, which only starts after a human approves (`manager.py:281-289`).

### 6.3 Exactly where data + inputs get persisted

| Artifact | Where written | On-disk / store path |
|---|---|---|
| Job row (id, url, status, T, Y) | `firestore.create_job_if_capacity` (`manager.py:229`) | Firestore, or local JSON under `local_storage_path` (default `./data`, `config/settings.py:44`) |
| Raw Kaggle files (unzipped) | `reset_job_raw_dir` + `dataset_download_files` (`loading.py:359-360`) | `{local_storage_path}/{job_id}/raw/` (`job_data.py:30-34,48-58`) |
| Normalised parquet (per file) | `normalize_bundle` (`loading.py:366`) | `{local_storage_path}/{job_id}/normalized/*.parquet` (`job_data.py:37-43`) |
| Manifest (typed record: per-file columns, row counts, hashes, normalised paths, full Kaggle dict) | `write_manifest(build_manifest(...))` (`loading.py:386`) | `{local_storage_path}/{job_id}/manifest.json` (`job_data.py:78-140`) |
| Working dataframe snapshot | (post-approval, in profiler) `save_dataframe` (`data_profiler/agent.py:156`) | `/tmp` working dir (`storage.cleanup.CAUSAL_TEMP_DIR`); not part of this gate |
| Parked full state (the data the user reviews) | `save_parked_state` (`manager.py:413`) | parked-states store (firestore/local) |

The manifest is the durable record the Data panel and the rows endpoint both
read (`job_data.py:1-6`); it carries the full raw Kaggle metadata dict
(`build_manifest`, `job_data.py:132`) so the metadata block survives eviction.

### 6.4 Fields the deterministic data_profiler writes into `state.data_profile`

From `backend/src/analysis/agents/data_profiler/output.py:16-34` (the `DataProfile`
model stored at `AnalysisState.data_profile`). **Note: this is written by the
profiler that runs AFTER approval, so these fields are populated post-gate, not
within this slice's data-review parking.**

```
n_samples, n_features, feature_names, feature_types, missing_values,
numeric_stats, categorical_stats,
treatment_candidates, outcome_candidates,
potential_confounders, potential_instruments,
has_time_dimension (bool, default False),
time_column (str|None),
discontinuity_candidates
```

The `has_time_dimension` / `time_column` pair is the time tag carried downstream
(`output.py:32-33`; consumed at `causal/estimators/effect_estimator.py:238-240`).

---

## 7. Deterministic vs LLM (this slice only)

| Step | Det. or LLM | Evidence |
|---|---|---|
| URL + variable validation | **Deterministic** | regex / pattern checks (`utils/index.ts:184`; `api/schemas/job.py:64-95`) |
| Job creation + state build | **Deterministic** | `manager.py:167-250` |
| Kaggle metadata fetch (at gate) | **Deterministic** (API call, no inference) | `infer_domain_label=False` skips the one derived field (`loading.py:462-487`) |
| Kaggle download + unzip + normalise + manifest | **Deterministic** | `loading.py:359-394`; `job_data.py:78-140` |
| Default-file pick | **Deterministic** | `pick_default_file` (`loading.py:368`) |
| Relational / bundle-structure profile | **Deterministic** | structural, `domain/relational.py` (per `api.ts:368-371`) |
| Time-column detection | **Deterministic** | keyword + dtype scan (`tools/check_time_dimension.py:18,32-60`; `helpers.py:195-204`) |
| Row paging (preview) | **Deterministic** | parquet slice (`job_data.py:156-203`) |
| Park for human review (the gate) | **Deterministic** | `park_for_approval` (`orchestrator/base.py:128-146`) |

**No LLM call occurs in this slice as shipped.** The data-review gate
(`_download_and_gate`) deliberately runs no analysis agent before approval
(`manager.py:281-289,379-420`). The `data_profiler` is a `ReActAgent`
(`data_profiler/agent.py:44`) and its tool-driven reasoning is the first
LLM-using step, but it runs **after** the human approves at this gate, which is
outside this slice. The only LLM-shaped field in the gate's metadata fetch
(`infer_domain`) is explicitly gated off (`infer_domain_label=False`,
`loading.py:404`) so the pre-approval surface carries facts only.

---

## Appendix: file inventory (this slice)

| File | LOC |
|---|---|
| `frontend/src/pages/HomePage.tsx` | 202 |
| `frontend/src/pages/JobPage.tsx` | 324 |
| `frontend/src/components/job/terminal/DatasetView.tsx` | 218 |
| `frontend/src/components/job/terminal/SampleRowsView.tsx` | 173 |
| `frontend/src/components/job/terminal/RelationalBlock.tsx` | 61 |
| `frontend/src/components/job/terminal/ApprovalBar.tsx` | 124 |
| `frontend/src/components/dataset/SchemaSection.tsx` (orphaned) | 218 |
| `frontend/src/hooks/useJob.ts` | 255 |
| `frontend/src/hooks/useDatasetView.ts` | 61 |
| `frontend/src/hooks/useDatasetRows.ts` | 34 |
| `frontend/src/store/jobStore.ts` | 222 |
| `frontend/src/services/api.ts` | 558 |
| `frontend/src/types/index.ts` | 220 |
| `backend/src/api/routes/jobs.py` | 911 |
| `backend/src/jobs/manager.py` | 925 |
| `backend/src/api/utils/dataset_view.py` | 247 |
| `backend/src/storage/job_data.py` | 203 |
| `backend/src/analysis/agents/data_profiler/loading.py` | 541 |
| `backend/src/analysis/agents/data_profiler/output.py` | 48 |
| `backend/src/analysis/agents/data_profiler/tools/check_time_dimension.py` | 67 |
| `backend/src/api/routes/health.py` | 26 |

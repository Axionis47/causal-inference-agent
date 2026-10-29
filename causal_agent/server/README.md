# The web server

One FastAPI process: it lists and writes datasets, drives the desk graph on a persistent checkpointer, and serves
the run directories. It adds no model calls and no analysis logic; the graph judges, the server projects.

```bash
npm run build --prefix web              # once, and after page changes
uv run python -m causal_agent.server    # http://127.0.0.1:8000
```

For page work run the Vite dev server beside it: `npm run dev --prefix web` (port 5173, proxies `/api`).
Both are in `.claude/launch.json` as `api` and `web`.

## Routes

| route | what |
|---|---|
| `GET /api/datasets` | paginated ready ingestion datasets owned by the server-derived principal |
| `POST /api/datasets` | multipart CSV/title/optional original context/idempotency key; save immutable originals to PostgreSQL and private S3 without profiling or chat |
| `GET /api/datasets/{dataset_id}` | retrieve owned saved ingestion metadata |
| `GET /api/datasets/{dataset_id}/file` | download the owned original CSV |
| `GET /api/conversations/{name}` | the view: stage, phase, prompt, questions, claims, status, runs, brief, transcript, the analysis id and its journal |
| `POST /api/conversations/{name}/messages` | resume the graph with the text; 409 while busy or after the end |
| `POST /api/conversations/{name}/resume` | continue a checkpoint whose step died with the process |
| `POST /api/conversations/{name}/new-question` | a separate owned conversation on the selected dataset, with independent memory and checkpoints |
| `GET /api/runs/{id}` and `/files/{name}` | the run directory's known files; `report.md` is cut before the model thoughts unless `?raw=1` |

Everything else serves `web/dist`.

## How a conversation runs

`sessions.py` compiles the chat graph with a `SqliteSaver` at `.artifacts/web/checkpoints.sqlite`, built with the
graph's own serializer so the claim table and the run records round-trip. Each step is the CLI's drain loop run in
a worker thread: stream until the next interrupt, keep the node names as the activity line, keep the interrupt
payload as the prompt. The page polls the view every 1.5 s while the stage is `busy`. The transcript is appended to
`data/web/<name>/transcript.jsonl`; the last prompt, the thread id and the analysis id live in `meta.json`, so a reload or a
new server process shows the same conversation, and the past analyses are listed there with their ids.

Stages: `busy` (a step is running), `waiting` (the graph is interrupted), `ended` (the graph reached END),
`stale` (a checkpoint with a pending node and no worker: the process died mid-step; Resume continues it),
`error` (the step raised; Try again resumes), `new` (a dataset with no thread yet).

## Tests

`uv run pytest causal_agent/server -q`: bounded ingestion validation,
atomic conversation metadata writes; a scripted interview through run, answer, done, and a new analysis
with the interview and desk fakes and a canned pipeline; the view surviving a second server over the same
checkpoint file; the run-file guards.

Run `make test-ingestion` for real storage and HTTP tests. It creates isolated Docker PostgreSQL/S3 resources and migrates a fresh database for each case. `make test-ingestion-mutations` requires a passing baseline and verifies named assertion failures for deliberate defects. CI runs both the mutation controls and the fresh-image browser/persistence smoke.

The manager (`sessions.py`) drives the graph; the transcript on disk is `transcript.py`; the projection into the view
models is `views.py`. The schema the web's types are generated from: `python -m causal_agent.server --openapi`.

The `/new` page stops at saved ingestion. Its source of truth is [ingestion](ingestion/); there is no staged-upload/profile API or filesystem registration fallback. The homepage and sidebar read owned ingestion metadata; `/datasets/:datasetId` shows saved originals. The desk remains a separate runtime awaiting integration. See [intake](../../docs/components/02-intake.md).

## Owned profiling (Phase 3)

`profiling/` owns explicit generation and relational persistence of read-only original-version reports. `GET`/`POST /api/datasets/{id}/profile` retrieve/generate dataset and column summaries; `GET /api/datasets/{id}/profiles/{profile_id}/columns/{column_id}` retrieves one column. All use the ingestion principal and ready/owner gate. Computation lives in `profile/inspection.py`, contracts in `profile/report.py`. PostgreSQL stores reports; S3 originals and context remain unchanged. See [the component chapter](../../docs/components/03-profiling.md).

Owned conversations are created through `POST /api/datasets/{dataset_id}/conversations`. Every chat and artifact read checks ownership. The `conversations/` service publishes exploration requests, attempts, explanations and artifact metadata in PostgreSQL; registered files live in private S3. Compose runs the A2A service persistently and submits generated code to the isolated Docker worker through a private queue.

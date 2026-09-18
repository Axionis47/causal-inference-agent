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
| `GET /api/datasets` | every entry in `data/datasets.yaml` plus the web-made ones, with the conversation's stage |
| `POST /api/profile` (multipart `file`) | stage a CSV under `.artifacts/web/uploads/<id>/` and profile it; returns the columns with kinds and examples |
| `POST /api/datasets` | move the upload to `data/raw/<name>/`, write the profile, the description note, the index entry, and `data/web/<name>/meta.json`; start the conversation |
| `DELETE /api/datasets/{name}` | remove the claims, note, profile, entry, meta, the checkpoint thread, the run directories, and the raw folder when no other entry shares the file |
| `GET /api/sessions/{name}` | the view: stage, phase, prompt, questions, claims, status, runs, brief, transcript |
| `POST /api/sessions/{name}/messages` | resume the graph with the text; 409 while busy or after the end |
| `POST /api/sessions/{name}/resume` | continue a checkpoint whose step died with the process |
| `POST /api/sessions/{name}/restart` | a new thread after the conversation ended |
| `GET /api/runs/{id}` and `/files/{name}` | the run directory's known files; `report.md` is cut before the model thoughts unless `?raw=1` |

Everything else serves `web/dist`.

## How a conversation runs

`sessions.py` compiles the chat graph with a `SqliteSaver` at `.artifacts/web/checkpoints.sqlite`, built with the
graph's own serializer so the claim table and the run records round-trip. Each step is the CLI's drain loop run in
a worker thread: stream until the next interrupt, keep the node names as the activity line, keep the interrupt
payload as the prompt. The page polls the view every 1.5 s while the stage is `busy`. The transcript is appended to
`data/web/<name>/transcript.jsonl`; the last prompt and the thread id live in `meta.json`, so a reload or a new
server process shows the same conversation.

Stages: `busy` (a step is running), `waiting` (the graph is interrupted), `ended` (the graph reached END),
`stale` (a checkpoint with a pending node and no worker: the process died mid-step; Resume continues it),
`error` (the step raised; Try again resumes), `new` (a dataset with no thread yet).

## The description

The form's paragraphs and column lines become the same three-heading note the hand-written datasets use
(`context.py`), so the router can load the dataset before the interview rewrites the note from the claims. The
person's original words stay in `meta.json`.

## Tests

`uv run pytest causal_agent/server -q`: the context template against the pack loader; upload, create, list,
delete with the caches popped and a shared CSV kept; a scripted interview through run, answer, done, and restart
with the interview and desk fakes and a canned pipeline; the view surviving a second server over the same
checkpoint file; the run-file guards.

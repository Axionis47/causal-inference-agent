# The server and the page

The server is one FastAPI process that drives the desk graph on a persistent checkpointer and projects it into view models. It adds
no model calls and no analysis logic. The page shows and never decides.

## The routes

| route | what |
|---|---|
| `GET /api/datasets` | every dataset with its conversation's stage |
| `POST /api/profile` | stage an uploaded CSV and profile it |
| `POST /api/datasets` | keep the upload, write the profile and the index entry, start the conversation |
| `DELETE /api/datasets/{name}` | remove the memory, the note, the profile, the checkpoint thread, the run folders |
| `GET /api/sessions/{name}` | the `SessionView` |
| `POST /api/sessions/{name}/messages` | resume the graph with the text; 409 while busy |
| `POST /api/sessions/{name}/resume` | continue a checkpoint whose step died with the process |
| `POST /api/sessions/{name}/analyses` | a new question on the same file: a new thread over the memory as it stands, its own journal |
| `GET /api/runs/{id}`, `/files/{name}` | a run folder's known files |
| `GET /api/artifacts/{name}/{moment}/{design}/{id}/figure.png` | a drawn picture |

Everything else serves `web/dist`. The routes are in [server/app.py](../causal_agent/server/app.py); the session manager that runs
each step in a worker thread and keeps the interrupt payload as the prompt is [server/sessions.py](../causal_agent/server/sessions.py).

## One view

`SessionView` ([server/models.py:303](../causal_agent/server/models.py)) is what the page polls every 1.5 seconds while a step runs:

| field | what |
|---|---|
| `stage` | busy, waiting, ended, stale, error, new |
| `phase`, `ready`, `prompt` | before or after the run; whether the matrix says ready; the current ask with its chips |
| `claims`, `status` | every claim with its status and source; the matrix as the page draws it |
| `runs` | one `RunView` per design: the decision, the checks, the estimates, the refutations, the interpretations, the declines, the figures, the pictures drawn after it |
| `transcript` | one `Turn` per line of the conversation, each with an optional figure or picture |
| `journal`, `analysis` | the conversation's steps and its id |
| `artifacts` | the pictures drawn before any run |

## The wire is one definition

`server/models.py` declares every response. `make schema` dumps the OpenAPI schema to `web/openapi.json`; `npm run types` generates
`web/src/generated/schema.ts` from it; `web/src/types.ts` is a page of aliases over that. CI regenerates both and fails on a diff,
so a field the server renames breaks the build, not the page. The page never types a response by hand.

## The page

Vite, React, TypeScript, no UI kit. Three routes in [App.tsx](../web/src/App.tsx): `/` the datasets, `/new` upload and describe,
`/d/:name` the conversation. One shell, the datasets in a sidebar, the conversation in the middle, and a slim strip on the right
that opens the inspector.

| part | file | shows |
|---|---|---|
| the transcript | [Transcript.tsx](../web/src/components/Transcript.tsx), [Message.tsx](../web/src/components/Message.tsx) | each turn; a run figure as inline SVG, a drawn picture as an image with its facts |
| the composer | [Composer.tsx](../web/src/components/Composer.tsx), [QuestionList.tsx](../web/src/components/QuestionList.tsx) | the ask with chips; `compose.ts` quotes each answered question beside its answer so the graph still reads prose |
| the status strip | [StatusStrip.tsx](../web/src/components/StatusStrip.tsx) | one mark per claim kind, settled or open |
| the claims tab | [inspector/StatusMatrix.tsx](../web/src/components/inspector/StatusMatrix.tsx), [ClaimsTable.tsx](../web/src/components/inspector/ClaimsTable.tsx) | the matrix, claim kinds down and families across, a struck family marked; every claim with its source and sentence |
| the results tab | [inspector/RunDetail.tsx](../web/src/components/inspector/RunDetail.tsx), [RunTables.tsx](../web/src/components/inspector/RunTables.tsx) | a run's effect and interval, its figures, the estimates, checks, refutations and declines as tables |
| the files tab | [inspector/FilesView.tsx](../web/src/components/inspector/FilesView.tsx) | the run folder: the report, the design, the artifacts |
| the journal tab | [inspector/JournalView.tsx](../web/src/components/inspector/JournalView.tsx) | the conversation's steps |
| a run figure | [Figure.tsx](../web/src/components/Figure.tsx) | a `FigureSpec` drawn as SVG: bars, lines, points, density, interval, graph |
| a drawn picture | [Picture.tsx](../web/src/components/Picture.tsx) | the caption with its `artifact:` address, the image, the facts with their addresses |

What is open lives in the URL hash, so a reload keeps it. Below 960px the sidebar and the inspector open as overlays.

```bash
make dev-api
```

```bash
make dev-web
```

The API on 8000, the page on 5173 proxying `/api`. `npm run build --prefix web` makes the bundle the API serves.

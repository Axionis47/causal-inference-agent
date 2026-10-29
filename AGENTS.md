# AGENTS.md

Guidance for agents working in this repository. Paths are repository-relative; package shorthand such as `desk/` or `memory/`
means the directory under `causal_agent/`.

## Working style and workflow

- Never use em dashes in conversation or user-facing writing. Use commas, colons, parentheses or separate sentences.
- Explain changes in plain language. Connect technical details to their purpose and distinguish verified behavior from assumptions.
- Inspect `git status --short` before editing. Preserve unrelated work already in the checkout, including untracked files.
- Read [architecture](docs/architecture.md) and [gates](docs/gates.md) before changing structure or decision authority. Use the
  [component handbook](docs/components/README.md) and [source index](docs/components/source-index.md) to find the responsible code.
- Check current behavior against implementation, declarations and active contracts. [docs/README.md](docs/README.md) indexes the
  references. Archived ADRs describe earlier shapes and do not bind a change.
- Keep changes in the owning component. Update its consumers, declarations, generated outputs and documentation together.
- The [production plan](docs/production/README.md) separates current container packaging from proposed authentication, private
  workspaces, database storage, review screens and complete CSV lineage. Do not describe planned features as implemented.
- Before finishing, inspect the diff and run checks appropriate to the change. Report results and material limitations.
  Documentation-only edits need accuracy and reference checks, not a paid model run, unless runtime prompts or policy also change.

## Project rules

### No past to support

The code supports no earlier layout, name, checkpoint or artifact.

- When a shape changes, delete the old shape. Do not add a loader, branch or fallback for a file written before the change.
- Superseded analyses are deleted, not migrated: designs, threads, checkpoints and run directories. Scope cleanup to affected artifacts.
- Do not add aliases for renamed routes, classes, functions or state keys.
- Delete tests whose only purpose is proving that an old shape still loads.
- Fixture datasets under `data/` and their claims files remain test inputs. Rename a field in the fixture itself instead of mapping its old name in code.

### Causal authority and authorization

One CSV and one causal question enter a conversation. The desk prepares context with the person, who authorizes a specific
snapshot. One family reasons over that snapshot and executes its frozen designs. Discussion continues over the results.

- Models draft. `memory/ops.py::apply` gates field updates; only the person's word or an applicable data check confirms.
  Causal beliefs require the person's word. The desk alone writes the verbatim `story` claim.
- Context sufficiency comes from endpoint obligations, not a universal context score. Relevant activated endpoints must be
  admitted, explicitly excluded or unavailable with a reason. Model drafts and advisory explanations cannot grant admission.
- A run instruction authorizes the admitted context identity, dataset bytes/version and family contract. The lane validates
  these through `lane/endpoints.py`. Do not silently substitute updated context into an authorized run.
- Essential missing context returns a typed `ContextDeficiency`. The desk gathers clarification, prepares a new snapshot and
  waits for another run authorization. Lanes do not interview the person.
- Corrections and hypothetical contexts require a fresh authorized snapshot. A hypothetical copy must not overwrite durable memory.
- Diagnostics cannot establish causal assumptions, erase a supplied hidden cause or promote a proposal to confirmed knowledge.
  Hard diagnostic failures remain hard when the person reasserts an assumption.

### Reasoning ends at the freeze

- Respect the family reasoning, engine, execution and artifact import boundaries. Reasoning cannot fit target effects;
  execution cannot reopen reasoning; artifact assembly reads results.
- Freeze endpoint/context identity, ledger revision, identification certificate, target, admitted rows/support, weights,
  estimator, inference and parameters before target-effect inspection. Verify the integrity seal before fitting.
- Select and record the lead before fitting. Retain lead failure or an unidentified requested target visibly. Do not silently
  promote a successful companion estimate.
- Only an equivalent-target fallback declared before freeze may be attempted. Do not change population, target, horizon,
  clustering or inference in response to observed effects.
- Preserve declared budgets, gate attempts and bounded revisits. An essential bound overflow produces a visible limitation,
  not an undeclared loop or newly invented endpoint.
- Compare numbers only when targets, populations, contrasts, scales and aggregation/support definitions match. Preserve failed
  attempts, unavailable inference and nonstandard confidence-set shapes in results and explanations.

### The trace mirrors the graph

LangSmith must show the workflow as it ran: prompts, records, model thoughts when enabled, gates, tools, files and child processes.
`causal_agent/common/tests/test_trace_policy.py` enforces the tracing boundaries.

- Model calls go through `common/llm.py::structured` or `lane/episode.py::run_episode`; decision-model calls go through
  `common/llm.py::decide`. Do not invoke models elsewhere.
- Compile every graph with `name=`. A root run must not be named `LangGraph`.
- Pass the parent run and thread to a child process using `common/trace.py::child_env`, and enter `trace.parent()` in the child.
  A lane launched by the desk belongs under its `run` node, not in a new trace root.
- Write analysis files under design, run or visualization folders through `trace.write` so they are attached.
- Re-prompts after refusals go through `trace.gate`; every retry retains a visible reason.
- Open a span around a call. Do not use runnable `with_config` for tracing; test fakes expose bare `invoke` methods.
- A necessary caller, process starter or raw writer outside a chokepoint needs a reason beside its trace-policy allowlist entry.
  Do not use an allowlist entry to bypass required provenance.

### A real run is the bar

The scripted-model suite is the floor. A change to a judgement, gate, prompt, rung or the desk is done only after Gemini has
run through the affected path and the trace reads right. Do not add fakes or scripted tests as the acceptance check for such
changes. Keep the suite green with the smallest necessary touch, run a real analysis and read it back. The owner decides the cost.

```bash
# One lane from a retained real-source handoff
uv run python -m causal_agent.evals.lane adjustment --handoff causal_agent/families/adjustment/evals/handoffs/gov_transfers_forced.json

# Routing followed by the lane
uv run python -m causal_agent.evals.lane adjustment students "Did completing the prep course raise math scores?"

# Inspect the actual trace, including refusals and retries
make trace
make trace ARGS="--thoughts"
make trace ARGS="--last 8"
```

Choose a case that exercises the changed family and branch. A full conversation runs against `make dev-api`: upload, question,
interview, `run`, chat, `done`. Report what the model actually did, its citations, gates, stopping behavior and limitations.
If credentials, sandbox capability or another dependency blocks acceptance, report it explicitly; scripted tests do not substitute.

Analysis acceptance uses real-source data with its source and supplied study account recorded. Do not reintroduce synthetic
evaluation generators or dependent handoffs/reports as endpoint acceptance. A CSV alone does not establish causal assumptions.
Small unit inputs remain appropriate for parsing, contracts and numerical invariants. Existing fixtures under `data/` stay.
See [testing](docs/testing.md) and the [acceptance policy](docs/acceptance/context-lane-policy.md).

## Commands and environment

Run Python through `uv run` and web commands through `npm --prefix web`. Python is 3.12; CI uses Node 20.
Install locked dependencies with `uv sync --frozen` and `npm --prefix web ci`. `make` lists available targets.

| Task | Command | Scope |
|---|---|---|
| Combined checks | `LANGSMITH_TRACING=false make check` | Lint, formatting checks, imports, types, Python/web tests, schema and ladders |
| Python suite | `LANGSMITH_TRACING=false make test` | Scripted models; real numerical engines on fixtures |
| Focused tests | `LANGSMITH_TRACING=false uv run pytest causal_agent/desk -q` | Replace the package with the affected owner |
| One journey test | `LANGSMITH_TRACING=false uv run pytest causal_agent/server/tests/context/test_chat_commands.py --ingestion-integration -q` | Focused conversation coverage |
| Web tests | `make test-web` | Vitest |
| Lint and types | `make lint types` | Ruff, import contracts, ESLint, Prettier, mypy and TypeScript |
| Format in place | `make fmt` | Ruff fixes/formatting and web Prettier; review unrelated edits |
| Generate API contract | `make schema` | `web/openapi.json` and `web/src/generated/schema.ts` |
| Generate ladders | `make ladder` | `docs/ladders/<family>.md` from declarations |
| Generate diagrams | `make docfigs` | Declared trajectories under `docs/diagrams/` |
| Production web build | `make build-web` | Separate from `make check`; also runs in CI |
| API | `make dev-api` | Port 8000 by default; serves `web/dist` when built |
| Web development | `make dev-web` | Port 5173, proxying `/api` |
| Graph Studio | `make studio` | Port 2024; removes `.langgraph_api` first; exposes desk and adjustment |
| Family evaluations | `make evals FAMILY=adjustment` | Uses the family's LangSmith dataset and cases |
| Trace readback | `make trace` | Latest trace; options through `ARGS` |

Regenerate the schema after changing `causal_agent/server/models.py` or routes that affect OpenAPI. Never hand-edit generated
response types. Regenerate ladders after changes to `rungs.yaml`, `space.yaml`, `checks.yaml`, `endpoints.yaml` or the catalogues
they render. Review generated diffs; CI checks freshness. `make check` generates files but does not itself run CI's Git freshness
comparisons or the web build.

```bash
uv run python -m causal_agent.evals.dataset <family>
uv run python -m causal_agent.evals.turns
uv run python -m causal_agent.desk.handoff <dataset> --family <f> --outcome ... -o handoff.json
uv run python -m causal_agent.desk.pipeline data/memory/<name>/designs/<n>/handoff.json --json-file out.json
```

`scripts/forced_handoffs.sh` regenerates stored forced handoffs. Inspect its scope before running it.

Copy `.env.example` to `.env` for a new checkout; preserve an existing `.env`. Keep configuration definitions in
`causal_agent/common/config.py`. Gemini uses Vertex AI application-default credentials (`gcloud auth application-default login`);
the model wrapper falls back to the active gcloud project when `VERTEX_PROJECT` is unset.

- `GEMINI_MODEL`, `GEMINI_MODEL_LIGHT`, `VERTEX_PROJECT` and `VERTEX_LOCATION` configure model access.
- `INCLUDE_THOUGHTS` and `THINKING_BUDGET` are debugging knobs.
- `RUN_DIR` and `PROFILE_CACHE_DIR` relocate runtime artifacts and cached profiles.
- `EXPLORE_IMAGE` selects the mandatory Docker sandbox image. Generated scripts have no host runner or fallback.
- `DECIDER=laya` loads the local decision model on first use. `DECIDER=jev` uses `DECIDER_URL` and `DECIDER_KEY`.
- Set `LANGSMITH_TRACING=false` for local tests. Keep tracing enabled for real acceptance and trace readback.

### Test isolation

Tests never call a model. Install fakes through `set_llm(fake)` and, where needed, `set_decider(fake)`, restoring them afterward.
See `causal_agent/desk/tests/fakes.py` and each family's test fakes.

Tests never write under repository `data/`. `causal_agent/conftest.py` redirects `RUN_DIR` and `PROFILE_CACHE_DIR` to `tmp_path`.
The opt-in `no_memory_writes` fixture makes `store.save` a no-op. Redirect `store.memory_for`, `store.snapshot`, `J.open_journal`
and `config.ROOT` as appropriate; use temporary stores when testing real persistence. Stubs for `desk.pipeline.run` must accept
its actual signature, including `thread_id`.

### Containers

The current app packages frontend and backend into one image. Ingestion and reviewed chat use PostgreSQL and private S3. Persistent application volumes retain immutable analysis snapshots/results and SQLite execution checkpoints.
Read [containerization](docs/production/containerization.md) before changing deployment behavior.

```bash
make container-build
make container-up
make container-down       # keeps application volumes
make container-smoke      # disposable test volumes; no model calls
make explore-image       # required Docker image for generated scripts
```

Generated scripts run only in Docker. Compose starts a persistent A2A agent and a network-disabled worker with a private queue and enforced per-script filesystem boundary. An unavailable sandbox must remain a visible failure. Do not remove isolation, grant
privileged mode, mount the Docker socket or enable an unrestricted fallback to make a smoke check pass. The documented baseline
verifies page/API/storage; script-dependent workflows still need a verified sandbox and real-model acceptance. HTTP liveness
alone does not establish those capabilities.

## Architecture and ownership

Code paths in this section are under `causal_agent/` unless qualified otherwise. The same shorthand is used in project rules.
Import order, from bottom to top:

```text
common → profile → memory → viz → lane → explore → families → desk → server
```

`evals` sits outside the application layers. `pyproject.toml` enforces imports. Reach family implementations through
`families/registry.py`; core packages must not name a family or engine. `families/tests/test_core_names_no_family.py` enforces this.

| Owner | Responsibility |
|---|---|
| `common/` | Contracts, addresses, configuration, model wrappers and tracing |
| `profile/` | Deterministic CSV and column facts, not causal interpretation |
| `memory/` | Sourced fields, checks, fit/admission, versions, journals and artifacts |
| `viz/` | Shared visualization contracts and rendering |
| `lane/` | Graph construction, episodes, tools, ledgers, intake validation and common results |
| `explore/` | Sandboxed exploration/transformation and shared isolated diagnostic-code support |
| `families/` | Endpoint declarations, causal reasoning, engine adapters, frozen execution and artifacts |
| `desk/` | Question, interview, snapshot preparation, run authorization and cited discussion |
| `server/` | Intake, sessions, checkpoints, HTTP projections and artifact delivery |
| `web/` (repository root) | React/Vite interface showing server decisions through generated types |

### Desk and memory

`desk/graph.py` interprets all free-text turns through one semantic path. Question entry and completed review precede chat creation.
Typed commands use a client submission ID and selected revision; PostgreSQL stores accepted operations and messages. SQLite
checkpoints contain only execution references. Fast-track never authorizes a run. Do not replace semantic interpretation with
keyword routing. `desk/route.py` reuses routing functions over an ephemeral fixture-memory copy.

Memory fields use `col:<key>.<field>` and `claim:<kind>.<field>`, with value, status, source, supporting sentence and reason.
`memory/fields.yaml` is the claim catalogue; `ops.check` runs data/consistency checks and `ops.fit` builds family fit and endpoint
admissions. Required context gates readiness; optional `rests_on` interview relations alone do not block it. `memory/facts.py`
computes addressed facts without joining outcome and treatment. PostgreSQL context revisions own knowledge; `Memory` is a disposable derived view.

### Family lanes

`desk/pipeline.py` launches a lane in its own process with the admitted handoff and referenced dataset. It must not consult
mutable conversation memory to fill gaps. Shared runtime code lives in `lane/`; family engines, judgements and knowledge stay
under `families/<name>/`.

`lane/graph.py::build` builds each graph from the family's `lane/knowledge/rungs.yaml`. `families/tests/test_ladders.py` checks
its declared trajectory. `endpoints.yaml` owns endpoint obligations/capabilities; `space.yaml` owns dimensions and support limits.
Use the generated [ladders](docs/ladders/) as the detailed reference instead of copying catalogues into another specification.

A rung uses deterministic code where evidence settles it, or a bounded `lane/episode.py` episode with permitted tools and a typed,
gated record. Citations must resolve against supplied context, tool facts or earlier rungs. Uncertainty stays an explicit flag
or decline. A lane's revision ledger is separate from durable dataset memory.

The built lanes are adjustment (DoWhy), difference in differences (pyfixest) and discontinuity (rdrobust and local-randomization
tools). Additional declared families support routing without claiming an implemented engine. Add a family through its package,
contracts/declarations, probes, handoff, lane, artifacts, evaluations and tests, then register it. Do not add family-specific
branches to shared core code.

### Exploration and diagnostic data boundaries

The desk reaches the exploration agent over A2A in a separate process. Its `run_code` tool executes bounded scripts through
`explore/sandbox.py`; gated products enter `memory/artifacts.py` and numerical facts become citable material.

- User-facing exploration receives the whole selected CSV unchanged, plus explicitly supplied artifacts. It can describe correlations but cannot update context or establish causal assumptions. Transformations and family diagnostics retain their separate input restrictions.
- Transformations retain row identities. `explore/client.py::transform` rejoins any withheld outcome by row identity and writes
  a new version through `memory/versions.py`. Selection uses `Memory.switch_version` and a journal step on the person's word.
  Do not overwrite the source CSV; subsequent analysis follows the ordinary authorization path.
- Adjustment estimator-selection code receives treatment and identified adjustment variables without outcome values.
- DiD diagnostic code preserves assignment/time metadata rows and exposes outcomes and approved covariates only before the
  earliest adoption anywhere, including for never-treated and not-yet-treated units.
- Discontinuity diagnostic code receives admitted rows with score, eligibility, optional receipt/cluster and approved
  predetermined covariates. Exclude outcomes and known outcome proxies; outcome missingness must not select diagnostic rows.
- Enforce source-role and reserved-name checks so metadata aliases cannot expose forbidden values.
- Family diagnostic code reuses `explore/diagnostics.py`. Adjustment multi-option selection requires successful finite numerical
  evidence. DiD and discontinuity may use existing evidence without code; after attempting code, they must cite a successful
  numerical probe. Failed scripts stay visible and cannot support a choice. Keep the three-call budget and stop when it is
  exhausted without usable evidence.
- All generated scripts require Docker. Missing Docker or an unavailable image fails visibly. There are no native runners or unrestricted fallbacks.

### Results, storage and wire

Addresses are the evidence spine: `col:`, `claim:`, `probe:`, `check:`, `estimate:`, `refute:`/`placebo:`, `figure:`, `artifact:`,
`version:`, `step:` and `user:turn:`. Preserve design identity on result addresses. `desk/material.py`, `desk/explainer.py` and
`desk/graph.py` restrict explanations to citable material; stated numbers must match their cited source within one percent.

| Store | Repository-relative location |
|---|---|
| Reviewed dataset context | PostgreSQL `context.revisions`; `data/memory/<name>/` remains fixture/CLI input |
| Authorized chat analysis | `data/conversations/<id>/analyses/<operation>/`, immutable snapshot reference and assembled result |
| Conversation operations and messages | PostgreSQL `chat.commands`, `chat.messages`, `chat.questions` and `chat.state` |
| Derived table versions | `data/memory/<name>/versions/<v>/` |
| Lane runtime files | `.artifacts/runs/` by default |
| Desk checkpoints | `.artifacts/web/checkpoints.sqlite` |
| Profile cache | `.artifacts/profiles/` by default |
| Artifact catalogue and files | PostgreSQL `chat` and private S3 |

Use `common/config.py` for resolved paths. Distinguish dataset/table versions, memory versions, conversation/thread IDs, outer
design numbers, frozen candidate IDs, runtime run IDs and artifact IDs. See the [persistence chapter](docs/components/09-persistence.md)
for writer ownership and lifecycle.

`server/models.py` declares response contracts. The web page consumes generated types, polls the session view and displays
records. Readiness, identification, admission and method selection belong to the backend.

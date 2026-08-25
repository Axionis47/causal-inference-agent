# T-014 — CLI seven commands, runtime composition, live Kaggle adapter

Status: frozen for implementation
Owning scope: cli (500) + runtime (800) + PRD-002 §11.1; closes D-034
Depends on: T-013

## 1. Deliverables

1. `src/causal/cli/main.py` + `src/causal/cli/render.py` — the argparse boundary (SC §1.1):
   exactly `new`, `status`, `select-table`, `answer-context`, `approve-design`, `run`,
   `presentation`; `CliCommandEnvelopeV1`/`CliResultV1` models; `--format human|json`;
   `presentation` returns a typed not-yet-available failure until PRD-005 exists.
2. `src/causal/runtime/composition.py` — concrete dependency construction from environment
   configuration (PostgreSQL DSN, S3 endpoint/bucket, LangSmith project, prompts root,
   registries root): builds the intake coordinator and the design `DesignDeps`; applies
   migrations idempotently at startup; wires the tracer into the committer (fail-closed);
   startup fingerprint checks (Python version, dot presence recorded, lock hash).
3. `src/causal/runtime/kaggle_live.py` — the live `kaggle==2.2.4` adapter implementing
   `KaggleClientProtocol`; authenticates only from `~/.kaggle/kaggle.json` at construction
   (D-034); no credential parameter anywhere; never logs credentials.
4. Command semantics (SC §1.1): idempotency keys required on mutating commands; advisory lock
   per analysis (`pg_advisory_lock` session lock; contention → `analysis_busy` blocker);
   interrupt commands validate analysis id, interrupt kind, artifact id + hash, expected
   revision, schema version, idempotency key before calling the design coordinator resume;
   stale/conflicting → typed blockers; `status` reads committed indexes only.
5. Tests: CLI parsing/validation matrix (`EV-SYS-007` fixtures: duplicate key, stale revision,
   wrong interrupt, unknown command, undeclared field), JSON result golden shapes, runtime
   composition against docker Postgres/MinIO (build deps, run intake end-to-end through the
   CLI `new` with a frozen fixture client injected), live-Kaggle smoke behind
   `RUN_LIVE_KAGGLE=1` (never default).

## 2. Notes

- The CLI never imports langgraph/model/tool modules; it calls coordinator entry points only.
- `causal new` runs intake to its boundary; `causal run` drives the design stage to the next
  interrupt or terminal state and prints the exact permitted next command.
- Budgets: cli ≤ 480 of 500; runtime ≤ 520 of 800; tests ≤ 700 new lines.

## Amendment 2 — live-provider fix: version resolution (2026-08-25, pilot finding)

The first live pilot run showed Kaggle retired the `GetDatasetStatus` endpoint:
`kaggle==2.2.4`'s `api.dataset_status()` now returns HTTP 404 for every dataset, so
`capture()` fails with `fetch_failed` before download. `api.dataset_list(search=slug)`
still works and its rows carry `current_version_number`.

Fix (runtime scope only; the pin, the `KaggleClientProtocol`, and `intake/kaggle.py`
are unchanged):

1. `LiveKaggleClient.dataset_status(owner, slug)` stops calling `api.dataset_status`.
   It calls `api.dataset_list(search=slug)`, selects the row whose `ref` equals
   `f"{owner}/{slug}"` exactly, and returns
   `{"currentVersionNumber": str(row.current_version_number)}`. No match, an empty
   version, or any provider failure keeps the existing sanitized `KaggleError` path
   (class name only; message never includes provider bodies).
2. Docstring records the endpoint retirement and that the protocol method name is kept
   for the frozen capture contract.
3. Tests: unit test with a fake api (exact-ref match among decoys; no-match failure);
   the `RUN_LIVE_KAGGLE=1` smoke asserts a non-empty resolved version for a public
   dataset so a future endpoint retirement is caught live.

Budgets: runtime stays ≤ 800; tests additions ≤ 40 lines.

## Amendment 3 — typed terminal results for harness-boundary errors (2026-08-25, pilot finding, D-062)

The first live design run crashed `causal run` with a raw traceback: the intent node's
live Vertex call raised `GatewayError`, which subclasses `Exception` (not `ValueError`),
so `cli._execute`'s typed-refusal channel never saw it. `ObservabilityError` has the
same gap. This violates SC §1.1 ("no failure escapes as an untyped traceback") and §7.1
(blocker + terminal stage state).

Fix (runtime scope only; cli and design untouched — design sits at 3,398/3,400):

1. In `CausalRuntime`, every coordinator invocation (`run`, and the resume paths in
   `select_table`, `answer_context`, `approve-design`) converts terminal
   harness-boundary errors:
   - `causal.shared.gateway.GatewayError` → record the stage run terminal state
     `failed`, emit one `blocker.raised` with the error's stable code (do not
     double-emit if the gateway already emitted for this failure — verify and follow
     the T-010 event contract), and return the command's result with status `failed`
     and the stable code (CLI exit 4 via the existing map).
   - `causal.shared.tracing.ObservabilityError` → stage run `failed_observability`,
     result status `failed_observability` (exit 5), committed artifacts preserved.
2. Follow the existing failure semantics for idempotency keys and stage-run rows that
   the design-outcome `failed` path already uses; introduce no new result shape.
3. Tests: runtime composition tests with an injected gateway that raises a terminal
   `GatewayError` (assert exit code 4, stage-run state, single blocker event) and a
   tracer/committer path raising `ObservabilityError` (assert exit 5 and state
   `failed_observability`). No raw traceback may escape `cli.main` for either.

Budgets: runtime ≤ 800 total; tests additions ≤ 60 lines.

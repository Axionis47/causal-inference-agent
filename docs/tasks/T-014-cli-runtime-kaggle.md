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

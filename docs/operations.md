# Installation and operations

## Scope

This document owns application setup and operator procedures.
Run commands from the repository root unless a step states otherwise.
Commands describe the local installation.
They do not provision cloud infrastructure.

Runtime sources: [Compose](../compose.yaml), [settings](../backend/config.py), [worker](../backend/worker.py), and [CLI](../backend/cli.py).

## Prerequisites

| Component | Required baseline | Source |
| --- | --- | --- |
| Python | 3.12 | `pyproject.toml`, `.python-version` |
| Python dependency tool | uv | `uv.lock` |
| Node.js | 24.5 or later within major 24 | `web/package.json` |
| Frontend dependency tool | pnpm 11.3.0 | `web/package.json` |
| Containers | Docker with Compose | `compose.yaml` |
| Database | PostgreSQL 17 | Compose database service |
| Model access | Vertex AI project and Application Default Credentials | Worker configuration |
| Tracing | LangSmith project and key, when enabled | Worker configuration |

Use the lockfiles for exact dependency versions.
Do not substitute floating package versions during a reproducible installation.

## Configure the local installation

1. Copy `.env.example` to `.env`.
2. Set a private `DB_PASSWORD`.
3. Put the same database password in `DATABASE_URL`.
4. Set `BOOTSTRAP_EMAIL` and a password of at least 12 characters.
5. Configure Vertex AI and tracing if model work is required.
6. Create the private runtime directories.

```sh
mkdir -p .runtime/files .runtime/sandbox
```

The password in `DATABASE_URL` must be valid in a URL.
Keep `.env`, credentials, source data, and trace exports private.

Expected result: Configuration contains the required local values.
Do not print secret values into issue reports or terminal transcripts.

## Start development processes

Precondition: Configuration is complete and Docker is available.

1. Install locked dependencies.

```sh
uv sync --frozen
pnpm --dir web install --frozen-lockfile
```

2. Start PostgreSQL.

```sh
docker compose up -d db
```

3. Install the application and worker schemas.

```sh
uv run alembic upgrade head
uv run python -m backend.worker schema
```

4. Create the development account.

```sh
uv run --env-file .env python -m backend.cli user
```

5. Start each process below in a separate terminal.

```sh
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8008
```

```sh
uv run --env-file .env python -m backend.worker run
```

```sh
pnpm --dir web dev
```

6. Open [the development frontend](http://127.0.0.1:5178).

Expected result: Login, intake, and saved-record operations are available.
Model work requires valid worker credentials.
Generated Python also requires the sandbox service.
The Vite development server proxies `/api` to port 8008.

Re-running the account command for the same email changes its password and revokes existing sessions.
It retains the user identity and dataset ownership.

## Start the container application

Precondition: Docker, configuration, and private directories are available.

1. Set `LOCAL_UID` and `LOCAL_GID` in `.env` to the host user's values.

```sh
id -u
id -g
```

2. Set `VERTEX_ADC_FILE` to an existing absolute ADC JSON path.
3. Build the images.

```sh
docker compose --profile app build
```

4. Start the database.

```sh
docker compose up -d db
```

5. Install the schemas.

```sh
docker compose --profile app run --rm api .venv/bin/alembic upgrade head
docker compose --profile app run --rm api .venv/bin/python -m backend.worker schema
```

6. Create the account if this is a new environment.

```sh
docker compose --profile app run --rm api .venv/bin/python -m backend.cli user --email you@example.com
```

This command prompts for the password.
The Compose API service does not automatically receive bootstrap credentials from `.env`.

7. Start the application services.

```sh
docker compose --profile app up -d
```

8. Open [the container application](http://127.0.0.1:8088).

Expected result: Caddy serves the frontend and proxies the API under one origin.
The worker receives model and trace credentials.
The sandbox health check must pass before computation.

For intake without model work, start only the required services:

```sh
docker compose --profile app up -d db api web
```

Queued model operations wait for a worker.
Do not run host and container workers unintentionally against the same development database.

## Configuration reference

The defaults below come from current settings and Compose.
Compose forwards selected variables explicitly.
Changing a `.env` variable that Compose does not forward will not change that container's settings.

| Setting | Default or purpose | Consumer |
| --- | --- | --- |
| `DATABASE_URL` | Host database DSN; Compose constructs an internal DSN | API and worker |
| `DB_PASSWORD` | Required Compose database password | Database and internal DSNs |
| `STORAGE_DIR` | `.runtime/files`; `/storage` in containers | API and worker |
| `SANDBOX_QUEUE` | `.runtime/sandbox`; `/queue` in worker container | Worker adapter |
| `COOKIE_SECURE` | `false` locally | API |
| `ALLOWED_ORIGINS` | Local development origins | Host API |
| `APP_ORIGIN` | `http://127.0.0.1:8088` | Compose API origin configuration |
| `SESSION_HOURS` | 24 | Settings default |
| `MAX_UPLOAD_BYTES` | 10,485,760 | Settings default |
| `MAX_ROWS` | 100,000 | Settings default |
| `MAX_COLUMNS` | 200 | Settings default |
| `CONTEXT_SOURCE_BYTES` | 1,048,576 | Settings default |
| `CONTEXT_MAX_SOURCES` | 10 | Settings default |
| `CONTEXT_MAX_CHARACTERS` | 100,000 | Settings default |
| `VERTEX_PROJECT` | Required for model work | Worker |
| `VERTEX_LOCATION` | `us-central1` | Worker |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Saved operation configuration and worker |
| `MODEL_TIMEOUT_SECONDS` | 180 | Settings default |
| `VERTEX_ADC_FILE` | Host path mounted read-only | Compose worker |
| `GOOGLE_APPLICATION_CREDENTIALS` | ADC path for host; injected container path | Provider SDK |
| `LANGSMITH_TRACING` | Compose defaults to `false` | Trace SDK |
| `LANGSMITH_PROJECT` | `causal-context` | Trace SDK |
| `LANGSMITH_API_KEY` | Optional trace-service credential | Worker |
| `BOOTSTRAP_EMAIL`, `BOOTSTRAP_PASSWORD` | Account creation inputs | CLI only |
| `LOCAL_UID`, `LOCAL_GID` | `10001` unless overridden | Container file ownership |
| `OPENSSL_ARM_CAP` | `0` | Compose maps this to worker `OPENSSL_armcap` |
| `CADDY_ADDRESS` | `:80` | Container web server |

## Health and logs

1. Inspect service state.

```sh
docker compose --profile app ps
```

2. Check API database access.

```sh
curl --fail http://127.0.0.1:8088/api/health
```

3. Inspect relevant recent logs.

```sh
docker compose --profile app logs --tail 100 api worker sandbox
```

Expected result: The health endpoint returns `{"status":"ok"}`.
This checks the API and database connection only.
It does not establish model, queue, or sandbox health.
The sandbox has a separate enforcement and heartbeat check.

Use the operation's saved error, task attempts, and trace IDs for diagnosis.
Logs can contain sensitive operation details.
Review them before sharing.

## Restart and recovery

Use restart for a routine process restart:

```sh
docker compose --profile app restart api worker web
```

Keep the database volume and private files.
Do not use `docker compose down -v` as a restart command.
That command deletes the database volume.

The queue recovery jobs inspect stale worker deliveries.
Domain recovery reuses saved receipts and consumed budgets.
The API marks eligible operations retryable.
Do not edit counters, approvals, or receipts to force a retry.

## Backup and restore

This procedure documents the required consistency boundary.
A fresh backup/restore drill was not performed during this documentation update.

Precondition: Stop new submissions and let active work settle.
Use a new empty backup directory for each backup.

1. Stop services that can write application state.

```sh
docker compose --profile app stop web api worker sandbox
```

2. Create a private backup directory.

```sh
umask 077
mkdir -p .runtime/backups
mkdir .runtime/backups/manual-backup-01
```

If the final directory already exists, choose a different name before continuing.

3. Dump the database and archive the matching file store.

```sh
docker compose exec -T db pg_dump -U causal -d causal -Fc > .runtime/backups/manual-backup-01/database.dump
tar -czf .runtime/backups/manual-backup-01/files.tar.gz -C .runtime files
```

4. Confirm both commands succeeded and both files are nonempty.
5. Record the code version, image identities, migration head, and configuration version with the backup.
6. Restart the application services.

```sh
docker compose --profile app start sandbox api worker web
```

Store credentials separately under appropriate access controls.
The sandbox spool is temporary transport.
Settled execution receipts are saved in the private file store.
An in-flight backup needs a separate coordinated recovery procedure.

Restore first into an isolated installation with an empty database and empty storage directory.
Use matching application images and configuration.
Restore the database dump with `pg_restore` and extract the paired file archive.
Keep the worker stopped until inputs and historical records have been inspected.

Check login, source checksums, context history, saved artifacts, event order, and migration compatibility.
Do not use a restore drill to re-execute an approved analysis automatically.
Do not overwrite the live database to test a backup.

## Release and rollback

1. Record the current application and schema versions.
2. Complete the applicable [verification checks](verification.md).
3. Back up the database and private files together.
4. Build locked application images.
5. Apply compatible migrations explicitly.
6. Install or migrate the library-owned queue and checkpointer schemas as required.
7. Start the new application processes.
8. Check health, login, saved records, and worker recovery.

Expected result: Existing records remain readable under the declared compatibility rules.
Do not run schema creation implicitly on every API start.
Library upgrades need their own compatible schema migration procedure.

Prefer application rollback with an additive schema retained when compatibility permits.
Some Alembic downgrades drop data or reject previously consumed budgets.
Inspect the migration before a downgrade.
Historical analysis contracts remain read-only when the current worker cannot resume them safely.

## Troubleshooting

| Symptom | Check | Action |
| --- | --- | --- |
| Login returns 401 | Account details and session expiry | Sign in again or use the account CLI |
| Mutation returns 403 | Client header and configured Origin | Use the typed client and matching application origin |
| Request returns 409 | Expected revision, turn, fingerprint, or hashes | Refresh and inspect current saved state |
| Upload returns 413 or 422 | Request size, encoding, headers, and row shape | Correct the input; retain original source bytes |
| Operation remains queued | Worker process, queue schema, credentials | Restore worker availability; inspect delivery state |
| Storage returns 503 | Database and file-store permissions | Restore storage, then retry the same submission |
| Runner is unavailable | Sandbox health and runtime hash | Restore the same runtime before eligible recovery |
| Claim gate rejects a submission | Exact source, scope, qualifications, and rule | Inspect the preserved proposal and bounded feedback |
| Run is blocked or exhausted | Accepted gates and consumed allowances | Preserve the failure; do not reset counters |
| Provider returns 400 | Saved request schema and provider error | Diagnose compatibility separately from scientific rejection |
| SSE disconnects | Session, network, last sequence | Reconnect with the saved cursor |
| Trace link is absent | Tracing configuration and captured telemetry | Use local receipts; do not fabricate trace evidence |

The current Treatments provider `400 INVALID_ARGUMENT` remains unresolved.
A schema-only success would not by itself verify a complete scientific trajectory.

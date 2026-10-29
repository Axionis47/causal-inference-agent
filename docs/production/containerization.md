# Containerize the current application, keep one application boundary

The [consolidated plan](README.md) separates packaging the current app from implementing private workspaces and the new data model. This setup packages the current frontend, API, desk, profiling, memory, visualization and three causal families into one application image. PostgreSQL and private S3 storage support ingestion, the owned library, profiles and organized context. These storage services do not split the causal logic into microservices.

## Files and responsibilities

| File | Responsibility |
|---|---|
| [Dockerfile](../../Dockerfile) | Build frontend with npm lockfile and backend dependencies with uv lockfile; assemble one non-root runtime image |
| [.dockerignore](../../.dockerignore) | Allowlisted build inputs; exclude local data, output, credentials, Git metadata and caches |
| [compose.yaml](../../compose.yaml) | One application service, localhost port, durable volumes and restricted runtime filesystem |
| [docker/compose.vertex.yaml](../../docker/compose.vertex.yaml) | Optional runtime-only mount of Vertex application-default credentials |
| [docker/healthcheck.py](../../docker/healthcheck.py) | Check HTTP liveness, built frontend, A2A and enforcing worker readiness without a model call |
| [docker/smoke.py](../../docker/smoke.py) | Test the built image in a unique disposable project and verify persistence after container replacement |
| [docker/explore.Dockerfile](../../docker/explore.Dockerfile) | Pinned analysis libraries and enforcing worker, started by Compose |
| [docker/explore_worker.py](../../docker/explore_worker.py) | Persistent private-queue supervisor, fresh isolated task filesystem and bounded cleanup |

The Dockerfile uses separate build stages, not separately deployed frontend/backend containers. The final process serves the built React assets and API on port 8000. Existing lane subprocesses stay in that container. The source checkout remains at `/app` because current code derives file addresses from it. This avoids a simultaneous persistence rewrite.

The image contains no existing workspace uploads, examples, profiles, memories or conversations. Its dataset library starts empty. It also excludes evaluation/test directories. Upload a CSV through the page; run fixture-based CLI evaluations from the source checkout using the existing commands. Dependency installation includes the current Laya/Torch stack and its locked Linux dependencies, so the image can be large. Size reduction is a separate dependency decision, not an unrecorded model/backend substitution.

## Run locally

Prerequisites: Docker Engine/Desktop running Linux containers and Compose 2.24.4 or newer. From the repository root, these deployment commands require no host Python, Node or Make:

```bash
docker compose build app sandbox
docker compose up -d --wait app
docker compose exec -T app python /usr/local/bin/causal-sandbox-probe.py
```

Open `http://localhost:8000`, upload a dataset and start a conversation from its details page. Configure model credentials as below before asking the chat to explore. Set `CAUSAL_HTTP_PORT` to another host port if 8000 is already in use. The port is bound to loopback because the current application uses a development principal without a login. Dataset and conversation routes check that principal's ownership. This package is a local baseline, not a hosted multi-user release.

Compose reads recognized variable values from the local `.env` or shell, but never copies that file into the image. `compose.yaml` lists the variables passed to the application. It fixes data/run/cache paths to its mounted volumes rather than accepting arbitrary local path overrides. No model credentials are required for ingestion. Compose supplies local database and S3 credentials. The create-dataset endpoint saves originals without profiling or starting a conversation.

Stop without deleting data:

```bash
docker compose down
```

Do not add `--volumes` to the normal shutdown command unless intentionally deleting the installation's data. The smoke test uses that flag only for the unique test project and volumes it creates itself. No user data is copied from the checkout into those volumes.

## Model credentials

The app still uses operator-configured Gemini on Vertex AI. Bring-your-own-key is not included. The container does not include the gcloud CLI, so set `VERTEX_PROJECT` explicitly. For a local application-default credentials file created through the existing gcloud login workflow:

```bash
export VERTEX_PROJECT=your-project-id
export VERTEX_ADC_FILE="$HOME/.config/gcloud/application_default_credentials.json"
docker compose -f compose.yaml -f docker/compose.vertex.yaml up -d --wait app
```

The optional override mounts the selected file read-only at `/run/secrets/vertex_adc`. It does not read or bake credentials during the build. File accessibility to the runtime user must be verified on the deployment host; do not make credentials public to fix a mount problem. Cloud deployment should use the hosting platform's workload identity where supported. Real model access and analysis correctness need a separate live acceptance run.

The default local decision model is Laya and may download its checkpoint on first use. Its cache is persisted. Existing `DECIDER=jev`, URL and key configuration can be supplied deliberately; the container does not switch providers automatically. LangSmith configuration remains operator-controlled. The smoke test forces tracing off and does not call a model. The app image sets `OPENSSL_armcap=0` to use baseline ARM instructions, avoiding an observed cryptography/OpenSSL capability-probe SIGILL under Docker Desktop. An image build imports the Vertex client to catch native import failures before deployment. See [OpenSSL capability controls](https://docs.openssl.org/master/man3/OPENSSL_armcap/).

## Persistence boundaries

| Named volume | Container path | Saved products |
|---|---|---|
| `app-data` | `/app/data` | Retained fixture inputs and local authorized analysis snapshots/results |
| `app-artifacts` | `/app/.artifacts` | SQLite checkpoints, analysis run folders and profile cache |
| `model-cache` | `/home/causal/.cache` | Downloaded model assets and plotting cache |
| `ingestion-db` | `/var/lib/postgresql/data` | Owned ingestion metadata, profile facts, context revisions, conversations and exploration publication |
| `ingestion-objects` | `/data` in SeaweedFS | Private original CSV and published conversation artifact objects |
| `explore-queue` | `/queue` in app and sandbox | Private transient jobs and worker readiness, never a publication catalogue |

New volumes receive empty initialized directories from the image with the runtime user's permissions. Replacing the container preserves the volumes. Existing volumes are not silently reseeded or migrated. The runtime source filesystem is read-only; `/tmp` is temporary. Docker explains [volume persistence and initial population](https://docs.docker.com/engine/storage/volumes/).

Dataset context revisions, reviews, conversation selection, commands and messages use PostgreSQL. Registered artifact files and uploads use private objects. SQLite stores execution checkpoints, while local immutable analysis snapshots/results remain on the durable application volume. The consolidated chat introduces no additional container.

Container volumes are not backups. Until the planned backup/restore protocol is implemented, stop application writes before making a coordinated copy of both durable volumes. Preserve their consistency and retain the matching application version. Do not advertise recovery guarantees from a successful container restart alone.

## Generated-code sandbox boundary

Compose starts the persistent A2A agent with the HTTP application and a separate `sandbox` service. Build both with `make container-build`, then run `make container-up`. No host Python, Node or statistical packages are required for this deployment. Docker Engine or Docker Desktop must run Linux containers; Gemini credentials and a Vertex project are required for real model work. The optional credential override mounts ADC only into the trusted app, never the sandbox.

The worker has no network, application credentials, application volumes or Docker socket. It executes one script at a time inside a library-only filesystem root, with root-owned read-only inputs and a separate unprivileged scratch directory. Its supervisor has only the capabilities needed to enter that filesystem root, drop UID/GID and reap processes. Generated scripts have none of those capabilities. Seccomp denies sockets and process-group escape. Docker limits CPU, memory and processes; script deadlines, file limits and cancellation cleanup remain enforced.

`make container-sandbox` verifies this boundary through the running app. `/api/health` is liveness; `/api/ready` checks the enforcing worker and A2A card. Container health includes readiness. An unavailable worker or failed enforcement probe fails visibly. No privileged mode, Docker socket or unrestricted fallback is used.

The local checkout can still launch the same image as disposable Docker containers with read-only input mounts. `EXPLORE_IMAGE` selects that image. Compose sets `EXPLORE_QUEUE` and `EXPLORE_AGENT_URL` internally. See [exploration](../exploration.md) for the full limits, publication and chat-read contract.

Conversation identities and exploration publication now live in PostgreSQL. Files use the private conversation-owned S3 prefix; the dataset shelf is removed. Causal memory and checkpoints remain on the application's durable volumes. Starting a chat from dataset details connects the owned upload to the existing desk runtime.

## Verification scope

`make container-smoke` starts a uniquely named Compose project with fresh volumes and a random loopback port, then checks:

1. Build the current checkout and start the app, sandbox, PostgreSQL and private object store.
2. Upload two datasets through Chromium, return through the library, profile them and retrieve exact originals.
3. Publish explicitly marked deterministic context fixtures through the production publication function, without calling a model.
4. Browse saved column context, relationship links and exact source quotations through Chromium.
5. Replace all four containers, verify their IDs changed and retrieve identical metadata, original bytes, profiles and context.
6. Exercise ingestion cleanup without deleting a published original.

It removes only its disposable project and volumes. Context fixtures test persistence and presentation, not extraction quality. Real Gemini acceptance is separate. It also runs the enforcing sandbox probe. This smoke does not certify model credentials, real-model conversations, production authentication or backup restoration. Historical checks below describe earlier images, not extra assertions in the current smoke.

The application readiness check verifies the persistent A2A service and enforcing sandbox; it does not make a paid model request.

## Observed baseline, 22 October 2026

| Check | Observed result |
|---|---|
| `docker compose build app` | Passed on Docker Desktop, `linux/arm64`; build-only compiler dependencies were needed for a locked statistical dependency |
| Image | `causal-desk:local`, digest `sha256:1c7c4c472435384342086b32342ec7f535531311227d846f9ed7d26c98fa19d8`; Docker reported 3,554,568,815 bytes |
| Frontend build | Passed both locally and in the image build |
| Base and optional credential Compose configuration | Syntax validation passed; no actual credentials mounted or validated |
| HTTP and data smoke checks | Passed: health, built page, empty library, real CSV upload/profile, storage registration and API listing |
| Runtime boundaries | Passed: non-root user, read-only application source, no baked `.env`, no Docker socket |
| Statistical engines | Imports passed for all five libraries listed above; no estimates or model judgments executed |
| Container replacement | CSV bytes, saved profile, library entry and SQLite checkpoint file survived; not a mid-analysis resume test |
| Generated-script sandbox | Unavailable under the tested Docker security configuration: `bwrap: Can't mount proc on /newroot/proc: Operation not permitted`. The explicit sandbox refused; no unrestricted fallback ran |
| Helper code and docs | Ruff lint/format, Compose validation and local documentation-link checks passed |
| Cleanup | The disposable smoke-test container, network and named volumes were removed; normal installation volumes and repository datasets were not used |

This historical baseline verified page/API/storage only. The Compose worker and real-model chat acceptance added on 25 October are recorded in [whole-dataset exploration acceptance](../acceptance/whole-dataset-exploration.md). Linux x86 builds and hosted production behavior remain unverified locally.

## Ingestion storage deployment

The ingestion-only implementation supersedes the original upload/profile smoke journey above. Compose now adds PostgreSQL 17.6 and SeaweedFS 4.17, pinned by image digest, with separate persistent volumes. The application runs Alembic initialization and private bucket initialization before starting HTTP. The existing model sandbox restrictions remain unchanged.

Run `make container-build` then `make container-up`; open `/new` to ingest. `make container-smoke` is a developer check requiring local uv, Node and Chromium. It builds the current checkout, records the image digest, and uses a disposable project to verify the real browser upload, multipart endpoint, exact context, downloads, idempotency and replacement of all four service containers by comparing their IDs. Install Chromium first with `npm --prefix web exec -- playwright install chromium`. The current smoke additionally profiles uploads and reads deterministic saved context; it makes no model calls. `docker compose down` preserves volumes.

Run `docker compose exec app python -m causal_agent.server.ingestion cleanup` periodically (for example hourly) to remove unpublished objects older than 24 hours. A failed cleanup is retryable and never removes published objects. No cleanup scheduler is installed automatically.

The committed Compose credentials are **local development credentials** on unexposed infrastructure ports. `APP_ENV=development` enables the single configured `DEV_USER_ID`; non-development identity requests are rejected. Conversation routes now enforce ownership against that principal. Public hosting still needs real authentication, deployment secrets and restricted infrastructure access.

For local API development, provide `DATABASE_URL` (SQLAlchemy `postgresql+psycopg://...`), `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY` and `S3_SECRET_KEY`, then run `uv run python -m causal_agent.server.ingestion init`. Config lives in `common/config.py`. `UPLOAD_MAX_BYTES`, `UPLOAD_MAX_ROWS`, `UPLOAD_MAX_COLUMNS` and `UPLOAD_MAX_CELLS` control ingestion limits.

Run `make test-ingestion` with Docker available. The harness creates its own randomly named Compose project, then a separately migrated PostgreSQL database and private S3 bucket for every test. Deployment database and S3 environment values are ignored. Teardown removes only harness-owned resources. The ordinary Python suite explicitly skips these integration cases; the dedicated command and CI job require them and fail on missing Docker, empty integration selection or skipped tests.

Run `make test-ingestion-mutations` to establish a passing baseline and prove that the targeted backend and frontend defects fail their named assertions. The current inventory is defined in the two mutation scripts. Setup or teardown errors do not count as detected defects. JUnit, logs and summaries are written under `.artifacts/ingestion-tests/`; CI uploads them even on failure. The dedicated `ingestion` job runs these controls and the fresh-image browser/persistence smoke.

### Ingestion verification, 22 October 2026

- Built `causal-desk:local` on Docker Desktop Linux/arm64 with the new dependencies and initial migration.
- All 26 server tests passed with a real disposable PostgreSQL/SeaweedFS stack, including 16 ingestion tests. Anonymous S3 reads returned 403.
- All 47 frontend tests passed, including rendered form failure/retry/progress tests. Production web build, TypeScript, frontend lint, Python lint/types, all 12 import contracts and five trace-policy tests passed.
- Disposable Compose smoke passed: real multipart registration, exact CSV/context retrieval, duplicate submission, recreation of app/database/object-store containers, and safe cleanup. No model or profiling call was made.
- Existing frontend-tooling audit findings remain (Vitest/tinypool, Vite/source-map-js and mocker). A major toolchain upgrade is outside this ingestion milestone. Authentication is still development-only; no public production-readiness claim is made.


## Phase 2 owned library

The homepage and sidebar now read paginated ingestion metadata from PostgreSQL. Opening `/datasets/:datasetId` retrieves saved context automatically; downloads remain owner-checked. There is no name-based dataset deletion endpoint and no fixture-backed library fallback. The second Alembic migration adds the library index while preserving existing ingestion records.

`make test-library` runs the Phase 2 group in isolated PostgreSQL/S3. `make test-ingestion` runs both phases; `make test-ingestion-mutations` requires their baseline plus ten backend and three frontend negative controls. The existing CI storage job includes both phases. `make container-smoke` uploads two datasets through Chromium, returns through the library, and verifies metadata and bytes after replacing all service containers. Real authentication and integration with the separate analysis runtime remain deferred.

## Phase 3 dataset and column profiles

The same three containers now support explicit, read-only profiling from dataset details. Migration `0003_profiling` adds the `profiling` schema: stable column identities, dataset-level profile records and independently retrievable column results. Reports live only in PostgreSQL; source CSV bytes remain in private object storage. The older analysis filesystem stores are unchanged and not connected to new profiles.

`make test-profiling` runs calculation tests and required isolated PostgreSQL/S3 tests. `make test-storage` runs Phases 1-4 together. The shared mutation command includes four additional profiling controls. The browser/Compose smoke now profiles datasets, reads column details and verifies both report levels after replacing all three services. No model calls or saved conversions are involved.

## Phase 4 organized context

Migration `0004_context` adds saved context revisions, column context and shared relationships in PostgreSQL. Source text remains in the immutable ingestion context record; raw CSVs stay in private S3. Preparation uses one bounded extractor/assessor workflow. Saved views do not invoke a model or fetch CSV bytes.

The existing browser and replacement smoke includes context fixtures. `make test-context` runs focused extraction and real isolated storage checks. `make test-ingestion-mutations` includes targeted Phase 4 defects alongside Phases 1-3. See the [Phase 4 verification record](../audits/phase4-context.md) for observed results and outstanding acceptance, rather than treating the existence of these commands as proof they passed.

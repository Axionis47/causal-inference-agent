# Technology and deployment decisions

Causal Workspace combines a static React frontend with a Python API, a background worker, PostgreSQL, and an isolated computation runner.
The local installation runs through Docker Compose.

See [architecture](architecture.md) for module ownership and [operations](operations.md) for setup commands.

## Frontend

| Responsibility | Technology | Rationale |
| --- | --- | --- |
| Application | React 19, strict TypeScript, Vite | A static browser application with typed components and a production asset build |
| Navigation | TanStack Router | Typed routes and URL-owned workspace selection |
| Server state | TanStack Query | Shared request caching, mutations, invalidation, and status polling |
| UI | Tailwind CSS, shadcn/ui, Radix primitives, Lucide icons | Reusable controls and a consistent terminal-style interface |
| Forms | React Hook Form and Zod | Form ownership and boundary validation |
| Tables | TanStack Table | Column definitions and reusable tabular presentation |
| API client | openapi-typescript, openapi-fetch, openapi-react-query | Transport types generated from FastAPI and typed requests |
| Checks | TypeScript, ESLint, Playwright | Type checks, source checks, and connected browser journeys |
| Dependencies | pnpm and its committed lockfile | Reproducible frontend installation |

Server state belongs to Query, shareable navigation belongs to Router, and unsaved input belongs to the form or feature that owns it.
The backend owns scientific validation; browser schemas validate submitted forms and URL parameters.
Shared visual primitives express the dark palette, monospace typography, thin borders, and visible focus states.

The frontend is served as static assets. It does not require server rendering or a separate JavaScript backend.
See the [frontend manifest](../web/package.json) and [lockfile](../web/pnpm-lock.yaml) for installed versions.

## Backend and persistence

| Responsibility | Technology | Rationale |
| --- | --- | --- |
| Runtime and API | Python 3.12, FastAPI, Pydantic | Typed HTTP boundaries alongside the Python scientific ecosystem |
| Domain records | PostgreSQL, SQLAlchemy, Alembic | Transactional ownership, explicit schemas, and versioned migrations |
| Job delivery | Procrastinate | Durable PostgreSQL-backed jobs without a separate broker |
| Workflow progress | LangGraph with PostgreSQL checkpoints | Resumable chat and analysis execution |
| Model access | Google Gen AI SDK with Vertex AI | One configured provider boundary with typed outputs and native tool calls |
| Tracing | LangSmith | Correlated requests, outputs, and diagnostic traces when configured |
| Authentication | Password hashing and backend-validated sessions | One configured development account with centralized owner checks |
| Files | Private local storage | Preserved source bytes and artifacts outside container writable layers |
| Checks | pytest, Ruff, mypy | Regression, numerical, source, and typing checks |
| Dependencies | uv and its committed lockfile | Reproducible Python installation |

Domain records, job delivery, and workflow checkpoints have separate owners.
A checkpoint stores execution position; it does not replace an accepted scientific record or immutable handoff.
The API and worker share the Python codebase and run as separate processes, so jobs continue after browser disconnection.

See the [backend manifest](../pyproject.toml), [lockfile](../uv.lock), and [data model](data-model.md).

## Generated-code execution

A separate runner executes generated Python with network access disabled, bounded resources, and restricted filesystem access.
The generated process receives approved inputs and an output directory, without application credentials or Docker control.
Its scientific libraries are pinned in [the runner requirements](../docker/explore.requirements.txt).

Static source checks, independent review, and explicit approval precede target-effect execution.
Those checks complement numerical verification; they do not prove arbitrary generated programs correct.
See [security boundaries](security.md#generated-code-boundary) and [analysis approval](analysis-slice.md).

## Deployment

The delivered installation uses [Docker Compose](../compose.yaml), PostgreSQL, private local file storage, and a separate sandbox service.
Caddy serves the built frontend and proxies `/api` under one origin.
The [operations guide](operations.md) covers migrations, startup, backup, restore, and rollback.

The selected cloud design uses these components:

| Component | Target |
| --- | --- |
| Application services | Docker Compose on a Compute Engine VM |
| Database | Cloud SQL for PostgreSQL |
| Source files and artifacts | Private Cloud Storage |
| Generated-code execution | A separate Compute Engine VM with Docker and gVisor |
| Images and secrets | Artifact Registry and Secret Manager |
| Extended account management | Firebase Authentication behind the existing current-user boundary |

This cloud topology is a design target. Cloud provisioning, the Cloud Storage adapter, Firebase integration, and gVisor compatibility verification are not part of the local installation.
The single application VM design requires host maintenance and backups and does not provide high availability.

Keep deployment configuration and credentials server-side.
Apply migrations as an explicit release step, preserve stable owner identities, and validate restore and schema compatibility before a production release.

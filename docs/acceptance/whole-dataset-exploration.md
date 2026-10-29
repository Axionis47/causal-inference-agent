# Whole-dataset exploration acceptance, 25 October 2026

## Implemented boundary

The main chat supplies its question, complete selected CSV version, context and optional published artifact references. The persistent A2A agent writes Python itself through the shared episode. Generated code runs only in Docker. Inputs are unchanged and read-only. Family diagnostic projections retain their separate restrictions.

The integration service publishes one hierarchy in PostgreSQL:

`Conversation → ExplorationRequest → Attempt → ExplorationResult → Artifacts and registered files`

Results retain the answer, methods, evidence-backed findings, limitations, tags, timestamps, exact input version/hash and parent references. Private files live at `users/<user-id>/conversations/<conversation-id>/artifacts/<artifact-id>/`. The chat's owner-checked artifact routes retrieve the explanation and checksum-verified files. The filesystem contains scratch/cache only, with no second artifact catalogue or dual-write fallback. Another conversation cannot read these artifacts implicitly.

Compose starts the main app, persistent A2A agent, enforcing sandbox worker, PostgreSQL and private S3 storage. The worker processes one script at a time, with no network, application credentials, application volumes or Docker socket. A library-only child filesystem, dropped credentials and seccomp isolate each task. Inputs are root-owned and read-only; only task scratch/output is writable. Unavailable isolation fails visibly.

The worker is capped at 1 CPU, 1 GiB of memory, 64 container processes and a 512 MiB task filesystem. An idle restarted worker used about 10 MiB in this acceptance. Per-request limits remain eight scripts, sixteen inspections, three final-record attempts and sixty seconds per script. A generated plot is opened automatically, consuming one inspection, and actual pixels are supplied to the next action and final-record call. The agent may explicitly reopen outputs.

## Real Gemini main-chat runs

Source: the repository's public Uruguay government-transfers CSV, distributed with `causaldata` from the Manacorda, Miguel and Vigorito study. Supplied account: `data/context/gov_transfers.md`. Acceptance used descriptive statistics only and did not fit a causal effect.

CSV SHA-256: `c1f1699a0f9d2d3182aa4373c18830e0eefa5d32862b265fa5e642a9ad97d116`.

Conversation: `94d0e3f6-0210-4d44-bcce-8599d9c811f2`.

1. Correlation and scatter plot through the main chat. Gemini computed Pearson correlation between Education and Age: `-0.04454008478033806`, with 1,897 complete pairs and 51 excluded rows. The independent calculation matched. It ran one script, inspected the image and passed the final gate without a refusal. Artifact `389b0c62a364495e9bc8ed32cf066f9c`; trace `01a174f2-0e3b-7323-bc16-570549fe259e`.
2. Saved-result visualization after replacing all four containers. Gemini read the supplied saved JSON, preserved the three numerical values and generated a horizontal bar plot without reading the CSV or recomputing the correlation. Artifact `3f1b72171da840b89c768da9b28bad1b`; trace `01a174f8-7472-7913-b747-b62cfdb69bd2`. This run succeeded but wasted all eight script executions before explicitly inspecting the final plot. Automatic image delivery was added to address that observed behavior.
3. Final saved-result run after image and retry feedback fixes. Gemini inspected the supplied facts JSON, used its exact values in one Python script, reviewed the automatically attached plot pixels and passed the first final-record gate. Two explicit actions, two inspections including the automatic image, one execution and no refusals. Artifact `952dac886be7492d85a487baa7a49b23`; trace `01a17506-ddea-73e1-894d-4808dc72ed61`. All 112 trace spans ended. The script used values read by the inspection tool and did not access or recompute from the CSV. The rendered plot and explanation were verified in the main chat after a page reload.

Both published plots and their generated code were inspected. Trace readback showed the main chat, A2A episode, Docker execution, gates and publication under the same parent run. Evidence files are retained under `.artifacts/owned-exploration-acceptance/e27a/` in the acceptance checkout.

Earlier runs exposed an ARM OpenSSL native import failure, prefixed artifact IDs returned by the routing model, repeated use of output inspection for a supplied input, and final records citing sample counts absent from their selected output. The runtime now uses baseline ARM OpenSSL capabilities and checks the Vertex import at build time. Routing validates bare artifact IDs with bounded feedback and passes the person's exact question unchanged. Inspection refusals identify the appropriate action and filename. Shared episode retries show the rejected proposal alongside the gate errors; exploration can repair the record using existing evidence instead of regenerating a plot. Failed requests publish nothing; execution budgets were never bypassed.

## Persistence and isolation checks

All four service containers were replaced while retaining their volumes. The first artifact's local cache was removed before replacement. The original file, published facts, explanation and plot were retrieved through their owner-checked API routes afterward. The waiting conversation checkpoint survived. The follow-up used the saved result from private storage. Original CSV bytes remained identical.

The worker probe verified read-only inputs, absent application credentials and files, denied network, timeout, cancellation and a fresh filesystem for the next task. Stopping the worker changed readiness to HTTP 503. A persisted interrupted-job marker produced a visible failure after restart instead of replaying the script.

Real PostgreSQL/S3 tests cover publication, cache-independent retrieval, exact dataset versions, parent references, cross-owner and cross-conversation denials, database ownership constraints, partial upload failure and delayed cleanup that preserves published files. Shared adjustment, DiD and discontinuity diagnostic regressions passed against the Docker image.

## Check results

- Full Python suite: 550 passed, 87 opt-in integration cases skipped. After the final shared retry-feedback change, all 150 focused exploration, episode, tracing and family diagnostic tests passed.
- Separate real PostgreSQL/S3 publication and routing coverage passed, including the main-chat create, publish, retrieve and reuse journey. The most recent combined storage/routing/desk run had 40 passing tests.
- Web suite: 74 passed. TypeScript, frontend lint and production build passed.
- Ruff lint/format, mypy (253 source files), all twelve import contracts and diff whitespace checks passed. OpenAPI and generated client types were regenerated.
- Final disposable Compose smoke passed: browser ingestion and profiling, saved context browsing, enforcing sandbox probe, replacement of all four services, unchanged originals and persistent metadata/context. Its containers and volumes were removed. The separate live acceptance deployment was retained.

## Scope

Verified on Docker Desktop Linux/ARM64. Docker Engine/Desktop running Linux containers is the deployment requirement; host Python and Node are not required. Model work also requires configured Vertex credentials and project access. Windows and Linux x86 were not executed locally, so this is not a claim that every machine has been tested.

The application remains a local development deployment with an owner-checked development principal, not a public login system. Causal working memory, transcripts and SQLite checkpoints remain on durable application volumes. Full production authentication, cross-conversation artifact sharing, database-backed causal memory and backup restoration remain separate work. See [container setup](../production/containerization.md) for startup and credential configuration.

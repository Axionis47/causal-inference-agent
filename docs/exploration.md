# Whole-dataset exploration

The chatbot calls `explore_data(question, dataset_version, context, artifact_refs=[])`. It supplies the complete selected CSV unchanged. There is no column list, row list or projection argument. The integration envelope adds conversation, request and attempt IDs and a task output location.

## Supplied inputs

`DatasetInput` and `ArtifactInput` identify a version/file, path and SHA-256. The trusted caller resolves references before dispatch. The agent receives only a task manifest and copies of those files. CSV, TSV, JSON, PNG, JPEG and text artifacts are supported. Unknown references, changed checksums and unsupported files fail visibly.

User-facing exploration may inspect all supplied columns, including correlations. It does not modify CSV versions or column context. Transformations remain a separate authorized operation. Family diagnostic workflows still construct their own restricted inputs before calling the shared sandbox; this exploration change grants no additional causal authority to a lane.

## Agent and evidence

The existing A2A process runs the shared `lane/episode.py` loop. `ExploreAction` chooses `inspect_inputs`, `run_code`, `inspect_output`, or completion. A successful generated plot is opened automatically and its pixels accompany the execution result, spending one inspection. Explicit inspection can reopen it. Successful step outputs can be reused through the read-only task input directory.

Budgets in `explore/checks.yaml`: eight executions, sixteen inspections, three final-record validation attempts and sixty seconds per script. Failed execution attempts consume budget. The agent can repair code within the same bounds.

A completed `Exploration` contains a direct summary, calculation method and rationale, evidence-backed findings, limitations, tags and a numerical, tabular or visual product. Every selected fact must exist in a successful step's finite `facts.json`. Numerical answer claims must agree with the selected facts. Each finding cites fact names, `[table]` or `[figure]`. Images must decode and must have been inspected. Unsupported claims or products trigger a visible gate and bounded retry.

A saved causal result can be plotted using its supplied estimates and uncertainty without rerunning the causal fit. Missing draws, confidence bounds or other essential data produce `status=needs_input` with specific requests for the chatbot. The agent does not search storage to fill those gaps.

## Docker runtime

Build the installed runtime with:

```sh
docker build -t causal-desk-explore -f docker/explore.Dockerfile .
```

This is also `make explore-image`. The same Linux image definition supports Docker Engine or Docker Desktop on Linux, macOS and Windows with Linux containers. Python and the numerical dependencies are pinned. Supported build architectures are determined by availability of those pinned binary wheels; local verification does not establish that every OS/architecture has been tested.

Generated code runs only inside Docker. Local development launches disposable containers. Compose keeps a small worker running, with one active script at a time. Both paths use the same pinned analysis libraries and return the same bounded outcome contract. There is no host Python execution fallback.

The Compose worker has no network, credentials, application mounts or Docker socket. Its root is read-only. A trusted supervisor reads a private queue, copies only the supplied inputs, then enters a library-only filesystem root and drops to UID 20000 before executing generated code. Inputs are root-owned and read-only. The child cannot see the queue, supervisor, `/proc`, other jobs or application files. Seccomp denies sockets and process-group escape; no privilege escalation is allowed. Task processes are killed as a group on completion, timeout or cancellation, and scratch is removed before the next task.

The worker ceiling is one CPU, one GiB memory, 64 container processes and a 512 MiB task filesystem. Individual scripts have a 60-second deadline, 32-process and 64-file-descriptor limits, a 5 MiB per-file limit, 25 MiB output limit and bounded stdout/stderr. Large inputs can fail visibly at these limits. Startup probes enforce the actual boundary before the worker becomes healthy. No weakened mode is available.

`make container-build` builds app and worker. `make container-up` starts PostgreSQL, private S3, the sandbox, the persistent A2A agent and the web/API runtime. `make container-sandbox` runs the numerical/plot, read-only-input, network, privilege, timeout, cancellation and fresh-task checks through the running app. `/api/ready` checks the sandbox and A2A service. Model credentials remain required for model calls. See [containerization](production/containerization.md).

## Publication and later chat reads

Starting a conversation from dataset details pins an owned dataset version and supplied context. PostgreSQL owns this hierarchy:

```text
Conversation -> ExplorationRequest -> Attempt -> ExplorationResult -> Artifacts/files
```

The integration service verifies the complete input checksum, records the request and attempt, validates the returned explanation/products, and writes immutable files to:

```text
users/<user-id>/conversations/<conversation-id>/artifacts/<artifact-id>/
```

The file store verifies the returned bytes and SHA-256 before one database transaction publishes the result, artifact and file registrations. A failed upload never creates a readable artifact. Pending manifests support cleanup without deleting published files. `python -m causal_agent.server.ingestion cleanup` reconciles abandoned uploads and exploration attempts older than 24 hours.

The result retains the answer, method, findings, limitations, facts, tags, timestamp, input version/checksum and parent references. Registered files include `result.json`, `facts.json`, `request.json`, `trail.json`, code, optional plot/table and metadata. Input copies are removed after execution. The filesystem under `.artifacts/runs/explorations/` is scratch/cache only. It is never scanned as a publication catalogue, and no dataset shelf fallback or dual write remains.

Chat tools use an owner-bound repository supplied by the integration service. `desk/material.py` makes prior conclusions and numerical evidence citable; explicitly chosen artifact IDs resolve their saved result, numerical file, table and image for follow-up exploration. The main chat can begin with a descriptive question or visualization, without first inventing a causal question. The chatbot supplies the question and data; the exploration agent writes Python.

HTTP reads use `/api/conversations/<id>/artifacts`, `/artifacts/<artifact-id>` for the full linked explanation, and `/artifacts/<artifact-id>/files/<filename>` for registered files. Every route checks the principal and conversation. References from another conversation are rejected; cross-conversation sharing requires a future explicit ArtifactUse operation. The local deployment uses its configured development principal; public authentication is still separate work.

This slice adds owned conversation identity and exploration publication. Causal working memory, transcripts, journals and checkpoints still use the existing persistent filesystem/SQLite runtime behind the owner-checked conversation boundary. It does not claim the full production memory, job-lease or transformation-lineage migration is complete.

## Verification

Tests cover complete CSV delivery, unchanged bytes, exact input hashes, successful and failed scripts, repairs, image inspection, grounded claims, missing-input requests, Docker boundaries and cleanup. Shared diagnostic tests execute against Docker with their existing family-specific input restrictions. A real Gemini acceptance and its trace are required in addition to these deterministic tests.

The [25 October acceptance](acceptance/whole-dataset-exploration.md) records real Gemini runs and trace readback. The owned Compose integration is covered by the conversation publication tests and runtime probe.

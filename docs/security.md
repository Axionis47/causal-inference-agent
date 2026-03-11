# Security and trust boundaries

## Scope

This document describes implemented access and isolation controls.
It is not a penetration-test report or a production certification.
The current installation is a local development application.

Sources: [authentication](../backend/auth.py), [HTTP boundary](../backend/main.py), [storage](../backend/storage.py), and [runner isolation](../docker/isolation.py).

## Trust boundary map

| Boundary | Input treated as untrusted | Control | Remaining limit |
| --- | --- | --- | --- |
| Browser → API | IDs, payloads, cookies, Origin | Session validation, owner checks, typed contracts, request limits | Development account system is not a full identity platform |
| Upload → Storage | Filenames and bytes | Format limits, generated storage keys, checksums | No general malware or document-conversion service |
| Source text → Model | Instructions embedded in source content | Source-only prompts and typed proposal validation | Model interpretation remains fallible |
| Model → Scientific records | Claims, references, code, explanations | Admission rules, gates, frozen prerequisites | Finite rules do not prove arbitrary scientific correctness |
| Generated code → Host | Python and requested file access | Separate runner, network isolation, identity and syscall restrictions | Local controls are not the proposed cloud gVisor deployment |
| Saved artifacts → Chat | Model-selected IDs and files | Owner checks and turn-bound catalogue hashes | Nonnumeric explanation still requires review |

## Accounts and sessions

The CLI creates or updates a development account.
Passwords use the configured Argon2 password-hash implementation.
The browser receives a random token in an HttpOnly cookie.
The database stores a SHA-256 token hash, user reference, and expiry.

The cookie uses SameSite Strict and the `/api` path.
Secure cookies are configurable and disabled in the local HTTP setup.
Logout deletes the session record.
Updating an existing account password revokes its sessions without changing ownership.

Registration, password recovery, invitations, roles, and Firebase integration remain deferred.
The current code does not establish production login rate limiting or multi-factor authentication.

## HTTP and ownership controls

Mutations require the application's custom header.
An Origin header, when present, must match the configured allowed origins.
The backend enforces Content-Length and operation-specific request bounds.

Dataset ownership controls access to versions and dependent records.
Context, conversations, analyses, and artifacts validate their parent scope.
An unknown or foreign-owned domain identity generally returns 404.

Responses disable ordinary caching and set content-type and frame protections.
Caddy also supplies the frontend content security policy.
These controls do not replace correct owner checks or output validation.

## Source and artifact storage

The application generates storage paths.
User filenames do not become unrestricted filesystem paths.
Source publication is immutable.
Reads verify checksums before returning bytes.

PostgreSQL holds metadata and manifests.
Private local storage holds original source files, generated outputs, and execution receipts.
Browser access goes through authenticated endpoints.
No public file bucket is configured.

The application has no general deletion or retention workflow.
Backups contain sensitive study material and analysis outputs.
Protect them with the same access controls as live storage.

## Generated-code boundary

The worker submits code to a separate Docker runner.
The runner has no network.
It does not mount application credentials, the source-file store, or the Docker socket.
It receives only the submitted input bundle and code.

The trusted supervisor prepares a per-script environment.
The execution path uses chroot, a dropped identity, syscall restrictions, and resource limits.
The Compose service also limits memory, CPU, processes, capabilities, and writable locations.
The supervisor's required privileges do not become general permissions for generated code.

The runner reports an enforcement heartbeat and runtime hash.
The worker rejects an unavailable or mismatched runtime.
It checks output file type, size, format, and declared numerical facts before publication.

For analysis diagnostics, the harness constructs restricted input bytes first.
The script cannot fetch withheld outcomes from application storage.
For approved execution, the input manifest binds the permitted columns and exact source identity.

Local isolation is implemented.
A dedicated cloud runner with gVisor remains a deployment target.

## External data transfer

| Destination | Data that can be sent | Trigger |
| --- | --- | --- |
| Vertex AI | Selected source text, context, user requests, evidence packets, code, observations, and inspected images | Model-backed preparation, chat, or analysis |
| LangSmith | Instrumented model/workflow inputs, outputs, and diagnostic metadata | Tracing enabled with configured credentials |
| Sandbox runner | Supplied source or projected bytes, selected artifacts, code | Explicit exploration or permitted analysis computation |

Do not assume that local data stays entirely on the local machine during model work.
Review the provider and trace configuration before using sensitive study material.
Credentials remain server-side and are not part of frontend bundles.
Compose injects model and trace credentials into the worker.

The documentation update did not change provider retention settings or organization policy.
It did not send a new model test request.

## Scientific integrity boundaries

Only the frozen context can supply source authority after analysis launch.
Model memory and later chat cannot add source facts.
Assumption candidates remain inactive until the supported causal rules admit reliance.

The graph freezes before diagnostics.
The executable specification freezes before target-effect execution.
Approval binds exact science, program, inputs, and runtime.
Programming recovery cannot authorize scientific redesign.

These controls protect recorded decisions and access boundaries.
They do not prove that supplied evidence or generated code is scientifically correct.
See [verification](verification.md) for required scientific checks.

## Deployment boundary

The local app binds its public port to loopback and uses HTTP.
External deployment requires HTTPS, secure cookies, the correct origin, protected credentials, and validated storage and execution services.
Cloud provisioning and production security verification remain deferred.
Use the [technology decision](tech-stack.md) for the selected target architecture.

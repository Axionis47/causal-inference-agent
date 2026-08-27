# Verification and evidence

## Purpose

Verification establishes a specific behavior under specific inputs and versions.
It does not transfer automatically to a later contract or broader scientific claim.
This guide maps application boundaries to their test sources and gives commands for reproducing checks.
Record each execution's inputs, versions, logs, and results alongside its evidence.

## Verification layers

| Layer | Question | Evidence |
| --- | --- | --- |
| Static | Do source code and types satisfy configured checks? | Lint, typing, build, schema drift |
| Contract | Are invalid requests, references, and lifecycle changes rejected? | Focused deterministic tests |
| Persistence | Do writes, replay, ownership, and restart preserve records? | PostgreSQL integration tests |
| Numerical | Does the frozen program match an independent calculation? | Numerical fixtures and comparison tolerances |
| Model trajectory | Does the configured provider complete the intended scientific task? | Actual requests, outputs, gates, and trace inspection |
| Browser journey | Can a person complete the connected UI action? | Real API, worker, persisted records, and browser inspection |
| Operations | Can the installation restart, back up, and restore? | Recorded operator drill against isolated data |
| Documentation | Do routes, records, links, terms, and procedures match the code? | Documentation review and source-derived coverage checks |

No one layer substitutes for the others.
A provider schema check does not establish a completed analysis.
A passing numerical fixture does not prove arbitrary generated code correct.

## Static and database checks

```sh
uv run ruff check backend tests migrations
uv run mypy backend
pnpm --dir web lint
pnpm --dir web build
uv run alembic check
```

Run database integration tests only against the isolated `causal_test` database.
Create that database once if it is absent:

```sh
docker compose exec -T db createdb -U causal causal_test
```

Run the suite:

```sh
uv run python scripts/test.py
```

The script installs application and worker schemas before pytest.
It refuses a test database name other than `causal_test`.
The fixtures reset test data.
Do not point destructive tests at the preview or a user's database.

## Boundary coverage

| Boundary or invariant | Representative test source |
| --- | --- |
| Login, upload, owner isolation, immutable source | [test_first_slice.py](../tests/test_first_slice.py) |
| Context citations, revisions, stale writes, preparation recovery | [test_context.py](../tests/test_context.py) |
| Ordered turns, exact inputs, artifact access, Apply changes | [test_chat.py](../tests/test_chat.py) |
| Launch readiness and exact input freeze | [test_analysis_readiness.py](../tests/test_analysis_readiness.py), [test_analysis_launch.py](../tests/test_analysis_launch.py) |
| Unsupported claims and inactive assumptions | [test_analysis_claim_authority.py](../tests/test_analysis_claim_authority.py) |
| Native agent loops and constant column-count budgets | [test_agents_journey.py](../tests/test_agents_journey.py) |
| Exact numeric display handles and source validation | [test_numerical_handles.py](../tests/test_numerical_handles.py) |
| Task allowances, widening, and ownership | [test_analysis_tasks.py](../tests/test_analysis_tasks.py), [test_analysis_widening.py](../tests/test_analysis_widening.py) |
| Scope-specific roles and changing treatment time | [test_analysis_role_scopes.py](../tests/test_analysis_role_scopes.py), [test_analysis_temporal_role_scopes.py](../tests/test_analysis_temporal_role_scopes.py) |
| Diagnostic isolation and execution projection | [test_analysis_controls.py](../tests/test_analysis_controls.py), [test_execution_projection.py](../tests/test_execution_projection.py) |
| Approval rejection and preserved failures | [test_analysis_rejection.py](../tests/test_analysis_rejection.py) |
| Recovery, telemetry, and provider attempts | [test_analysis_recovery.py](../tests/test_analysis_recovery.py), [test_analysis_telemetry.py](../tests/test_analysis_telemetry.py) |
| Numerical reference cases | [test_analysis_reference_numerics.py](../tests/test_analysis_reference_numerics.py), [test_analysis_local_wald.py](../tests/test_analysis_local_wald.py) |
| Import boundaries, ordered schema parity, public API, and runtime hashes | [test_component_boundaries.py](../tests/test_component_boundaries.py) |
| Provider schema conversion | [test_provider_schema.py](../tests/test_provider_schema.py), [test_treatments_provider_encoding.py](../tests/test_treatments_provider_encoding.py) |
| Workspace navigation, unsent drafts, context return, optional diagnostics | [navigation.spec.ts](../web/e2e/navigation.spec.ts) |

These sources identify relevant coverage.
The existence of a test file is not evidence of its latest result.
Use the recorded execution log and contract version.

## Invariance and sensitivity

Fixed-plan invariance and fresh semantic agreement are different claims.

**Fixed-plan invariance:** Explicit semantic mappings preserve normalized decisions and numerical inputs across renamed, reordered, or wider datasets.

**Fresh semantic agreement:** Independent model trajectories interpret equivalent supplied meaning consistently.

The verification set must include these cases:

- Renamed and reordered columns with explicit mappings.
- 12-, 60-, and 200-column packet-isolation cases.
- An irrelevant added column that does not enlarge downstream scientific packets.
- A relevant common cause, assignment change, selection mechanism, or derived outcome proxy.
- Multiple treatment factors with shared assignment.
- Effect-relative roles and adoption histories.
- Unsupported switching that blocks rather than inventing an estimator.
- Familiar labels with different supplied mechanisms and opaque private labels.

Use [packet-order tests](../tests/test_analysis_packet_order.py) and [authority-view tests](../tests/test_analysis_authority_views.py) for structural isolation.
Preserve actual failed proposals for regression replay.
Do not change the causal plan to make a renamed or wider test pass.

## Real-model and browser acceptance

These checks send data to configured model and trace services.
Run them intentionally in the review environment.
Do not auto-approve generated programs to obtain a passing trajectory.

The current-contract synthetic journey starts with:

```sh
uv run python scripts/frozen_context_acceptance.py start
```

Continue with the script's `inspect` and `discuss` actions.
Review the proposal, Apply changes, and launch through the browser.
Keep its independent numerical oracle outside model packets.
Inspect exact requests, model outputs, gates, and available LangSmith traces.

Earlier `context_acceptance.py`, `chat_acceptance.py`, and `analysis_acceptance.py` scripts serve their recorded workflows.
Check contract compatibility before reusing an older analysis fixture.

Browser tests use Playwright:

```sh
pnpm --dir web exec playwright install chromium
E2E_BASE_URL=http://127.0.0.1:8088 pnpm --dir web test:browser
```

Some journeys require `.runtime/context-acceptance.json` from context acceptance.
Inspect test preconditions before running the suite.
Browser checks must exercise real saved state, not only a screenshot or mock endpoint.

## Documentation checks

```sh
uv run python scripts/check_docs.py
```

The checker validates local Markdown links, API operation coverage, table coverage, and agent identifiers.
It also checks em dashes in the current manual.
It does not establish technical correctness by itself.

Review each component handoff for sender, receiver, input, validation, durable output, and failure behavior.
Inspect source imports before claiming a one-way dependency architecture.
Check enforced dependencies in [architecture](architecture.md#enforced-code-dependencies).

## Evidence record

For a verification result, record these fields:

- Application, schema, and scientific contract versions.
- Dataset and context identities or approved fixture description.
- Command or browser procedure.
- Expected result and observed result.
- Logs, receipts, traces, and numerical comparisons.
- Failures, omitted checks, and limits of the claim.

Do not record a skipped or blocked check as passed.
Do not describe historical success as current-contract verification.
Do not reset budgets or reroll failed scientific work without preserving its original evidence.

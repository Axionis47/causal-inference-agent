# T-040 — Intake ownership and deterministic resource processing

Requested 2026-09-10: refactor intake meaningfully and clarify the CLI/design boundary.

Implementation and focused verification complete; see [verification](T-040-verification.md).

Intake owns source acquisition, archive admission, resource profiling, source evidence,
semantic availability, durable outcomes, and the intake-to-design handoff. The CLI parses
commands and renders results; design owns table selection, causal interpretation,
clarification, and approval. `new` performs intake; `run` advances the pipeline.

Implementation separates a public intake dependency/entry boundary, invocation-local
run state and persistence, deterministic resource processing, and workflow assembly.
The existing coordinator constructor, submission/result contracts, seven CLI commands,
artifact schemas, and ordinary successful payloads remain compatible. Existing context
and temporal-profile work is preserved. No model or design dependency belongs in intake.

Resource processing refuses duplicate ZIP names, records failed extraction per resource,
and lets independent good tables finish. Numeric profiles remain serializable when source
values contain NaN or infinity; those values remain counted explicitly. Refused outcomes
remain closed to design. Replay must avoid provider work and cross-run state leakage.

Verification covers existing intake/CLI integration behavior, deterministic resource
failures, finite numeric summaries, independent public entry, replay, and downstream
entry. Record lint, type checking, and complexity measurements against the starting
worktree. Existing repository complexity ceilings are unchanged. This task does not
rewrite the separately active post-analysis work or claim live-provider verification.

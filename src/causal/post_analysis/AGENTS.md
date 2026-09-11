# Post-analysis

This directory owns causal interpretation, visualization decisions, report
composition, inspection and delivery. Refactor by deleting duplication and
consolidating responsibility. A new facade around the old workflow does not
complete the refactor.

## Component map

- [Contract](../../../docs/post_analysis/contract.md): inputs, invariants and issue ownership.
- [Graph](../../../docs/post_analysis/graph.md): nodes, routes, state writers and recovery.
- [Refactor map](../../../docs/post_analysis/refactor.md): callers and move/merge/remove targets.
- [Coverage](../../../docs/post_analysis/coverage.md): existing capabilities and integration gaps.

The documents track the implemented cutover and explicit compatibility boundaries.
Verify actual code and callers when extending behavior. Follow the user's
current task scope; this file does not authorize unrelated rewrites.

## Delete duplication, consolidate ownership

1. Identify the behavior, its live callers and its single destination owner.
   Name the superseded code that each replacement will remove.
2. Move or merge useful logic rather than copying it. Switch callers, verify the
   replacement and delete superseded execution code, prompts and obsolete tests
   in the same completed cutover. Preserve unique behavioral coverage.
3. Do not leave another claim agent, layout curator, report narrator or coordinator
   active behind this package. A graph node must not call the old reporting pipeline.
4. Avoid pass-through layers, speculative abstractions, duplicated schemas and
   fallback routes to old behavior. Necessary public/integration adapters call
   the single active implementation. Document temporary adapters' callers and
   removal conditions in the refactor map.
5. Preserve original artifact bytes/hashes and only the historical readers still
   required. Historical readability does not justify a second execution pipeline.
6. Keep stage-owned resources and tests here. Shared registrations, migrations
   and generic dispatch remain shared; external callers delegate through the
   public entry without implementing post-analysis behavior.

Measure cleanup by distinct implementations, live paths and clear ownership.
Do not reduce line count by dropping checks, hiding logic in configuration or
compressing code into oversized modules. Do not add modules for every method,
diagnostic, chart or artifact name.

## Scientific and shared boundaries

- Consume one common-harness handoff with exact approved context/DAG, columns,
  roles, requests and evidence. Reuse shared references, storage, model-task/gateway,
  tracing and operational services; do not create competing generic infrastructure.
- Read expected computations from the actual request. An evidence index is a
  derived view, not another editable scientific record or capability catalog.
- Estimation, diagnostic execution and scientific supporting calculations remain
  upstream. Trace numerical dependencies before removing figure-related code;
  visual prescriptions and layout belong here.
- Source IDs/revisions, roles, causal edges, targets, coding, values, uncertainty
  and statuses are immutable. Display derivations have source lineage and cannot
  introduce new fits, tests or uncertainty estimates.
- Reject input defects with the source/path, expected versus received information,
  responsible owner and required action. No LLM repairs upstream science. Receiver
  support gaps and storage outages are identified separately.
- Failed diagnostics are accounted-for evidence, not passes. Missing expected
  records are defects. If no response exists, validate the authoritative attempt
  failure and disclose it without inventing computation outcomes.
- Upstream decides whether a traceable clarification preserves existing results
  or requires reassessment/reanalysis. Post-analysis does neither silently.

## Keep the LangGraph understandable

LangGraph and LLM reasoning are mandatory: one autonomous authoring agent, plus
a bounded read-only LLM review alongside code checks. The core path is
`input_check → agent → review → release → finish`, with `agent ↔ tools` for work
and `review → agent` for bounded corrections. Source defects and unrecovered
failures use documented stop paths. Only code can release the exact reviewed export.

- The compiled graph owns authoring, tool, review and release transitions. Do not
  hide another coordinator or autonomous model loop inside a node or tool.
- State holds source/report references and bounded control/observations. Validated
  tools write revisions; the model cannot patch authority fields or budgets.
- Late source defects take the same owner-aware rejection path as intake defects.
  Invalid authored content receives bounded revision feedback instead.
- Review examines evidence-linked claims and actual final previews. Changes to
  dependent artifacts invalidate review. Missing review is not a pass, and stale
  review cannot release a newer draft/export.
- Reuse shared retries/idempotency. Recovery preserves exact sources, committed
  artifacts and spent budget; it cannot restart the science.
- Update the graph document in the same change as nodes, edges, state ownership,
  tool permissions, stop rules or recovery. Keep its diagram, node table and
  failure paths aligned with code; record justified deviations from the proposal.

## Verify the cutover

Check affected behavior: missing versus failed diagnostics, failure before a
response, wrong context/column/DAG bindings, branch diagnostic coverage, source
fidelity through export, review-driven revision, interrupted/idempotent recovery
and stale-review rejection. Use representative evidence and appropriate existing
checks; avoid tests that merely mirror filenames or implementation details.

Search imports, dispatch, recovery and delivery callers before deleting old
entry points. Confirm one active route and unchanged scientific outputs. Do not
claim completion while the new entry still delegates domain work to the old path.

Report what was removed, where merged behavior lives, which callers changed and
what verified the result. Name remaining temporary adapters and their removal
conditions; do not present unfinished consolidation as complete.

## Trace the complete application exchange

- Live post-analysis requires LangSmith preflight and acknowledged final flush.
- Trace the rendered prompts, schemas, responses, exposed provider reasoning summaries,
  concise decision summaries, actual preview images, node/tool inputs and outputs.
- Nest model spans under the graph task. Do not replace content with hashes or
  silently sample it. Preserve hashes and identities alongside complete content.
- Redact credentials through shared tracing. The user authorized full study data;
  do not blanket-redact research evidence. Unavailable internal chain-of-thought
  cannot be logged and must not be requested as an application requirement.
- Keep ordinary operational events/logs for lifecycle, validation and failures.

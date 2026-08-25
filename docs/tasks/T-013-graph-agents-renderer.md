# T-013 — Agents, prompts, LangGraph graph, interrupts, renderer, coordinator, outcome, handoff

Status: frozen; BLOCKED_COMPLEXITY_BUDGET pending a user decision (§5)
Owning PRD: PRD-002 (§8, §9, §11.1, §12.3, §16.4, §19–§23)
Depends on: T-010, T-012 (both ACCEPTED)

## 1. Deliverables

1. `registries/design-tasks.v1.json` — one row per model task kind (`intent`,
   `semantic_batch`, `role_evidence`, `causal_synthesis`, `method_design`): prompt template
   path + `prompt_version`, output artifact type + schema version, highest validation wall,
   allowed stopping states, token/tool/correction budgets. Declarative.
2. `prompts/design/*.v1.txt` — the five prompt templates (declarative scope). Each instructs
   JSON-only output against the registered schema, cites only allowlisted evidence ids, and
   forbids inventing columns, evidence, or roles.
3. `src/causal/design/compile.py` — deterministic `MeasurementMap` compiler (PRD-002 §9.5 —
   concepts from validated cards + intent proposals; links from card `concept_id`s; unmeasured
   concepts preserved), task-envelope construction from the task table + manifest recipient
   map (generic assembly helpers live in shared), prompt loading/rendering.
4. `src/causal/design/renderer.py` — harness-owned CausalGraphView compiler (§12.3): DOT via
   the pinned `graphviz` package, SVG via the local `dot` binary (subprocess; actual Graphviz
   version recorded in `renderer_version` metadata; absent binary raises a typed blocker —
   D-042, no substitution), status→line-style mapping with text legend, role annotations,
   alternatives as separate labelled views, accessible summary + node-edge table, spec hash,
   wall-8 fidelity check (every validated node/edge/status present; nothing added).
5. `src/causal/design/graph.py` — the LangGraph harness: `StateGraph` over the §19.1 state
   allowlist; PostgreSQL checkpointer (`langgraph-checkpoint-postgres`, strict msgpack, no
   pickle); nodes for entry gate, table selection (durable interrupt), manifest, intent,
   triage, semantic batches (dispatch sequentially within the ≤8 cap — parallelism is
   permitted by §9.1 but not required; recorded as D-050), fan-in + measurement map, role
   batches, synthesis + causal validation with ≤2 targeted corrections per artifact+code,
   method eligibility (all four packs) + method design + preflight diagnostics, design/frame
   validation, renderer, capacity check, ask gate (≤2 clarification rounds, durable
   interrupt), approval (durable interrupt), DesignOutcome commit, PRD-003 handoff opening;
   `design_runs`/`design_tasks`/`context_requirements` row upkeep; every boundary emits the
   §20.8 events through the existing emitter and commits through the flush-gated committer.
6. Tests: scripted-gateway end-to-end design run over the T-008 fixture dataset reaching an
   approved outcome; interrupt/resume across a new process boundary (same thread id);
   correction-loop exhaustion; two-round ask exhaustion → needs_context; refusal paths
   (no CSV, unsupported method); renderer fidelity (skipped without `dot`); handoff gate
   acceptance of the four-artifact manifest.

## 2. Already-landed preparation (D-049, commit 6ac88b9)

`ToolRouter`/`ToolError`/`ToolResult`/`tool_allowlists` → `causal.shared.toolrouter`;
`CatalogReader`/`ProductsReader`/`SqlCatalogReader` and the `FrameSource` family →
`causal.shared.readers`. Design re-exports keep the T-011/T-012 APIs. Design scope after the
move: 2,310 / 2,500.

## 3. Budget projection (§14.1.2 pre-code declaration)

| Piece | Projected logical lines |
|---|---:|
| `compile.py` | 80 |
| `renderer.py` | 115 |
| `graph.py` | 280 |
| planned compression of accepted design modules | −80 |
| **Projected design scope** | **2,310 − 80 + 475 = 2,705 / 2,500 — breach** |

## 4. ComplexityRethinkV1 (the task's one rethink, consumed pre-code)

1. **Duplication:** tool routing, catalog/object/product reads, and generic validation
   already moved to shared (D-048, D-049); no further duplication found — graph nodes call
   the accepted modules and add no second implementation.
2. **Pinned library instead of code:** LangGraph provides checkpointing, interrupts, and
   resume (no bespoke state machine); `graphviz` builds DOT; pydantic validates payloads.
   Already assumed by the projection.
3. **Interfaces without two V1 consumers:** none left in the additions; the planned
   `agents.py` module was already dissolved into the declarative task table + shared
   envelope assembly.
4. **Behavior outside the PRDs:** none to drop. Sequential-within-cap worker dispatch
   (D-050) already removes the parallel-dispatch wiring; every remaining piece maps to a
   numbered PRD-002 requirement (§11.1, §12.3, §16.4, §19–§23) or an acceptance criterion.
5. **Smaller revised plan:** floor ≈ 455 additions (compile 75, renderer 110, graph 270)
   with compression −80 → projected ≈ 2,685. **Still breaches.**

## 5. Amendment 1 — implementation record (2026-08-25)

Implementation landed at commit (see ledger): compile.py 105, renderer.py 133, graph.py 790;
suite 650 green including two end-to-end design runs (real `dot` 15.1.1 — the D-042 premise is
stale, the binary is present at exactly the pinned version), durable interrupt/resume across
coordinator instances, correction-loop and refusal paths. Recorded deviations, all
harness-agent decisions reviewed and accepted:

1. Entry/selection split into two nodes (gate acceptance must never sit in an interrupting
   node — duplicate_handoff on replay).
2. Ask gate asks one round per node execution and self-loops (round-1 replay over a mutated
   requirement store would break answer validation).
3. V1 model tasks receive fully harness-hydrated context sections; `VertexGateway` has no
   tool-calling loop (AFC disabled per SC §10.4), so `ToolRouter` and the retrieval handlers
   are not driven at runtime in V1 (D-053). Pre-repair diagnostics run harness-side over
   `CsvObjectFrameSource`.
4. Synthesis and method each run two bounded model tasks (context→ledger; design→contract),
   matching the one-strict-model-per-payload validator shape.
5. `handoff_open` builds but never records the manifest (D-037 pattern);
   `open_design_handoff` exposes it.
6. Interrupt anchors: IntakeOutcome (selection), UserQuestionPacket (clarification),
   ExperimentDesign (approval) — aligned with registry parent rules for T-014 validation.
7. Capacity cardinalities derived deterministically from the design (arms = contrasts+1,
   series = arms, evidence_items = visuals + post-repair diagnostics).
8. Registry fix (D-052): `TableSelection` and `DeliveryCapacityCheck` gained
   `preparation-harness` readers and `preparation` destinations — without them the §23
   handoff gate would refuse `reader_not_allowed`.

Budget outcome: design actual 3,338 / 2,750 and graph.py 790 / 350 — the task's rethink was
consumed pre-code, so this breach is terminal per SC §14.1.2 until the user rules (D-054).

## 6. Terminal state and the user decision

Per SC §14.1.2 the revised projection still breaches, so T-013 is
`blocked_complexity_budget` before feature code. The user may:

- **Option A (recommended): approve an immutable budget revision** — SC §14.1 design row
  2,500 → **2,750** (production total stays 15,000 and binds on actual lines; actual
  production today is 5,222). Projected T-013 then fits with ≈ 45–125 margin. The revision
  is one table-cell edit in `docs/product/SYSTEM-CONTRACT.md` §14.1 plus a ledger decision;
  PRD-002 acceptance criterion 39's reference updates with it.
- **Option B: reduce PRD-002 scope** — every candidate cut (drop the SVG causal-graph view,
  drop alternative-graph rendering, drop durable CLI interrupts) violates a numbered PRD-002
  requirement and would need a PRD amendment larger than Option A.

Smallest identified scope reduction if neither is approved: none exists inside the frozen
PRD; the block stands.

## Amendment 2 — real response schema for model tasks (2026-08-25, pilot finding, D-063)

The first live design run failed `correction_exhausted`: `harness_base._invoke` passes
`{"type": "object"}` as the gateway `response_schema`, so Vertex constrained decoding
deterministically emits `{}` at temperature 0.0 (all three attempts identical). The
registered draft schema never reached the model. Live probe (2026-08-25, gemini-2.5-flash,
Vertex `v1`, google-genai 2.19.0) confirmed the API accepts a raw
`model_json_schema()` dict **including `$defs`/`$ref`**, and the model then emits the
full `AgentTaskResultV1` structure.

Fix:

1. New `src/causal/shared/agenttask.py` (shared scope, ≤ 25 logical lines):
   `result_schema(draft: type[BaseModel]) -> dict[str, object]` — deep-copy
   `AgentTaskResultV1.model_json_schema()`, replace `properties.payload` with the
   draft's schema (draft `$defs` merged into the result schema's `$defs`; a key
   collision with a non-identical definition raises), memoized per draft class
   (schemas are static; determinism and no per-call rebuild). This module is the
   planned seed of the T-019 `agenttask` consolidation (D-059).
2. `design/harness_base._invoke` gains a `draft: type[BaseModel]` parameter;
   `_run_task` passes its `model`; the gateway call passes
   `agenttask.result_schema(draft)` instead of `{"type": "object"}`. Design scope is at
   3,398/3,400 and harness_base at 340/350 — the edit must land within both; trim
   within the touched lines if needed.
3. Tests: shared — payload replaced, `$defs` union, collision raises, memoization;
   design — a capturing FakeGateway asserts one task kind receives the merged schema
   (payload properties present, not bare `{"type": "object"}`).

Budgets: shared ≤ 2,500; design ≤ 3,400; tests additions ≤ 45 lines.

## Amendment 3 — registered requirement vocabulary in prompts (2026-08-25, pilot finding, D-064)

Live intent runs raised context requirements with invented ids (`req-treatment-cols-1`):
wall 2 rejects `unknown_requirement_id`, and the correction report names the failed rule
but not the allowed vocabulary, so corrections cannot converge. The registered ids live
only in `registries/context-requirements.v1.json`.

Fix (declarative + tests only; zero production lines):
1. Each of the five `prompts/design/*.v1.txt` templates gains a short "Context
   requirements" section: requirements MUST use one of the registered template ids,
   followed by the closed id list with a half-line usage cue each (from the registry;
   intent/semantic/role/synthesis/method may list only the ids meaningful to that task).
2. Edited in place without a version bump: no eval baseline exists yet (T-020 pending),
   recorded in the ledger as a pre-baseline prompt revision.
3. Test (≤ 20 lines, tests scope): every `design.`/`dataset.`/`column.` requirement id
   mentioned in any prompt template exists in `context-requirements.v1.json`, and every
   template that permits requirements names at least one registered id.

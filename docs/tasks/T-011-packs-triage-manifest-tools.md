# T-011 — Method packs, deterministic triage, context manifest, tool surface, entry gate

Status: frozen for implementation
Owning PRD: PRD-002 (§4, §5, §7, §9.4, §9.5, §13, §15; SC §5.4)
Depends on: T-009

## 1. Deliverables

1. `registries/method-packs.v1.json` — the four versioned method-pack manifests (§13).
2. `registries/context-requirements.v1.json` — the §10.1 common requirement templates.
3. `registries/design-tools.v1.json` — tool → task-kind allowlist (SC §5.4 audit ledger).
4. `src/causal/design/packs.py` — method-pack loading and structural validation (fail-closed).
5. `src/causal/design/triage.py` — deterministic column triage (`triage.v1`) producing the
   committed `ColumnTriageRecord` payload and frozen worker batches.
6. `src/causal/design/entry.py` — handoff opening checks, CSV candidate listing, table-selection
   routing, `DesignContextManifestV1` compilation.
7. `src/causal/design/tools.py` — the model-facing retrieval tool surface with allowlist
   enforcement (`list_intake_inventory`, `get_semantic_evidence`, `get_measured_facts`,
   `get_provenance`, `get_method_contract`). The three compute tools (`validate_causal_model`,
   `run_preflight_diagnostic`, `preview_eligibility_impact`) register in T-012 through the same
   router.
8. Registry row append: `ColumnTriageRecord` / `column-triage.v1` (D-046).
9. Tests under `tests/design/` (new files: `test_packs.py`, `test_triage.py`, `test_entry.py`,
   `test_tools.py`).

## 2. New artifact type (D-046)

PRD-002 §7 places deterministic triage after intent; §9.4 requires deferred columns to be
recorded and approval to be blocked when a deferred column could satisfy a blocking role. That
check runs at approval time, so the frozen tier assignment must be durable and hash-addressed:
`ColumnTriageRecord` (`column-triage.v1`), producer design-harness, readers design-harness,
required parents [DesignIntent], optional [], terminal [committed], destinations [design],
validator `column-triage-validator.v1`.

Payload model `ColumnTriageRecordV1` (add to `design/triage.py`, not contracts.py):
`schema_version` Literal; `triage_rule_version` Literal["triage.v1"]; `table_name`;
`tiers: dict[str, tuple[Identity, ...]]` with exactly the keys `critical`,
`plausible_adjustment`, `supporting`, `unused` (each a sorted tuple of column names; disjoint;
union must equal the structural inventory of the selected table); `batches:
tuple[TriageBatchV1, ...]` — `TriageBatchV1`: `batch_id`, `column_names` (min 1, from
critical ∪ plausible_adjustment only); `deferred: tuple[Identity, ...]`
(= supporting + unused, sorted); `match_trace: dict[str, str]` (column → rule id that placed
it). Cross-field validators enforce the disjoint/exhaustive rules and batch count ≤ 8.

## 3. Triage rule `triage.v1` (deterministic, versioned)

Inputs: `DesignIntentV1`, `DesignContextManifestV1` (structural inventory, semantic
availability rows, measured surface). No model call, no data read.

Column-name matching normalizes both sides: casefold, strip, replace spaces/hyphens with
underscores. Rules apply in order; first match wins (recorded in `match_trace`):

1. `intent_candidate` → critical: the column is named in any intent `ConceptProposalV1`
   `candidate_columns` (treatment, outcome, population, comparator, unit, timeframe, or a
   mandatory concept).
2. `evidenced_meaning` → plausible_adjustment: the manifest lists an `evidenced` semantic
   `meaning` slot for the column.
3. `flagged_by_profile` → plausible_adjustment: the measured surface carries a
   `missing_sentinel` or `identifier` hypothesis for the column (identifier hypotheses stay
   tier-2 so unit/time identity can be established; they are batch-grouped last).
4. `profiled_only` → supporting: the column appears in the measured surface with none of the
   above.
5. `no_signal` → unused: everything else (including columns whose profile shows a constant
   value when that fact is available in the measured surface).

Tier-2 is deliberately generous (over-inclusion costs one batched model glance; wrong exclusion
is recorded and blocks approval through the deferral rule — the D-039 wave discussion).

Batching (`build_batches`): critical columns first, one batch per ⌈n/limit⌉ preserving inventory
order, then plausible-adjustment columns grouped into remaining batches; at most 8 batches
total (SC §7 fan-out row "design semantics"); batch size balances to keep batches ≤
⌈(n_critical + n_plausible)/8⌉ + 1 columns. `batch_id` = `tb:{sha16 of table + sorted member
list}` (deterministic, D-031 style).

## 4. Method-pack registry (§13)

`registries/method-packs.v1.json`: `{"registry_version": "method-packs.v1", "packs": [...]}`,
four packs with `method_id` ∈ {`randomized_experiment`, `aipw`, `did`, `sharp_rdd`},
`pack_version` `<method>-pack.v1`. Per pack, exactly these keys (values from PRD-002
§13.1–§13.4; declarative only):

`method_id`, `pack_version`, `display_name`, `compatible_assignment_mechanisms` (e.g.
randomized/self_selected/policy_cutoff/time_of_adoption), `required_roles`, `optional_roles`
(RoleName values), `forbidden_adjustment_roles` (e.g. mediator, collider, post-treatment
descendants → `mediator`, `collider`), `supported_estimands` (rct: itt default + per_protocol
justified; aipw: ate, att; did: att_group_time_aggregate; rdd: late_at_cutoff),
`required_context_requirement_ids` (ids from context-requirements.v1.json),
`structural_requirements` (closed vocabulary strings, e.g. `one_row_per_unit`,
`unit_time_or_group_time_rows`, `running_variable_with_cutoff`, `treatment_binary`,
`arms_ge_2`), `allowed_prerepair_diagnostic_ids` (ids named in §13.x "Pre-repair diagnostics
may inspect …"; vocabulary fixed here, implementations land in T-012),
`eligibility_rule_vocabulary`, `imputation_forbidden_roles` (always includes treatment,
outcome), `imputation_eligible_roles`, `deletion_impact_dimensions`,
`invalidation_rules` (strings), `required_postrepair_diagnostic_ids`,
`required_visual_evidence_ids`, `reserved_estimator_id`, `runnable_frame_schema`
(= `runnable-frame-contract.v1`).

`design/packs.py`: `load_method_packs(path) -> MethodPackRegistry` — pydantic-validated rows
(`MethodPackV1` model lives here), fail-closed `RegistryError`-style error on wrong version,
duplicate method_id, count != 4, unknown RoleName, or a diagnostic id missing from the §13
vocabulary constant. `get(method_id)` raises stable `unsupported_method` code when absent.

## 5. Requirement templates

`registries/context-requirements.v1.json`: `registry_version` `context-requirements.v1`; rows
are ContextRequirementV1 templates minus instance fields — keys: `requirement_id` (dotted, e.g.
`design.assignment_mechanism`, `column.measurement_timing`, `column.missing_meaning`,
`dataset.sampling_mechanism`, `design.table_grain`, `design.treatment_meaning`,
`design.outcome_window`, `design.population_comparator`, `design.unit_identity`,
`column.meaning`, `column.encoding`, `design.treatment_descendants`,
`design.selection_variables`, `design.concept_mapping`, `design.conflict_resolution` — cover
every §10.1 bullet; one row each), `scope_kind`, `fact_required`, `why_required`,
`criticality`, `acceptable_evidence_types`, `required_support`, `methods_required_for` (subset
of the four method ids, or all), `missing_action`, `expected_answer_schema`, `user_may_know`.
Loader lives in `packs.py` (`load_requirement_templates`) returning validated
`ContextRequirementV1`-compatible templates (a small `RequirementTemplateV1` model; instance
fields `attempted_evidence`/`decisions_blocked` are added at instantiation time by later
tasks).

## 6. Entry gate and manifest compiler (`design/entry.py`)

Pure functions + one small store-facing class; no LangGraph, no events (the T-013 coordinator
owns events/state):

- `EntryError(code)` codes: `handoff_unavailable`, `entry_validation_failed`,
  `NO_ANALYSIS_CSV`, `UNSUPPORTED_ANALYSIS_FORMAT_V1`, `MULTI_TABLE_REQUIRED` (the last is
  raised by later question analysis, declared here in the vocabulary).
- `list_csv_candidates(catalog_reader, dataset_id) -> tuple[CsvCandidate, ...]` — admitted
  (`parse_status = 'parsed'`) resources with csv media type from `catalog.resources`.
- `resolve_selection(candidates, decision | None)`:
  exactly one candidate → `TableSelectionV1` (selection_source only_candidate);
  multiple + no decision → typed `SelectionRequired` result carrying the candidate list
  (the durable interrupt is T-013's job); multiple + decision → validate the decision names a
  candidate and build `TableSelectionV1` (selection_source user_decision, decision artifact
  parent); zero → `EntryError(NO_ANALYSIS_CSV)`; a decision naming a non-CSV admitted resource →
  `EntryError(UNSUPPORTED_ANALYSIS_FORMAT_V1)`.
- `validate_entry(...)`: PRD-002 §5 conditions 1–5 against the intake outcome payload, catalog
  row hashes, and registry versions; condition 6 (pinning) is recorded into the manifest.
- `compile_manifest(...) -> DesignContextManifestV1`: structural inventory from
  `catalog.structural_manifest` view rows for the selected table; semantic
  available/missing from the two catalog views; measured/provenance surfaces from their views;
  `retrieval_surfaces` = the five PRD-001 view names; `registry_versions` from the loaded
  registries (artifact_types, field_classes, method_packs, requirements, tools, validators
  `design-validators.v1`, capacity `delivery-capacity.v1`, graph `design-graph.v1`, schema
  `design-schemas.v1`); `recipient_map` per SC §5.4 rows (task kind → allowed tool ids from
  design-tools.v1.json).
- DB access: a narrow `CatalogReader` Protocol (methods it needs, e.g. `resources(dataset_id)`,
  `view_rows(view_name, dataset_id)`) implemented over psycopg in the same module; dockerized
  tests use the real catalog, unit tests use fakes.

## 7. Tool surface (`design/tools.py`)

- `registries/design-tools.v1.json`: `registry_version` `design-tools.v1`; rows
  `{tool_id, allowed_task_kinds: [...], registered: bool}` for all eleven §15 tools; the three
  harness-only tools have `allowed_task_kinds: []`; the three T-012 compute tools are listed
  with `registered: false` until T-012 flips them.
- `ToolRouter(registry, handlers, emitter, clock)`: `call(envelope, tool_id, arguments) ->
  ToolResult`; enforcement order: tool listed → registered → task-kind allowlisted → present in
  `envelope.allowed_tool_ids` → handler invoked. Violations emit `tool.denied` and raise
  `ToolError("tool_denied")` without calling the handler; handler exceptions emit `tool.failed`;
  success emits `tool.started`/`tool.completed`. Events via `build_event`, stage `design`,
  component `design-harness`, `required_eval_ids=("EV-SYS-002",)`, deterministic event ids
  `evt:{envelope.envelope_id}:tool:{n}`.
- Retrieval handlers (bounded, ID-addressed, over `CatalogReader` + a `ProductsReader` Protocol
  for committed payloads):
  - `list_intake_inventory(table_name)` → the manifest's own bounded rows (structural,
    available, missing, measured, provenance) for that table — served from the compiled
    manifest, not a new query, so workers see exactly the frozen surface.
  - `get_semantic_evidence(evidence_ids)` → items from the committed EvidenceBundle payload
    filtered to the requested ids ∩ envelope.allowed_evidence_ids; unknown ids reported in the
    result's `missing` list, never invented.
  - `get_measured_facts(table_name, column_names)` → per-column facts from the committed
    TableProfile payload (bounded: the profile's column dict entries).
  - `get_provenance(artifact_id)` → envelope fields of a committed artifact (type, hashes,
    parents, producer, locator) via `ProductsReader`; reader-permission is design-harness per
    the registry.
  - `get_method_contract(method_id)` → the validated pack row from `packs.py`.
- No handler returns raw rows, archives, or unrestricted documents; every result is a dict of
  scalars/lists already bounded by PRD-001 surfaces.

## 8. Tests

- `test_packs.py`: four packs load; unknown role/diagnostic id/duplicate/miscount fail closed;
  requirement templates load and every §10.1 bullet has a row; every pack's
  `required_context_requirement_ids` resolve into the requirement registry.
- `test_triage.py`: golden fixture (structural inventory + intent + availability rows) →
  expected tiers and match_trace; disjoint/exhaustive enforcement; batch cap 8; batch
  determinism (same input → same batch_ids); generous tier-2 (evidenced meaning without intent
  mention lands tier-2); deferred = supporting ∪ unused; registry row resolves.
- `test_entry.py`: candidate listing (dockerized against seeded catalog rows); zero/one/many
  routing incl. decision validation and format refusal; manifest compilation from seeded views
  (dockerized, reusing T-008-style fixtures) — structural inventory matches, registry_versions
  closed keys, recipient_map matches SC §5.4.
- `test_tools.py`: unit tests with fake readers/handlers — allowlist enforcement order,
  tool.denied on unlisted/unregistered/wrong-task-kind/absent-from-envelope, tool.failed on
  handler exception, bounded results, get_semantic_evidence missing-id reporting.

## 9. Budgets and acceptance

Design additions target: packs ≤ 130, triage ≤ 160, entry ≤ 300, tools ≤ 230 (design ≤ ~1420
total after task). Declarative: method packs ≈ 420, requirements ≈ 180, tools ≈ 40 (declarative
stays ≤ 3,000). Tests ≤ 700 new lines. Acceptance identical to T-009 §8 (suite, ruff,
MYPYPATH=tools mypy --strict, budget gate, ledger + checkpoint commit).

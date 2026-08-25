# T-012 — Validation walls, ask gate, pre-repair diagnostics, delivery capacity

Status: frozen for implementation
Owning PRD: PRD-002 (§10, §11, §13.5, §14, §16.3, §16.4; SC §6, §11)
Depends on: T-011

## 1. Deliverables

1. `registries/design-validation-rules.v1.json` — declarative rule rows for the walls that are
   expressible as data (see §3). Registry version `design-validators.v1` (the id already pinned
   in the manifest's `registry_versions`).
2. `registries/delivery-capacity.v1.json` — visualization catalog + capacity registry
   (versions `visualization-catalog.v1` / `delivery-capacity.v1`).
3. `src/causal/design/validators.py` — walls 1–7 (§16.3) as one engine + typed issues.
4. `src/causal/design/askgate.py` — requirement freeze/dedup/gate/packet/answer validation
   (SC §6.2; PRD-002 §11).
5. `src/causal/design/diagnostics.py` — polars read-only diagnostic engine (PRD-002 §4.2, §14).
6. `src/causal/design/capacity.py` — `DeliveryCapacityCheckV1` builder (PRD-002 §13.5).
7. Tools registry flip: `validate_causal_model`, `run_preflight_diagnostic`,
   `preview_eligibility_impact` become `registered: true` in `registries/design-tools.v1.json`,
   with handlers provided here (wired through the T-011 `ToolRouter`).
8. Tests: `tests/design/test_validators.py`, `test_askgate.py`, `test_diagnostics.py`,
   `test_capacity.py`.

Wall 8 (graph-view fidelity) ships with the renderer in T-013; walls 9–10 are capacity (here)
and approval (T-013).

## 2. Validation engine (`validators.py`)

- `ValidationIssueV1` (frozen model): `code` (stable), `json_path`, `rule_id`,
  `artifact_ids: tuple[Identity, ...]`, `allowed_actions: tuple[Identity, ...]`
  (`revise_field`, `add_evidence`, `relabel_epistemic_status`, `remove_claim`,
  `request_context`), `user_resolvable: bool` (§16.4).
- `ValidationReport`: `wall` (1–7), `issues: tuple[...]`, `passed` property.
- `validate(wall, task_kind, payload_model, result: AgentTaskResultV1, ctx: ValidationContext)`
  — `ValidationContext` (frozen) carries: manifest, triage record, committed parents by id,
  evidence-id set, requirement templates, pack registry, role ledger/causal context when
  present. Walls run in §16.3 order; the first failing wall stops (targeted correction needs
  one wall's issues).
- Wall 1 shape: `payload_model.model_validate(result.payload)`; pydantic errors map to issues
  (`code="shape_invalid"`, json_path from the error locs).
- Wall 2 references: every table/column id ∈ manifest inventory; every evidence id ∈ the
  admitted evidence set or user-answer artifacts; every parent id committed; every requirement
  id ∈ registry; claims cannot cite the claim's own id (defense in depth on top of the model
  rule).
- Wall 3 evidence: for each claim whose predicate matches a `blocking` requirement template's
  fact scope, `epistemic_status == evidenced` requires supporting evidence whose class is in
  the template's `acceptable_evidence_types` (evidence class resolution: `ev:doc/` and
  `ev:kaggle/` ids → `source_statement`/`data_dictionary` per a small prefix map;
  `ua:` ids → `user_confirmation`; measured pointers → `measured_observation`);
  `model_hypothesis` support can never satisfy a blocking template (PRD-002 §10.2).
- Wall 4 temporal: declarative rows (registry kind `temporal`): a claim/role with
  `timing == post_treatment` cannot carry role ∈ {confounder_candidate, precision_covariate,
  assignment_variable}; `mediator` requires post_treatment timing; outcome timing cannot be
  pre_treatment; running_variable requires pre_treatment or concurrent.
- Wall 5 causal: selected graph (base edges + selected alternative overrides) must be acyclic
  over `evidenced` + `hypothesis` edges (`code="graph_cycle"`); every role claim's
  `graph_edge_ids` resolve; role/graph consistency rows (registry kind `role_graph`):
  confounder_candidate ⇒ edges concept→treatment and concept→outcome present (any status);
  mediator ⇒ treatment→concept and concept→outcome; collider ⇒ treatment→concept and
  outcome→concept (or two cause-edges into it from T-side and Y-side);
  instrument_candidate ⇒ concept→treatment present and concept→outcome absent among
  evidenced edges; `disputed` edges must appear in ≥1 alternative (material alternatives stay
  visible).
- Wall 6 method: selected pack's `required_roles` all present in the ledger with status ≠
  unknown; no `forbidden_adjustment_roles` member inside the design's adjustment set (adjustment
  set = confounder_candidate + precision_covariate claims marked for the method);
  `structural_requirements` checked against manifest/measured facts via a small closed map
  (`treatment_binary` → treatment column facts show ≤2 non-null levels when available;
  `one_row_per_unit`/`unit_time_or_group_time_rows`/`running_variable_with_cutoff`/`arms_ge_2`
  → presence of the corresponding roles + diagnostics requested); every
  `required_context_requirement_ids` entry resolved or explicitly `retain_as_sensitivity`.
- Wall 7 frame: RFC vs pack + design: `exclusion_reason_vocabulary ⊇` pack vocabulary;
  imputation lists disjoint, pack's `imputation_forbidden_roles` columns (via measurement map)
  all in `imputation_forbidden`; `deletion_impact_dimensions ⊇` pack's; required diagnostics ⊇
  pack's `required_postrepair_diagnostic_ids`; `required_visual_evidence ⊇` pack's;
  `experiment_design_hash` matches the design parent; key_columns/grain non-empty and key
  columns exist in inventory; `estimator_input_schema == pack.reserved_estimator_id` input id.
- Declarative rows in `design-validation-rules.v1.json`: kinds `temporal`, `role_graph`,
  `frame_subset` (field-pair superset checks), `method_structural` — each row: `rule_id`,
  `wall`, `kind`, `params`, `code`, `allowed_actions`, `user_resolvable`. The engine interprets
  rows; walls 1–2 and graph algorithms stay in code. Loader fails closed on unknown kind/wall.
- Correction bound bookkeeping lives with the coordinator (T-013); validators are pure.

## 3. Ask gate (`askgate.py`)

- `RequirementState` StrEnum: `open`, `resolved`, `unknown_accepted`, `refused` (matches
  migration CHECK).
- `freeze_requirements(collected: Iterable[ContextRequirementV1]) -> tuple[...]` — dedup by
  `(requirement_id, scope_id)` merging `decisions_blocked` and `attempted_evidence` unions.
- `gate(requirements, evidence_index, round_number) -> GateDecision` where `GateDecision`
  routes each requirement per SC §6.2: `resolved` (an attempted evidence row satisfies the
  template's acceptable classes), `record_sensitivity` (supporting, or blocking with
  missing_action retain_as_sensitivity), `ask` (blocking + user_may_know + ask_user + schema
  declared + all non-user sources exhausted — attempted_evidence non-empty), `terminal`
  (blocking + not user-answerable → needs_context/refused by missing_action). A known
  `empty/not_offered/unreadable/withheld` availability status counts as exhausted, never
  re-queried (PRD-002 §11).
- `build_packet(asks, design_revision, round_number) -> UserQuestionPacketV1` — ≤5 questions
  (deterministic priority: blocking design-scope first, then column scope by inventory order;
  overflow stays open for round 2), question ids `q:{requirement_id}`, packet id
  `qp:{design_revision}:{round}`. Round 3 is never built: caller enforces ≤2 (§11).
- `validate_answers(packet, answer: UserContextAnswerV1) -> tuple[AnswerOutcome, ...]` — every
  packet question answered exactly once (missing → issue), `unknown` allowed always; value
  answers checked against the requirement's `expected_answer_schema` id via a closed schema map
  (`free_text`, `choice:<a|b|c>`, `iso_date`, `boolean`, `column_name`); returns per-requirement
  outcome (resolved-by-user / unknown → route by missing action).
- Persistence of requirement state uses a narrow `RequirementStore` Protocol (upsert/state
  transition rows in `design.context_requirements`); psycopg implementation included; the
  T-013 coordinator owns transactions and events.

## 4. Diagnostics (`diagnostics.py`)

- Engine over the pinned polars: `run_diagnostic(spec: DiagnosticSpec, frame_source:
  FrameSource, params) -> DiagnosticResultV1`. `FrameSource` Protocol: `scan() ->
  polars.LazyFrame` + `csv_artifact_ref` + `content hash`; implementation
  `CsvObjectFrameSource` reads the committed CSV bytes from object storage into polars
  (`read_csv`; memory-bounded via projected columns). Never writes.
- `DiagnosticSpec` rows are code-level (a dict constant): `diagnostic_id`, `primitive`,
  `default_params`, `methods` (from the T-011 pack vocabulary — every id in the packs'
  `allowed_prerepair_diagnostic_ids` must have a spec row; enforced by a test).
- Primitives (closed set, each ≤ 40 lines): `count_by(columns)` (group sizes incl. overall,
  arm counts, group-time cells, cutoff sides via `side = value >= cutoff`), `missing_share
  (target, by)` , `uniqueness(key_columns)` (duplicate count, distinct count),
  `level_profile(column, max_levels)` (level counts, sparsity), `numeric_support(column,
  cutoff?)` (min/max/quantiles, distance-to-cutoff histogram, mass points = top repeated
  values share), `availability(columns)` (non-null share per column). Feasibility-flavored ids
  map to primitives that report the raw inputs (e.g. `power_precision_feasibility` →
  `count_by(arms)` + outcome availability); no diagnostic passes judgment — values + warnings
  only.
- Every result records columns_read, total/used rows, unused reason counts
  (`null_excluded`, `parse_failed`), row_set_hash only when row selection occurred (sorted
  row-index hash), `implementation_version` `design-diagnostics.v1`; `partial` when some
  requested column is missing; `not_computable` when required roles/columns absent
  (issue recorded by caller per §7.1 — the pack declares which are non-blocking).
- Tool handlers: `run_preflight_diagnostic(envelope, {diagnostic_id, params})` (method-design
  task only, enforced by the T-011 router) and `preview_eligibility_impact(envelope,
  {rules})` → read-only counts under proposed eligibility rules: closed rule grammar
  `{column, op ∈ {eq, ne, ge, le, not_null, in}, value}` conjunction, returns kept/excluded
  counts by treatment/group when those roles are known. `validate_causal_model(envelope,
  {causal_context_payload})` → wall-5 issues via the engine (causal-synthesis task).

## 5. Capacity (`capacity.py` + `delivery-capacity.v1.json`)

- Registry: `visualization_catalog_version`, `capacity_registry_version`, `templates`: rows
  `{template_id, visual_evidence_ids: [...], max_panels, max_series, max_labels,
  max_annotations}` (at least one template covering every visual-evidence id used by the four
  packs — completeness enforced by test); `accessible_table_max_rows`, `max_concurrency: 8`.
- `check_capacity(method_pack, cardinalities: Mapping[str, int], required_visual_evidence) ->
  DeliveryCapacityCheckV1`: every required visual-evidence id must have ≥1 registered template
  whose limits fit the exact cardinalities (arms→series/panels, contrasts→labels/annotations,
  periods/event_times→series, cutoff_sides→panels, subgroups/cohorts→panels,
  evidence_items→accessible table rows); concurrency fields = min(8, registry); failures →
  `status=fail` with codes `no_template:<evidence_id>` / `over_limit:<template_id>:<dim>`;
  never invents a template (§13.5).

## 6. Tests

- `test_validators.py`: per wall, one passing and the named failing fixtures — shape error
  path mapping; unresolved reference (column/evidence/parent/requirement); blocking claim with
  hypothesis-only support rejected while supporting-criticality passes; each temporal row; a
  cyclic graph; each role_graph rule (confounder missing outcome edge etc.); disputed edge
  absent from alternatives; method wall missing required role + forbidden adjustment member +
  unresolved requirement; frame wall subset/hash/imputation failures. Declarative loader fails
  closed on unknown kind. Use compact fixture builders; parameterize by rule row.
- `test_askgate.py`: dedup merge; each gate route (resolved via acceptable evidence class;
  sensitivity; ask allowed only when all five SC §6.2 conditions hold — one test per violated
  condition; terminal by missing action); packet bounds (>5 overflow ordering, round 2, round
  cap), answer validation (missing answer, wrong schema, unknown, resolves + routes).
- `test_diagnostics.py`: golden CSV fixture (tests/design/data or inline `io.BytesIO`) through
  each primitive: arm counts, missing share by group, duplicate keys, level sparsity, cutoff
  support/mass points, availability; denominators/used-unused accounting exact; partial and
  not_computable statuses; determinism (same input → identical DiagnosticResultV1 including
  row_set_hash); every pack diagnostic id has a spec row; polars never writes (no write API
  used — assert FrameSource contract via fake recording reads only).
- `test_capacity.py`: pass and both failure codes; catalog covers all pack visual-evidence
  ids; registry loader fails closed; concurrency capped at 8.

## 7. Budgets and acceptance

Design additions target: validators ≤ 300, askgate ≤ 150, diagnostics ≤ 260, capacity ≤ 90
(design projected ≈ 2,220/2,500 after task — the remaining ~280 plus any T-011/T-012 underrun
is T-013's envelope; every module keeps one-line docstrings and data-driven rules to protect
it). Declarative additions ≈ 260. Tests ≤ 820 new lines. Acceptance identical to T-009 §8.

# T-009 — Design substrate: shared task envelope, design contracts, registry rows, design schema

Status: frozen for implementation
Owning PRD: PRD-002 (SC §5.2, §6.1, §11, §16, §21; PRD-002 §6, §10, §12, §16–§18)
Depends on: T-002, T-004, T-005, T-008
Wave: first task of the PRD-002 wave (D-039)

## 1. Deliverables

1. `src/causal/shared/envelope.py` — the cross-stage model-task contract (SC §5.2, §6.1, §16.2):
   `AgentTaskEnvelopeV1`, `AgentTaskResultV1`, `TaskStatus`, `TaskBudgets`, `ToolReceiptV1`,
   `ClaimV1`, `CausalFrameV1`, `EpistemicStatus`, `SupportClass`, `EvidenceClass`,
   `ContextRequirementV1`, `AttemptedEvidenceV1`, plus the small enums they need.
   Shared scope: these are used by PRD-002 through PRD-005 (SC §5.4).
2. `src/causal/design/contracts.py` — selection, manifest, intent, and interrupt payload models.
3. `src/causal/design/semantics.py` — semantic card, concepts, measurement map, causal context,
   role ledger, role-evidence models.
4. `src/causal/design/frame.py` — diagnostics, experiment design, runnable-frame contract, graph
   view, capacity check, design outcome models.
5. `registries/artifact-types.v1.json` — append the seventeen design artifact-type rows (§4).
6. `migrations/0004_design.sql` — the `design` schema per PRD-002 §21 (§5).
7. Tests under `tests/design/` (new package) and `tests/shared/test_envelope.py`.

No LangGraph, no model calls, no tools, no coordinator behavior in this task. This task lays the
typed substrate the rest of the wave builds on.

## 2. Conventions (binding)

- Pydantic v2, `ConfigDict(frozen=True, extra="forbid", strict=True)` exactly as
  `src/causal/shared/contracts.py`; reuse `Identity`, `Sha256Hex`, `UtcTimestamp`, `ArtifactRef`
  from there — do not redefine them.
- Payload models carry no wall-clock and no envelope identity fields (D-031); the envelope layer
  owns identity and time. `schema_version` literals live on each payload model.
- Enums are `StrEnum` with lowercase values unless the PRD names an exact casing.
- Every model exposes `canonical_payload()` only when it is a committed artifact payload; interrupt
  decision payloads (CLI-submitted) also get it. Plain row/fragment models do not need it.
- No single global confidence number anywhere (PRD-002 §12.1); support lives per claim/slot.
- Module budget targets: envelope ≤ 210, contracts ≤ 300, semantics ≤ 300, frame ≤ 330 logical
  lines. Function limit 75. These are targets inside the SC §14.1 gate, not new budgets.

## 3. Model inventory (field lists are binding; names may not drift)

### 3.1 `shared/envelope.py`

- `TaskStatus`: `complete`, `needs_context`, `conflict`, `refused`.
- `EpistemicStatus`: `evidenced`, `hypothesis`, `disputed`, `unknown`.
- `SupportClass` (SC/PRD-002 §10.2): `direct_user_confirmation`, `direct_source_statement`,
  `corroborated_source_inference`, `measured_observation`, `model_hypothesis`, `conflicting`,
  `unknown`.
- `EvidenceClass` = acceptable-evidence vocabulary for requirements: `user_confirmation`,
  `data_dictionary`, `study_protocol`, `source_statement`, `timestamp_relationship`,
  `measured_observation`.
- `Criticality`: `blocking`, `supporting`. `MissingAction`: `ask_user`, `retain_as_sensitivity`,
  `refuse`. `RequirementScopeKind`: `dataset`, `table`, `column`, `concept`, `relationship`,
  `design`. `SupportRequirement`: `direct`, `direct_or_corroborated`, `any_acceptable`.
- `CausalFrameV1`: `treatment`, `outcome`, `population`, `timeframe` (all `Identity`).
- `ClaimV1` (SC §16.2): `claim_id`, `subject_kind`, `subject_id`, `predicate`, `value`
  (JSON scalar/str), `epistemic_status`, `supporting_evidence_ids`, `contrary_evidence_ids`
  (tuples), `support_class`, `alternatives: tuple[str, ...]`, `causal_frame: CausalFrameV1 | None`.
  An agent inference cannot cite itself: validator rejects `claim_id` ∈ its own evidence ids.
- `AttemptedEvidenceV1`: `evidence_id`, `availability_status` (`Identity`; intake statuses pass
  through verbatim).
- `ContextRequirementV1` (SC §6.1): `requirement_id`, `registry_version`, `scope_kind`
  (`RequirementScopeKind`), `scope_id`, `fact_required`, `why_required`,
  `decisions_blocked: tuple[Identity, ...]`, `criticality`, `acceptable_evidence_types:
  tuple[EvidenceClass, ...]`, `required_support: SupportRequirement`,
  `methods_required_for: tuple[Identity, ...]`, `attempted_evidence:
  tuple[AttemptedEvidenceV1, ...]`, `user_may_know: bool`, `expected_answer_schema: Identity`,
  `missing_action`.
- `TaskBudgets`: `token_budget` (>0), `tool_call_budget` (≥0), `transient_attempt_budget`
  (default 3), `correction_budget` (default 2).
- `ToolReceiptV1`: `tool_id`, `call_index` (≥0), `status` (`completed`/`denied`/`failed`),
  `error_code: Identity | None`.
- `AgentTaskEnvelopeV1` (SC §5.2): `envelope_id`, `schema_version` = `agent-task-envelope.v1`,
  `analysis_id`, `stage_run_id`, `task_id`, `attempt_id`, `context_manifest: ArtifactRef`,
  `task_kind`, `scope_kind`, `scope_ids: tuple[Identity, ...]`,
  `parent_artifacts: tuple[ArtifactRef, ...]`, `allowed_evidence_ids`, `allowed_retrieval_ids`,
  `allowed_tool_ids` (tuples of `Identity`), `output_schema_version`, `validator_version`,
  `prompt_version`, `model_profile_version`, `budgets: TaskBudgets`,
  `allowed_stopping_states: tuple[TaskStatus, ...]`, `error_vocabulary: tuple[Identity, ...]`,
  `forbidden_payload_classes: tuple[Identity, ...]`, `payload_type: Identity`,
  `payload: dict[str, object]`. `canonical_payload()` provided (envelope hash gates tracing).
- `AgentTaskResultV1` (SC §5.2, PRD-002 §16.1): `envelope_id`, `schema_version` =
  `agent-task-result.v1`, `task_id`, `status: TaskStatus`, `artifact_type`,
  `artifact_schema_version`, `parent_artifact_ids: tuple[Identity, ...]`,
  `payload: dict[str, object]`, `claims: tuple[ClaimV1, ...]`,
  `missing_requirements: tuple[ContextRequirementV1, ...]`, `conflicts: tuple[str, ...]`,
  `warnings: tuple[str, ...]`, `evidence_ids: tuple[Identity, ...]`,
  `tool_receipts: tuple[ToolReceiptV1, ...]`, `output_hash: Sha256Hex | None`,
  `validation_target: Identity`.

### 3.2 `design/contracts.py`

- `SelectionSource`: `only_candidate`, `user_decision`. `QuestionKind`: `causal`, `predictive`,
  `descriptive`, `exploratory`. `InterruptKind`: `table_selection`, `clarification`, `approval`.
  `ApprovalDecision`: `approved`, `changes_requested`, `declined`. `AnswerKind`: `value`,
  `unknown`.
- `TableSelectionV1` (`table-selection.v1`): `dataset_id`, `logical_name`,
  `resource_object_locator`, `resource_sha256: Sha256Hex`, `media_type` (must be `text/csv`),
  `candidate_count` (≥1), `selection_source`, `decision_artifact_id: Identity | None`.
- `StructuralFieldV1`: `table_name`, `column_name`, `dtype`, `ordinal` (≥0).
- `AvailabilityRowV1`: `scope_kind` (`dataset`/`table`/`column`), `table_name | None`,
  `column_name | None`, `field_or_slot_name`, `status: Identity`, `evidence_count` (≥0),
  `json_pointer`.
- `DesignContextManifestV1` (`design-context-manifest.v1`, SC §5.1): `design_revision` (≥1),
  `question_artifact: ArtifactRef`, `intake_outcome_artifact: ArtifactRef`,
  `table_selection_artifact: ArtifactRef`, `selected_table: Identity`,
  `structural_inventory: tuple[StructuralFieldV1, ...]`,
  `semantic_available: tuple[AvailabilityRowV1, ...]`,
  `semantic_missing: tuple[AvailabilityRowV1, ...]`,
  `measured_surface: tuple[AvailabilityRowV1, ...]`,
  `provenance_surface: tuple[AvailabilityRowV1, ...]`,
  `retrieval_surfaces: tuple[Identity, ...]`,
  `registry_versions: dict[str, Identity]` (closed keys: `artifact_types`, `field_classes`,
  `method_packs`, `requirements`, `tools`, `validators`, `capacity`, `graph`, `schema`),
  `recipient_map: dict[str, tuple[Identity, ...]]` (task kind → allowed surface ids).
- `ConceptProposalV1`: `name`, `description`, `candidate_columns: tuple[Identity, ...]`.
- `DesignIntentV1` (`design-intent.v1`): `question_kind`, `causal_claim`, `intended_decision`,
  `treatment`, `outcome`, `population`, `comparator`, `unit`, `timeframe` (each
  `ConceptProposalV1`), `candidate_grain`, `mandatory_concepts: tuple[ConceptProposalV1, ...]`,
  `claims: tuple[ClaimV1, ...]`.
- `QuestionItemV1`: `question_id`, `requirement_ids` (min 1), `question_text`, `why_it_matters`,
  `blocked_decisions`, `expected_answer_schema`, `allow_unknown: bool` (must be True).
- `UserQuestionPacketV1` (`user-question-packet.v1`): `packet_id`, `design_revision`,
  `round_number` (1 or 2), `questions: tuple[QuestionItemV1, ...]` (1–5).
- `AnswerItemV1`: `question_id`, `answer_kind`, `value: str | None` (None iff `unknown`).
- `UserContextAnswerV1` (`user-context-answer.v1`): `packet_id`, `answers` (min 1),
  `provenance: Literal["user"]`.
- `TableSelectionDecisionV1` (`table-selection-decision.v1`): `interrupt_id`,
  `expected_interrupt_hash: Sha256Hex`, `expected_revision` (≥1), `selected_table: Identity`,
  `idempotency_key`.
- `DesignApprovalDecisionV1` (`design-approval-decision.v1`): `interrupt_id`,
  `expected_interrupt_hash: Sha256Hex`, `expected_revision` (≥1), `decision: ApprovalDecision`,
  `approved_artifacts: tuple[ArtifactRef, ...]` (min 1 when `approved`, else empty),
  `change_requests: tuple[str, ...]` (min 1 when `changes_requested`, else empty),
  `idempotency_key`.

### 3.3 `design/semantics.py`

- `TimingClass`: `pre_treatment`, `concurrent`, `post_treatment`, `unknown`.
- `SlotAssertionV1`: `value: str | None`, `status: EpistemicStatus`,
  `evidence_ids: tuple[Identity, ...]`.
- `COLUMN_CARD_SLOTS` (closed tuple): `meaning`, `concept`, `entity`, `kind`, `units`, `scale`,
  `levels`, `encoding`, `timing`, `measurement_window`, `missing_interpretation`,
  `source_process`.
- `ColumnSemanticCardV1` (`column-semantic-card.v1`): `table_name`, `column_name`,
  `display_name`, `concept_id: Identity | None`, `timing: TimingClass`,
  `slots: dict[str, SlotAssertionV1]` (keys must equal `COLUMN_CARD_SLOTS` exactly),
  `claims: tuple[ClaimV1, ...]`, `alternatives: tuple[str, ...]`,
  `conflicts: tuple[str, ...]`.
- `MeasurementRelation`: `measures`, `proxies`, `derived_proposed`. `ConceptStatus`:
  `observed`, `proxy_measured`, `unmeasured`.
- `MeasurementLinkV1`: `concept_id`, `table_name`, `column_name`, `relation`, `notes: str`.
- `ConceptV1`: `concept_id`, `name`, `description`, `status: ConceptStatus`.
- `MeasurementMapV1` (`measurement-map.v1`): `concepts: tuple[ConceptV1, ...]` (min 1),
  `links: tuple[MeasurementLinkV1, ...]`, `claims: tuple[ClaimV1, ...]`.
- `CausalEdgeV1` (PRD-002 §12.2): `edge_id`, `source_concept_id`, `target_concept_id`,
  `timeframe`, `mechanism_summary`, `supporting_evidence_ids`, `contrary_evidence_ids`,
  `status: EpistemicStatus`, `differing_alternative_ids: tuple[Identity, ...]`.
- `GraphAlternativeV1`: `alternative_id`, `label`, `edges: tuple[CausalEdgeV1, ...]`.
- `CausalContextV1` (`causal-context.v1`): `frame: CausalFrameV1`,
  `concept_ids: tuple[Identity, ...]` (min 2), `edges: tuple[CausalEdgeV1, ...]`,
  `alternatives: tuple[GraphAlternativeV1, ...]`, `selection_notes: str`,
  `claims: tuple[ClaimV1, ...]`.
- `RoleName` (PRD-002 §12.4, exactly 17): `treatment`, `outcome`, `unit_identifier`, `time`,
  `assignment_variable`, `group`, `cluster`, `stratum`, `running_variable`,
  `confounder_candidate`, `mediator`, `collider`, `instrument_candidate`, `effect_modifier`,
  `selection_variable`, `precision_covariate`, `excluded_from_design`, `unknown`.
- `RoleClaimV1`: `role: RoleName`, `concept_id`, `column_refs: tuple[Identity, ...]`,
  `evidence_ids`, `timing: TimingClass`, `graph_edge_ids: tuple[Identity, ...]`,
  `support_class: SupportClass`, `alternatives: tuple[str, ...]`,
  `status: EpistemicStatus`, `methods: tuple[Identity, ...]`.
- `RoleLedgerV1` (`role-ledger.v1`): `frame: CausalFrameV1`,
  `claims: tuple[RoleClaimV1, ...]` (min 1).
- `RoleEvidenceV1` (`role-evidence.v1`, worker output): `assigned_scope: tuple[Identity, ...]`
  (min 1), `edge_hypotheses: tuple[CausalEdgeV1, ...]`,
  `role_hypotheses: tuple[RoleClaimV1, ...]`, `competing_mechanisms: tuple[str, ...]`,
  `claims: tuple[ClaimV1, ...]`.

### 3.4 `design/frame.py`

- `DiagnosticStatus`: `computed`, `partial`, `not_computable`.
- `DiagnosticResultV1` (PRD-002 §14): `diagnostic_id`, `diagnostic_version`,
  `status: DiagnosticStatus`, `csv_artifact: ArtifactRef`,
  `columns_read: tuple[Identity, ...]`, `total_rows` (≥0), `used_rows` (≥0),
  `unused_reason_counts: dict[str, int]`, `row_set_hash: Sha256Hex | None`,
  `values: dict[str, float | int | str | bool | None]`, `warnings: tuple[str, ...]`,
  `implementation_version: Identity`.
- `PreRepairFeasibilityReportV1` (`pre-repair-feasibility-report.v1`): `method_id`,
  `results: tuple[DiagnosticResultV1, ...]` (min 1).
- `ExperimentDesignV1` (`experiment-design.v1`, PRD-002 §17): `causal_question`,
  `intended_decision`, `selected_csv: ArtifactRef`, `method_id`, `method_pack_version`,
  `rejected_methods: dict[str, str]` (method id → reason; exactly the three others),
  `frame: CausalFrameV1`, `comparator`, `unit`, `estimand: Identity`,
  `measurement_map: ArtifactRef`, `causal_context: ArtifactRef`, `role_ledger: ArtifactRef`,
  `assumptions: tuple[str, ...]`, `identification_risks: tuple[str, ...]`,
  `eligibility_rules: tuple[str, ...]`, `mandatory_repair_boundaries: tuple[str, ...]`,
  `forbidden_repair_boundaries: tuple[str, ...]`,
  `imputation_eligible_columns: tuple[Identity, ...]`,
  `imputation_forbidden_columns: tuple[Identity, ...]`,
  `deletion_impact_dimensions: tuple[Identity, ...]`,
  `invalidation_conditions: tuple[str, ...]`,
  `required_prerepair_diagnostics: tuple[Identity, ...]`,
  `required_postrepair_diagnostics: tuple[Identity, ...]`,
  `required_visual_evidence: tuple[Identity, ...]`,
  `primary_contrasts: tuple[str, ...]`, `multiplicity_policy: str | None`,
  `capacity_check: ArtifactRef | None`, `visualization_catalog_version: Identity`,
  `capacity_registry_version: Identity`, `sensitivity_requirements: tuple[str, ...]`,
  `registry_versions: dict[str, Identity]` (same closed keys as the manifest).
- `RunnableFrameContractV1` (`runnable-frame-contract.v1`, PRD-002 §18): `selected_csv:
  ArtifactRef`, `output_grain`, `key_columns: tuple[Identity, ...]` (min 1),
  `required_roles: tuple[RoleName, ...]`, `allowed_roles: tuple[RoleName, ...]`,
  `forbidden_roles: tuple[RoleName, ...]`, `type_constraints: dict[str, Identity]`,
  `uniqueness_constraints: tuple[str, ...]`, `eligibility_rules: tuple[str, ...]`,
  `exclusion_reason_vocabulary: tuple[Identity, ...]`,
  `treatment_missingness_rule`, `outcome_missingness_rule`,
  `method_structure: dict[str, Identity]`,
  `imputation_permitted: tuple[Identity, ...]`, `imputation_forbidden: tuple[Identity, ...]`,
  `required_missingness_indicators: tuple[Identity, ...]`,
  `deletion_impact_dimensions: tuple[Identity, ...]`,
  `revision_required_conditions: tuple[str, ...]`,
  `feasibility_gates: tuple[str, ...]`, `required_final_diagnostics: tuple[Identity, ...]`,
  `estimator_input_schema: Identity`, `experiment_design_hash: Sha256Hex`.
- `GraphNodeViewV1`: `concept_id`, `label`, `status: ConceptStatus`,
  `roles: tuple[RoleName, ...]`. `GraphEdgeViewV1`: `edge_id`, `source_concept_id`,
  `target_concept_id`, `status: EpistemicStatus`.
- `CausalGraphViewV1` (`causal-graph-view.v1`, PRD-002 §12.3): `parents: tuple[ArtifactRef, ...]`
  (min 1), `nodes: tuple[GraphNodeViewV1, ...]` (min 2), `edges: tuple[GraphEdgeViewV1, ...]`,
  `selected_alternative_id: Identity | None`, `layout_direction` (`TB`/`LR`),
  `renderer_profile: Identity`, `legend_text`, `disclosure_text`, `spec_hash: Sha256Hex`,
  `svg: str`, `accessible_summary: str`, `node_edge_table: str`, `renderer_version`,
  `theme_version`, `validator_version`, `validation_status: Identity`.
- `CapacityStatus`: `pass`, `fail`.
- `DeliveryCapacityCheckV1` (`delivery-capacity-check.v1`, SC §11): `method_id`,
  `method_profile_id`, `cardinalities: dict[str, int]` (closed keys: `arms`, `contrasts`,
  `subgroups`, `cohorts`, `periods`, `event_times`, `cutoff_sides`, `series`, `evidence_items`),
  `required_visual_evidence: tuple[Identity, ...]`,
  `compatible_templates: tuple[Identity, ...]`, `template_limits: dict[str, int]`,
  `accessible_table_capacity: int` (≥1), `execution_concurrency: int` (1–8),
  `render_concurrency: int` (1–8), `status: CapacityStatus`,
  `failure_codes: tuple[Identity, ...]` (empty iff `pass`),
  `visualization_catalog_version`, `capacity_registry_version`, `method_registry_version`.
- `DesignOutcomeStatus` (PRD-002 §6): `approved`, `needs_context`, `changes_requested`,
  `declined`, `refused`, `failed_observability`, `failed`.
- `DesignOutcomeV1` (`design-outcome.v1`): `status: DesignOutcomeStatus`, `design_revision` (≥1),
  `refusal_code: Identity | None` (required for `refused`; vocabulary includes
  `UNSUPPORTED_ANALYSIS_FORMAT_V1`, `MULTI_TABLE_REQUIRED`, `NO_ANALYSIS_CSV`,
  `NOT_IDENTIFIABLE`, `UNSUPPORTED_DESIGN`), `error_code: Identity | None`,
  `experiment_design: ArtifactRef | None`, `runnable_frame_contract: ArtifactRef | None`,
  `causal_graph_view: ArtifactRef | None`, `capacity_check: ArtifactRef | None`,
  `approval: ArtifactRef | None`, `open_requirement_ids: tuple[Identity, ...]`,
  `clarification_rounds_used: int` (0–2), `counts: dict[str, int]` (closed keys:
  `columns_triaged`, `columns_carded`, `deferred_columns`, `role_tasks`, `corrections`).
  Cross-field rule: `approved` requires the four handoff refs non-None.

## 4. Registry rows (append to `registries/artifact-types.v1.json`)

Seventeen new rows, all `producer_component: "design-harness"`, `sensitivity_class: "internal"`,
`terminal_statuses: ["committed"]` unless noted. Reader/destination/parents:

| artifact_type | schema_version | readers | required parents | optional parents | destinations |
|---|---|---|---|---|---|
| TableSelection | table-selection.v1 | design-harness | IntakeOutcome, QuestionRecord | TableSelectionDecision | design |
| DesignContextManifest | design-context-manifest.v1 | design-harness | TableSelection | — | design |
| DesignIntent | design-intent.v1 | design-harness | DesignContextManifest | UserContextAnswer | design |
| ColumnSemanticCard | column-semantic-card.v1 | design-harness | DesignIntent | UserContextAnswer | design |
| MeasurementMap | measurement-map.v1 | design-harness | DesignIntent | ColumnSemanticCard | design |
| RoleEvidence | role-evidence.v1 | design-harness | MeasurementMap | — | design |
| CausalContext | causal-context.v1 | design-harness | MeasurementMap | RoleEvidence, UserContextAnswer | design |
| RoleLedger | role-ledger.v1 | design-harness | CausalContext | — | design |
| PreRepairFeasibilityReport | pre-repair-feasibility-report.v1 | design-harness | RoleLedger | — | design |
| ExperimentDesign | experiment-design.v1 | design-harness, preparation-harness, estimation-harness, presentation-coordinator | RoleLedger | PreRepairFeasibilityReport, UserContextAnswer | design, preparation, estimation, presentation |
| CausalGraphView | causal-graph-view.v1 | design-harness, preparation-harness, presentation-coordinator | RunnableFrameContract | CausalContext, MeasurementMap, RoleLedger, ExperimentDesign | design, preparation, presentation |
| DeliveryCapacityCheck | delivery-capacity-check.v1 | design-harness, estimation-harness, presentation-coordinator | CausalGraphView | ExperimentDesign, RunnableFrameContract | design, estimation, presentation |
| UserQuestionPacket | user-question-packet.v1 | design-harness, cli | DesignContextManifest | — | design |
| UserContextAnswer | user-context-answer.v1 | design-harness | UserQuestionPacket | — | design |
| TableSelectionDecision | table-selection-decision.v1 | design-harness | IntakeOutcome | — | design |
| DesignApproval | design-approval.v1 | design-harness, preparation-harness | ExperimentDesign | RunnableFrameContract, CausalGraphView, DeliveryCapacityCheck | design, preparation |
| DesignOutcome | design-outcome.v1 | design-harness, preparation-harness | IntakeOutcome | TableSelection, DesignIntent, ExperimentDesign, RunnableFrameContract, CausalGraphView, DeliveryCapacityCheck, DesignApproval | design, preparation |

`terminal_statuses` exceptions: DesignOutcome uses the seven §6 statuses; DesignApproval uses
`["approved", "changes_requested", "declined"]`.

Validator versions: `<kebab-type>-validator.v1`. Required parents follow the D-027 lesson: the
nearest parent that exists on every committable path; the coordinator enforces the longer chain.
The existing RunnableFrameContract row is unchanged.

## 5. Migration `0004_design.sql` (PRD-002 §21)

`CREATE SCHEMA design;` plus eight tables (all FK `analysis_id`-free; artifact IDs are text and
reference `causal.artifacts(artifact_id)` where noted):

- `design_runs`: `stage_run_id` PK, `analysis_id`, `graph_thread_id`, `design_revision` int,
  `state` text CHECK in the §4 run states, `selected_table` text NULL, `method_id` text NULL,
  `outcome_artifact_id` text NULL FK, `observability_failure` text NULL, `created_at`,
  `updated_at` timestamptz; UNIQUE (`analysis_id`, `design_revision`).
- `design_artifact_refs`: `artifact_id` PK FK, `analysis_id`, `design_revision`, `kind`,
  `content_hash`, `schema_version`, `approval_bound` boolean default false.
- `design_context_manifests`: `artifact_id` PK FK, `analysis_id`, `design_revision`,
  `content_hash`, `intake_outcome_artifact_id`, `registry_versions` jsonb.
- `causal_graph_views`: `artifact_id` PK FK, `analysis_id`, `design_revision`, `renderer_version`,
  `validation_status`.
- `design_tasks`: `task_id` PK, `analysis_id`, `stage_run_id`, `design_revision`, `task_kind`,
  `scope` jsonb, `envelope_hash`, `prompt_version`, `model_profile_version`, `status` text CHECK
  (`dispatched`,`complete`,`needs_context`,`conflict`,`refused`,`failed`), `output_artifact_id`
  text NULL FK, `attempts` int default 1; UNIQUE (`analysis_id`, `design_revision`, `task_id`).
- `context_requirements`: `requirement_key` PK (`analysis_id` + `design_revision` +
  `requirement_id` composite via PRIMARY KEY (...)), `scope_kind`, `scope_id`, `criticality`
  CHECK, `missing_action` CHECK, `state` CHECK (`open`,`resolved`,`unknown_accepted`,
  `refused`), `attempted_evidence` jsonb, `resolving_answer_artifact_id` text NULL FK.
- `delivery_capacity_checks`: `artifact_id` PK FK, `analysis_id`, `design_revision`, `status`
  CHECK (`pass`,`fail`), `cardinalities` jsonb, `capacity_registry_version`,
  `visualization_catalog_version`.
- `design_approvals`: `artifact_id` PK FK, `analysis_id`, `design_revision`, `decision` CHECK,
  `approved_hashes` jsonb, `decided_at` timestamptz.

No views in this migration (design views, if any, come with the tasks that need them).

## 6. Tests

- `tests/shared/test_envelope.py`: round-trip + canonical hash stability for envelope/result;
  strict rejection of extra fields; claim self-citation rejection; budgets bounds; requirement
  enum enforcement; UserQuestionPacket 1–5 bound lives in design tests.
- `tests/design/test_contracts.py`: per-model happy path + one boundary failure each (packet >5
  questions, card slot key mismatch, approval decision cross-field rules, DesignOutcome
  approved-requires-refs rule, capacity failure_codes iff fail, RFC hash field).
- `tests/design/test_registry_rows.py`: registry loads; all seventeen rows resolve through
  `causal.shared.registry`; parent-type names all exist in the registry; validator-version format.
- `tests/design/test_migration.py` (dockerized, reuses `tests/conftest.py` fixtures): migration
  applies after 0001–0003; insert one row per table; CHECK/UNIQUE constraints enforced (state,
  decision, duplicate revision).

Budget targets: tests ≤ 550 new logical lines. Follow `tests/intake/` idiom; docker tests marked
like the existing integration tests.

## 7. Amendments

Amendment 1 (pre-acceptance errata, D-044):

1. §3.3 header said "exactly 17" `RoleName` values; the binding list has 18 (17 roles plus
   `unknown`). The list stands.
2. §3.4 prose said an `approved` outcome requires "the four handoff refs"; the binding rule is
   the five listed optional refs (`experiment_design`, `runnable_frame_contract`,
   `causal_graph_view`, `capacity_check`, `approval`) non-None, as implemented.
3. §4 gains an eighteenth row: `DesignApprovalDecision` / `design-approval-decision.v1`
   (producer design-harness, readers design-harness, required parent ExperimentDesign, optional
   RunnableFrameContract/CausalGraphView/DeliveryCapacityCheck, terminal `committed`,
   destination design). SC §11.1 commits the typed CLI decision verbatim; without a row that
   commit would fail closed at the registry. `DesignApproval` (the harness's hash-binding
   approval record) adds `DesignApprovalDecision` to its optional parents.

## 8. Acceptance

1. `uv run pytest` green (full suite).
2. `uv run ruff check .` and `uv run mypy --strict src tests` clean.
3. `tools/budget_check.py --task-id T-009 --base <pre-task commit> --worktree` status
   `within_budget` or `warning`; no scope over its ceiling.
4. Ledger updated (task row, checkpoint) and committed with the code.

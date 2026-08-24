# T-004 — Artifact-type registry and declarative registrations

Status: ACCEPTED (single-session, D-008)
Contract sections: SYSTEM-CONTRACT §3.1 (registry and routing ledger),
§15 criterion 2.

## Files

- `src/causal/shared/registry.py`
- `registries/artifact-types.v1.json`
- `tests/shared/test_registry.py`
- `tools/budget_check.py` + `tests/tools/test_budget_check.py`
  (only: add `registries/` to the declarative scope, D-009)

## Decisions

- D-009: `registries/` is a new top-level declarative-scope directory
  (SYSTEM-CONTRACT §14.1 "static registries").
- D-010: static registries are JSON, not YAML — parseable by the standard
  library; the pinned stack has no YAML parser and none is added.
- D-011: canonical component IDs: `intake-coordinator`, `design-harness`,
  `preparation-harness`, `estimation-harness`, `presentation-coordinator`,
  `cli`, `runtime`.
- D-012: `registries/artifact-types.v1.json` is append-only. Individual
  registrations are immutable once committed. This task seeds ONLY the §3.1
  rows whose required-parent names are exact in the contract
  (`IntakeOutcome`, `RunnableFrameContract`); the remaining entry rows and
  all stage-internal rows are appended by their stage tasks after the owning
  PRD is read, so no parent-type name is invented here.

## Deliverable — `registry.py`

1. `ArtifactTypeRegistrationV1` (frozen, extra=forbid, strict):
   `artifact_type: Identity`, `schema_version: Identity`,
   `producer_component: Identity`,
   `allowed_reader_components: tuple[Identity, ...]` (min 1),
   `required_parent_types: tuple[Identity, ...]`,
   `optional_parent_types: tuple[Identity, ...]`,
   `sensitivity_class: SensitivityClass`,
   `terminal_statuses: tuple[Identity, ...]` (min 1),
   `destinations: tuple[Identity, ...]`,
   `validator_version: Identity`.
2. `RegistryError(ValueError)` with stable `code`: `unsupported_schema`
   (missing/ambiguous lookup, §3.1) | `duplicate_registration` |
   `invalid_registry_file`.
3. `ArtifactTypeRegistry`: built from an iterable of registrations;
   duplicate `artifact_type` at load → `duplicate_registration`;
   `lookup(artifact_type)` returns the registration or raises
   `unsupported_schema`; exposes `registry_version`.
4. `load_artifact_type_registry(path: Path) -> ArtifactTypeRegistry`: reads
   `{"registry_version": "artifact-types.v1", "registrations": [...]}`;
   malformed JSON or shape → `invalid_registry_file`.

## Deliverable — `registries/artifact-types.v1.json`

Two seeded rows per §3.1 and D-011/D-012:

- `IntakeOutcome`: producer `intake-coordinator`; readers `design-harness`;
  required parents `QuestionRecord`, `SourceManifest`, `TableProfile`,
  `EvidenceBundle`, `SemanticMap`; sensitivity `internal`; terminal statuses
  `usable`, `partial`, `refused`; destinations `design`.
- `RunnableFrameContract`: producer `design-harness`; readers
  `preparation-harness`, `estimation-harness`; required parents
  `ExperimentDesign`; sensitivity `internal`; terminal statuses `approved`,
  `superseded`; destinations `preparation`, `estimation`.

## Tests

Load the real registry file and look both rows up; missing type raises
`unsupported_schema` with its code; duplicate registration raises at load;
malformed file raises `invalid_registry_file`; registration model rejects
extra fields and empty reader/status tuples; budget checker assigns
`registries/artifact-types.v1.json` to the declarative scope.

## Budget

`registry.py` ≤ 150 logical lines; registry JSON counts declarative;
tests ≤ 200; budget-checker delta ≤ 10.

"""Task table, prompt rendering, envelope assembly, deterministic MeasurementMap (T-013 §1, §6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from causal.design.compile import (
    build_task_envelope,
    compile_measurement_map,
    load_task_table,
    render_prompt,
)
from causal.design.contracts import ConceptProposalV1, DesignIntentV1, QuestionKind
from causal.design.packs import TASK_KINDS, PackRegistryError
from causal.design.semantics import (
    COLUMN_CARD_SLOTS,
    ColumnSemanticCardV1,
    ConceptStatus,
    ConceptV1,
    MeasurementLinkV1,
    MeasurementMapV1,
    MeasurementRelation,
    SlotAssertionV1,
    TimingClass,
)
from causal.shared.contracts import ArtifactRef
from causal.shared.envelope import AgentTaskEnvelopeV1, EpistemicStatus, TaskStatus

ROOT = Path(__file__).resolve().parents[2]
TABLE = load_task_table(ROOT / "registries" / "design-tasks.v1.json")
REF = ArtifactRef(artifact_id="manifest-1", content_hash="a" * 64)
TABLE_NAME = "nsw.csv"
MEASURES = MeasurementRelation.MEASURES
PROXIES = MeasurementRelation.PROXIES
UNKNOWN_SLOT = SlotAssertionV1(value=None, status=EpistemicStatus.UNKNOWN, evidence_ids=())

# task kind: artifact type, schema version, highest wall, allowed stopping states (T-013 §1.1).
EXPECTED_ROWS = {
    "intent": ("DesignIntent", "design-intent.v1", 3, ("complete", "needs_context")),
    "semantic_batch": ("ColumnSemanticCard", "column-semantic-card.v1", 4,
                       ("complete", "needs_context")),
    "role_evidence": ("RoleEvidence", "role-evidence.v1", 4,
                      ("complete", "needs_context", "conflict")),
    "causal_synthesis": ("CausalContext", "causal-context.v1", 5,
                         ("complete", "needs_context", "conflict")),
    "method_design": ("ExperimentDesign", "experiment-design.v1", 7,
                      ("complete", "needs_context", "refused")),
}


def rows() -> list[dict[str, Any]]:
    return [json.loads(spec.model_dump_json()) for spec in TABLE.values()]


def write_table(tmp_path: Path, table_rows: list[dict[str, Any]],
                version: str = "design-tasks.v1") -> Path:
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"registry_version": version, "tasks": table_rows}),
                    encoding="utf-8")
    return path


def proposal(name: str, *columns: str) -> ConceptProposalV1:
    return ConceptProposalV1(name=name, description=f"{name} concept", candidate_columns=columns)


def card(column: str, concept_id: str) -> ColumnSemanticCardV1:
    return ColumnSemanticCardV1(
        table_name=TABLE_NAME, column_name=column, display_name=column, concept_id=concept_id,
        timing=TimingClass.UNKNOWN, slots=dict.fromkeys(COLUMN_CARD_SLOTS, UNKNOWN_SLOT),
        claims=(), alternatives=(), conflicts=(),
    )


def link(concept_id: str, column: str, relation: MeasurementRelation) -> MeasurementLinkV1:
    return MeasurementLinkV1(concept_id=concept_id, table_name=TABLE_NAME, column_name=column,
                             relation=relation,
                             notes="direct" if relation is MEASURES else "proxy")


def proposed(concept_id: str, name: str, status: ConceptStatus) -> ConceptV1:
    return ConceptV1(concept_id=concept_id, name=name, description=f"{name} concept", status=status)


INTENT = DesignIntentV1(
    question_kind=QuestionKind.CAUSAL, causal_claim="training raises earnings",
    intended_decision="fund the programme", treatment=proposal("Training program", "treat"),
    outcome=proposal("Earnings", "re78", "re74"), population=proposal("Eligible adults"),
    comparator=proposal("No training"), unit=proposal("Person"), timeframe=proposal("1978"),
    candidate_grain="one row per person",
    mandatory_concepts=(proposal("Prior earnings", "re74"),), claims=(),
)
CARDS = (card("treat", "c:training_program"), card("re78", "c:earnings"),
         card("re74", "c:earnings_1974"))
EXPECTED_MAP = MeasurementMapV1(
    concepts=(
        proposed("c:1978", "1978", ConceptStatus.UNMEASURED),
        proposed("c:earnings", "Earnings", ConceptStatus.OBSERVED),
        ConceptV1(concept_id="c:earnings_1974", name="c:earnings_1974",
                  description="named by a validated column semantic card",
                  status=ConceptStatus.OBSERVED),
        proposed("c:eligible_adults", "Eligible adults", ConceptStatus.UNMEASURED),
        proposed("c:no_training", "No training", ConceptStatus.UNMEASURED),
        proposed("c:person", "Person", ConceptStatus.UNMEASURED),
        proposed("c:prior_earnings", "Prior earnings", ConceptStatus.PROXY_MEASURED),
        proposed("c:training_program", "Training program", ConceptStatus.OBSERVED),
    ),
    links=(
        link("c:earnings", "re74", PROXIES),
        link("c:earnings", "re78", MEASURES),
        link("c:earnings_1974", "re74", MEASURES),
        link("c:prior_earnings", "re74", PROXIES),
        link("c:training_program", "treat", MEASURES),
    ),
    claims=(),
)


class TestTaskTable:
    def test_holds_exactly_the_five_task_kinds(self) -> None:
        assert sorted(TABLE) == sorted(TASK_KINDS)

    @pytest.mark.parametrize("task_kind", TASK_KINDS)
    def test_row_carries_its_output_contract_and_budgets(self, task_kind: str) -> None:
        spec = TABLE[task_kind]
        artifact_type, schema_version, wall, states = EXPECTED_ROWS[task_kind]
        assert (spec.output_artifact_type, spec.output_schema_version) == (artifact_type,
                                                                          schema_version)
        assert spec.wall == wall
        assert spec.allowed_stopping_states == tuple(TaskStatus(state) for state in states)
        assert (spec.token_budget, spec.tool_call_budget, spec.correction_budget) == (24576, 8, 2)

    @pytest.mark.parametrize("task_kind", TASK_KINDS)
    def test_prompt_template_exists_and_names_its_version(self, task_kind: str) -> None:
        spec = TABLE[task_kind]
        text = (ROOT / spec.prompt_path).read_text(encoding="utf-8")
        assert spec.prompt_path.startswith("prompts/design/")
        assert spec.prompt_version in text

    def test_missing_file_fails_closed(self, tmp_path: Path) -> None:
        with pytest.raises(PackRegistryError) as caught:
            load_task_table(tmp_path / "absent.json")
        assert caught.value.code == "invalid_registry_file"

    @pytest.mark.parametrize("version", ["design-tasks.v2", ""])
    def test_wrong_registry_version_fails_closed(self, tmp_path: Path, version: str) -> None:
        with pytest.raises(PackRegistryError) as caught:
            load_task_table(write_table(tmp_path, rows(), version))
        assert caught.value.code == "invalid_registry_file"

    def test_unknown_task_kind_fails_closed(self, tmp_path: Path) -> None:
        table_rows = rows()
        table_rows[0]["task_kind"] = "estimation"
        with pytest.raises(PackRegistryError) as caught:
            load_task_table(write_table(tmp_path, table_rows))
        assert caught.value.code == "invalid_registry_file"

    def test_duplicate_row_fails_closed(self, tmp_path: Path) -> None:
        table_rows = rows()
        with pytest.raises(PackRegistryError) as caught:
            load_task_table(write_table(tmp_path, [*table_rows, table_rows[0]]))
        assert caught.value.code == "duplicate_task"

    def test_missing_row_fails_closed(self, tmp_path: Path) -> None:
        with pytest.raises(PackRegistryError) as caught:
            load_task_table(write_table(tmp_path, rows()[1:]))
        assert caught.value.code == "unknown_task_kind"


SECTIONS: dict[str, object] = {"intent": {"question": "does training raise earnings?"},
                               "columns": ["treat"]}


class TestRenderPrompt:
    def test_appends_sections_in_sorted_key_order_after_the_template(self) -> None:
        text = render_prompt(TABLE["intent"], ROOT, SECTIONS)
        template = (ROOT / TABLE["intent"].prompt_path).read_text(encoding="utf-8")
        assert text.startswith(template)
        assert text.index("## columns") < text.index("## intent")
        assert '\n\n## columns\n[\n "treat"\n]' in text

    def test_is_deterministic_across_calls_and_key_insertion_order(self) -> None:
        reordered = dict(reversed(list(SECTIONS.items())))
        assert render_prompt(TABLE["intent"], ROOT, SECTIONS) == render_prompt(
            TABLE["intent"], ROOT, reordered)


class TestBuildTaskEnvelope:
    def envelope(self) -> AgentTaskEnvelopeV1:
        return build_task_envelope(
            TABLE["role_evidence"], analysis_id="an-1", stage_run_id="sr-1", task_id="t-1",
            attempt_id="a-1", manifest_ref=REF, scope_kind="concept",
            scope_ids=["c:earnings"], parent_artifacts=[REF], allowed_evidence_ids=["ev-1"],
            allowed_tool_ids=["get_semantic_evidence", "get_provenance"],
            payload_type="RoleEvidenceTaskContext", payload={"assigned_scope": ["c:earnings"]},
        )

    def test_carries_the_spec_contract_and_budgets(self) -> None:
        envelope = self.envelope()
        spec = TABLE["role_evidence"]
        assert envelope.task_kind == "role_evidence"
        assert envelope.prompt_version == spec.prompt_version
        assert envelope.output_schema_version == spec.output_schema_version
        assert envelope.allowed_stopping_states == spec.allowed_stopping_states
        assert envelope.budgets.token_budget == spec.token_budget
        assert envelope.budgets.tool_call_budget == spec.tool_call_budget
        assert envelope.budgets.correction_budget == spec.correction_budget

    def test_carries_the_tool_allowlist_and_closed_vocabularies(self) -> None:
        envelope = self.envelope()
        tools = ("get_semantic_evidence", "get_provenance")
        assert envelope.allowed_tool_ids == tools
        assert envelope.allowed_retrieval_ids == tools
        assert envelope.error_vocabulary == ("schema_invalid", "tool_denied",
                                             "correction_exhausted")
        assert envelope.forbidden_payload_classes == ("raw_rows", "dataframe", "archive_bytes",
                                                      "provider_response", "credentials")

    def test_round_trips_through_strict_validation(self) -> None:
        envelope = self.envelope()
        assert AgentTaskEnvelopeV1.model_validate_json(envelope.model_dump_json()) == envelope
        assert envelope.envelope_id == "env:t-1:a-1"
        assert envelope.validator_version == "design-validators.v1"
        assert envelope.model_profile_version == "vertex-model-profile.v1"


class TestMeasurementMap:
    def test_compiles_the_golden_concepts_and_links(self) -> None:
        assert compile_measurement_map(INTENT, CARDS) == EXPECTED_MAP

    def test_keeps_a_proposal_with_no_measuring_column_unmeasured(self) -> None:
        statuses = {c.concept_id: c.status for c in compile_measurement_map(INTENT, ()).concepts}
        assert set(statuses.values()) == {ConceptStatus.UNMEASURED}
        assert len(statuses) == 7

    def test_is_deterministic(self) -> None:
        first = compile_measurement_map(INTENT, CARDS)
        assert first.model_dump() == compile_measurement_map(INTENT, tuple(reversed(CARDS)))\
            .model_dump()

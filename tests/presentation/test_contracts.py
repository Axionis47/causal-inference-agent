# The presentation payload contracts: catalog validation, manifest and plan shape, figure-spec
# hash identity, and the closed outcome status set (T-030 §2; PRD-005 §1, §7.1, §8, §10, §13).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from causal.presentation import contracts as pc
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef

CATALOG_PATH = Path(__file__).resolve().parents[2] / "registries" / "visualization-catalog.v1.json"
DOCUMENT: dict[str, Any] = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def ref(name: str) -> ArtifactRef:
    return ArtifactRef(artifact_id=name, content_hash=content_hash({"id": name}))


def catalog(**over: Any) -> pc.VisualizationCatalogV1:
    return pc.VisualizationCatalogV1.model_validate_json(json.dumps(DOCUMENT | over))


def evidence(evidence_id: str = "primary_contrast_estimates") -> pc.EvidenceEntryV1:
    return pc.EvidenceEntryV1(
        visual_evidence_id=evidence_id, figure_data=ref(evidence_id),
        figure_data_schema_id="figure-data-artifact.v1",
        compatible_template_ids=("estimate_forest.v1",), quantities={"y": "effect"},
        units={"y": "rate"}, cardinality=2, suppression_state="none")


def manifest(**over: Any) -> pc.PresentationContextManifestV1:
    base: dict[str, Any] = {
        "parents": (ref("estimation_bundle"),), "versions": {"schema": "presentation.v1"},
        "analysis_id": "analysis-1", "stage_run_id": "presentation-run-1",
        "handoff_manifest": ref("handoff"), "inputs": {key: ref(key) for key in pc.ENTRY_KEYS},
        "causal_graph_view": ref("graph_view"),
        "approved": {"question_id": "q1", "estimand_id": "att",
                     "method_id": "randomized_experiment", "profile_id": "rct-profile.v1"},
        "claim_status": "reportable_with_qualifications", "statement_ids": ("s1",),
        "qualification_ids": ("q-attrition",),
        "required_evidence_ids": ("primary_contrast_estimates",), "evidence": (evidence(),),
        "display_profile": catalog().display_profile,
        "allowlists": {"curator": ("resolve_registered_layout_facts",)}}
    return pc.PresentationContextManifestV1(**(base | over))


def figure(figure_id: str = "figure-1") -> pc.FigureEntryV1:
    return pc.FigureEntryV1(
        figure_id=figure_id, template_id="estimate_forest.v1",
        visual_evidence_ids=("primary_contrast_estimates",), panel_groups=(("panel-1",),),
        choices={"scale_sharing": "shared", "labels": "left", "legends": "bottom"},
        annotation_ids=(), qualification_ids=("q-attrition",),
        text={"title": "How large is the effect?", "caption": "Arm B versus control.",
              "accessible_description": "One point and interval per contrast."})


def spec(**over: Any) -> pc.FigureSpecV1:
    axis = pc.AxisSpecV1(quantity_id="effect", unit_id="rate", title="Effect (rate)",
                         scale="linear", domain=(-0.5, 0.5))
    panel = pc.PanelSpecV1(
        panel_id="panel-1", question="How large is the effect?", mark="point",
        axes={"x": axis, "y": axis.model_copy(update={"scale": "band"})},
        encodings={"series": ("arm_b",), "fields": ("x_value",),
                   "uncertainty": ("interval_lower", "interval_upper"),
                   "reference_lines": ("null_effect",)})
    base: dict[str, Any] = {
        "parents": (ref("figure_plan"),), "versions": {"compiler": "figure-compiler.v1"},
        "figure_id": "figure-1", "template_id": "estimate_forest.v1", "panels": (panel,),
        "text": {"title": "Effect", "caption": "Arm B versus control.",
                 "accessible_description": "One point and interval per contrast."},
        "qualification_ids": ("q-attrition",), "logical_height": 320}
    return pc.FigureSpecV1(**(base | over))


class TestCatalog:
    def test_the_registered_catalog_round_trips(self) -> None:
        loaded = catalog()
        assert loaded.font_id == pc.FONT_ID and loaded.font_sha256 == pc.FONT_SHA256
        assert loaded.display_profile.display_profile_id == pc.DISPLAY_PROFILE_ID
        assert {row.method_id for row in loaded.profiles} == {
            "randomized_experiment", "aipw", "did", "sharp_rdd"}
        assert pc.VisualizationCatalogV1.model_validate_json(
            loaded.model_dump_json()) == loaded

    def test_a_template_with_an_absent_capacity_bound_is_invalid(self) -> None:
        templates = [dict(row) for row in DOCUMENT["templates"]]
        templates[0] = templates[0] | {
            "bounds": {k: v for k, v in templates[0]["bounds"].items() if k != "max_panels"}}
        with pytest.raises(ValidationError):
            catalog(templates=templates)

    @pytest.mark.parametrize("bound", [0, -1])
    def test_a_template_may_not_declare_an_unbounded_capacity(self, bound: int) -> None:
        templates = [dict(row) for row in DOCUMENT["templates"]]
        templates[0] = templates[0] | {
            "bounds": templates[0]["bounds"] | {"max_series_per_figure": bound}}
        with pytest.raises(ValidationError):
            catalog(templates=templates)

    def test_a_template_naming_an_unregistered_profile_is_invalid(self) -> None:
        templates = [dict(row) for row in DOCUMENT["templates"]]
        templates[0] = templates[0] | {"allowed_profile_ids": ["nonesuch-profile.v1"]}
        with pytest.raises(ValidationError):
            catalog(templates=templates)

    def test_a_profile_naming_a_template_that_cannot_answer_it_is_invalid(self) -> None:
        profiles = [dict(row) for row in DOCUMENT["profiles"]]
        profiles[0] = profiles[0] | {"templates_by_evidence": dict(
            profiles[0]["templates_by_evidence"]) | {"balance_overview": ["trend_lines.v1"]}}
        with pytest.raises(ValidationError):
            catalog(profiles=profiles)

    def test_a_catalog_may_not_widen_the_v1_ceilings(self) -> None:
        with pytest.raises(ValidationError):
            catalog(limits=DOCUMENT["limits"] | {"max_figures": 7})
        with pytest.raises(ValidationError):
            catalog(limits=DOCUMENT["limits"] | {"max_panels_per_figure": 4})


class TestManifestAndPlan:
    def test_the_manifest_round_trips(self) -> None:
        frozen = manifest()
        assert pc.PresentationContextManifestV1.model_validate_json(
            frozen.model_dump_json()) == frozen

    def test_a_manifest_missing_one_handoff_entry_is_invalid(self) -> None:
        with pytest.raises(ValidationError):
            manifest(inputs={key: ref(key) for key in pc.ENTRY_KEYS[:4]})

    def test_a_manifest_dropping_a_required_question_is_invalid(self) -> None:
        with pytest.raises(ValidationError):
            manifest(required_evidence_ids=("primary_contrast_estimates", "balance_overview"))

    def test_a_draft_over_the_figure_ceiling_is_invalid(self) -> None:
        figures = tuple(figure(f"figure-{index}") for index in range(pc.MAX_FIGURES + 1))
        with pytest.raises(ValidationError):
            pc.FigurePlanDraftV1(figures=figures, summary="Seven figures.")

    def test_a_draft_is_a_plan_or_one_typed_inability(self) -> None:
        refused = pc.FigurePlanDraftV1(inability_code="no_honest_template",
                                       implicated_evidence_ids=("balance_overview",))
        assert refused.figures == ()
        with pytest.raises(ValidationError):
            pc.FigurePlanDraftV1(figures=(figure(),), inability_code="no_honest_template")

    def test_an_accepted_plan_covers_distinct_committed_figures(self) -> None:
        plan = pc.FigurePlanV1(
            parents=(ref("manifest"),), versions={"schema": "figure-plan.v1"}, plan_id="plan-1",
            figures=(figure(),), coverage={"primary_contrast_estimates": "figure-1"},
            figure_data={"primary_contrast_estimates": ref("figure_data")}, summary="One figure.")
        assert pc.FigurePlanV1.model_validate_json(plan.model_dump_json()) == plan
        with pytest.raises(ValidationError):
            pc.FigurePlanV1(
                parents=(ref("manifest"),), versions={}, plan_id="plan-1",
                figures=(figure(), figure()), coverage={"primary_contrast_estimates": "figure-9"},
                figure_data={}, summary="One id twice and an uncovered question.")


class TestSpecAndOutcome:
    def test_the_spec_hash_is_deterministic_and_content_addressed(self) -> None:
        first, second = spec(), spec()
        assert first.spec_hash() == second.spec_hash()
        assert first.spec_hash() == content_hash(first.canonical_payload())
        assert spec(logical_height=321).spec_hash() != first.spec_hash()

    def test_a_spec_may_not_exceed_the_panel_ceiling(self) -> None:
        panel = spec().panels[0]
        with pytest.raises(ValidationError):
            spec(panels=tuple(panel.model_copy(update={"panel_id": f"panel-{index}"})
                              for index in range(pc.MAX_PANELS + 1)))

    @pytest.mark.parametrize("status", ["complete", "complete_with_qualifications"])
    def test_only_a_delivered_outcome_carries_a_bundle(self, status: str) -> None:
        outcome = pc.PresentationOutcomeV1(
            status=status, stage_run_id="presentation-run-1", context_manifest=ref("manifest"),
            presentation_bundle=ref("bundle"), error_code=None)
        assert pc.PresentationOutcomeV1.model_validate_json(
            outcome.model_dump_json()) == outcome
        with pytest.raises(ValidationError):
            pc.PresentationOutcomeV1(
                status="blocked", stage_run_id="presentation-run-1", context_manifest=None,
                presentation_bundle=ref("bundle"), error_code="blocked")

    @pytest.mark.parametrize("status", ["blocked", "needs_template", "needs_layout_revision",
                                        "failed", "failed_observability"])
    def test_every_refusing_status_returns_no_bundle(self, status: str) -> None:
        outcome = pc.PresentationOutcomeV1(
            status=status, stage_run_id="presentation-run-1", context_manifest=None,
            presentation_bundle=None, error_code="entry_validation_failed",
            detail_codes=("missing_artifact",))
        assert outcome.presentation_bundle is None

    def test_an_unregistered_status_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            pc.PresentationOutcomeV1(
                status="partially_complete", stage_run_id="run-1", context_manifest=None,
                presentation_bundle=None, error_code=None)

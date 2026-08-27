# The §13 compiler: one accepted plan and its frozen figure data become one FigureSpec and one
# Vega-Lite document per figure, with §11's honesty rules mechanical (T-031 §2).
# This module also holds the presentation fixtures the curator and renderer suites share.

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from causal.estimation.contracts import FigureDataArtifactV1, FigureDataPointV1
from causal.presentation import catalog as vc
from causal.presentation import compile as co
from causal.presentation import contracts as pc
from causal.shared.contracts import ArtifactRef

REPO = Path(__file__).resolve().parents[2]
CATALOG = vc.load_visualization_catalog(REPO / "registries" / "visualization-catalog.v1.json")
TEMPLATES = {row.template_id: row for row in CATALOG.templates}
RCT = vc.profile_for_method(CATALOG, "randomized_experiment")
FOREST, PRIMARY = "estimate_forest.v1", "primary_contrast_estimates"
# One row per registered template family: the profile, the question, and the fixture it draws.
FAMILIES = (("randomized_experiment", PRIMARY, FOREST),
            ("randomized_experiment", "assignment_attrition_flow", "composition_stack.v1"),
            ("randomized_experiment", "balance_overview", "balance_overview.v1"),
            ("did", "trends", "trend_lines.v1"),
            ("sharp_rdd", "binned_outcome_fit", "binned_scatter.v1"))
DESCRIBED = ("counts and effects in pp, treated against control, with the 95% interval, the "
             "null_effect, zero, denominator, cutoff, treatment_start and balance_threshold "
             "references, and every stated qualification")
FIELDS = ("series_id", "category", "x_value", "y_value", "interval_lower", "interval_upper",
          "denominator")
LINEAGE: dict[str, Any] = {"parents": (ArtifactRef(
    artifact_id="figure_plan", content_hash="0" * 64),), "versions": {"schema": "figure-spec.v1"}}


def digest(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def ref(name: str) -> ArtifactRef:
    return ArtifactRef(artifact_id=name, content_hash=digest(name))


def point(series: str, category: str | None = None, **over: Any) -> dict[str, Any]:
    return dict.fromkeys(FIELDS) | {"series_id": series, "category": category} | over


# One committed FigureDataArtifact, dumped the way the object store holds it (T-026 shape).
def figure_data(evidence_id: str, points: list[dict[str, Any]]) -> dict[str, Any]:
    return FigureDataArtifactV1(
        parents=(ref("estimation_plan"),), versions={"schema": "figure-data-artifact.v1"},
        visual_evidence_id=evidence_id, builder_id=f"{evidence_id}_builder",
        builder_version="figure-builder.v1", rule_ids=("suppression.v1",),
        points=tuple(FigureDataPointV1(**row) for row in points), units={"x": "pp", "y": "count"},
        labels={"x": "effect (pp)", "y": "group"}, contributing_counts={"total": 200},
        contribution_mask_hash=None, disclosure_status="reportable").model_dump(mode="json")


POINTS: dict[str, list[dict[str, Any]]] = {
    PRIMARY: [point("arm_b_vs_control", "primary", x_value=0.12, interval_lower=0.01,
                    interval_upper=0.23)],
    "required_sensitivities": [point("trim_1pct", "sens", x_value=0.1, interval_lower=0.0,
                                     interval_upper=0.21)],
    "assignment_attrition_flow": [point("arm_a", "enrolled", y_value=120.0, denominator=200),
                                  point("arm_b", "enrolled", y_value=118.0, denominator=200)],
    "balance_overview": [point("arm_a", "age", x_value=0.02), point("arm_b", "age", x_value=-0.01)],
    "trends": [point("treated", x_value=1.0, y_value=3.0), point("control", x_value=2.0,
               y_value=2.0), point("treated", "treatment_start", x_value=1.5)],
    "binned_outcome_fit": [point("below", x_value=-1.0, y_value=0.4, interval_lower=0.3,
                                 interval_upper=0.5),
                           point("above", x_value=1.0, y_value=0.7, interval_lower=0.6,
                                 interval_upper=0.8), point("above", "cutoff", x_value=0.0)]}
DATA = {name: figure_data(name, rows) for name, rows in POINTS.items()}


def choices(template_id: str, **over: Any) -> dict[str, str]:
    # Every enumerated choice a plan takes, read off the registered template itself.
    return {key: values[0] for key, values in TEMPLATES[template_id].choices.items()} | over


def entry(figure_id: str, template_id: str, evidence_ids: tuple[str, ...],
          **over: Any) -> pc.FigureEntryV1:
    return pc.FigureEntryV1(**{
        "figure_id": figure_id, "template_id": template_id, "visual_evidence_ids": evidence_ids,
        "panel_groups": tuple((name,) for name in evidence_ids), "choices": choices(template_id),
        "annotation_ids": (), "qualification_ids": (),
        "text": {"title": f"What does {evidence_ids[0]} show?",
                 "caption": f"Frozen {evidence_ids[0]} evidence.",
                 "accessible_description": DESCRIBED}} | over)


def plan(figures: tuple[pc.FigureEntryV1, ...]) -> pc.FigurePlanV1:
    return pc.FigurePlanV1(
        parents=(ref("presentation_context_manifest"),), plan_id="plan-1", figures=figures,
        versions={"schema": "figure-plan.v1"}, summary="the approved figures in evidence order",
        coverage={name: row.figure_id for row in figures for name in row.visual_evidence_ids},
        figure_data={name: ref(name) for row in figures for name in row.visual_evidence_ids})


def compiled(evidence_id: str, template_id: str, method_id: str = "randomized_experiment",
             data: dict[str, Any] | None = None) -> tuple[pc.FigureSpecV1, dict[str, Any]]:
    # One figure through the whole compiler: the committed specification and the document the
    # renderer draws from, both built from the same frozen fields.
    profile, rows = vc.profile_for_method(CATALOG, method_id), data or DATA
    row = entry(f"fig_{evidence_id}", template_id, (evidence_id,))
    spec = co.compile_figures(plan((row,)), CATALOG, profile, rows, LINEAGE)[0]
    return spec, co.figure_document(spec, row, TEMPLATES[template_id], profile, rows,
                                    co.panel_width(CATALOG.display_profile))


def nodes(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, dict):
        return [document] + [row for value in document.values() for row in nodes(value)]
    return [row for value in document for row in nodes(value)] if isinstance(document, list) else []


class TestCompiler:
    @pytest.mark.parametrize(("method_id", "evidence_id", "template_id"), FAMILIES)
    def test_every_template_family_compiles_from_fixture_figure_data(
            self, method_id: str, evidence_id: str, template_id: str) -> None:
        spec, document = compiled(evidence_id, template_id, method_id)
        panel = spec.panels[0]
        assert spec.template_id == template_id and set(panel.axes) == {"x", "y"}
        assert panel.encodings["series"] and panel.encodings["fields"]
        assert TEMPLATES[template_id].bounds.min_logical_height <= spec.logical_height
        assert document["$schema"].startswith("https://vega.github.io/schema/vega-lite/")
        # §9.2/§11.2: no transform, no remote asset, no second y-axis, no reversed scale.
        rows = nodes(document)
        assert not any("transform" in row or "url" in row or row.get("resolve") for row in rows)
        assert not any((row.get("scale") or {}).get("reverse") for row in rows)
        assert all(set(row.get("encoding", {})) <= {"x", "y", "x2", "y2", "color", "shape",
                                                    "strokeDash"} for row in rows)

    def test_identical_inputs_produce_an_identical_spec_hash(self) -> None:
        moved = dict(DATA) | {PRIMARY: figure_data(PRIMARY, [point(
            "arm_b_vs_control", "primary", x_value=0.44, interval_lower=0.4, interval_upper=0.5)])}
        assert compiled(PRIMARY, FOREST)[0].spec_hash() == compiled(PRIMARY, FOREST)[0].spec_hash()
        assert compiled(PRIMARY, FOREST)[0].spec_hash() != compiled(
            PRIMARY, FOREST, data=moved)[0].spec_hash()

    def test_a_domain_starts_at_zero_for_bars_and_covers_every_interval_and_reference(self) -> None:
        bars, _ = compiled("assignment_attrition_flow", "composition_stack.v1")
        magnitude = bars.panels[0].axes["y"].domain
        assert magnitude is not None and magnitude == (0.0, 200.0)
        panel = compiled(PRIMARY, FOREST)[0].panels[0]
        low, high = panel.axes["x"].domain or (1.0, 1.0)
        assert low <= 0.0 and high >= 0.23
        assert panel.encodings["reference_lines"] == ("null_effect",)
        assert panel.encodings["uncertainty"] == ("low", "high")

    def test_missing_and_suppressed_values_stay_distinct(self) -> None:
        data = dict(DATA) | {"assignment_attrition_flow": figure_data(
            "assignment_attrition_flow", [point("arm_a", "enrolled", y_value=1.0, denominator=200),
                                          point("arm_b", "withheld", denominator=9),
                                          point("arm_c", "unobserved")])}
        spec, document = compiled("assignment_attrition_flow", "composition_stack.v1", data=data)
        assert spec.panels[0].encodings["states"] == ("missing", "suppressed", "value")
        drawn = [row for node in nodes(document)
                 for row in (node.get("data") or {}).get("values", []) if "state" in row]
        assert {row["state"] for row in drawn} == {"missing", "suppressed", "value"}

    @pytest.mark.parametrize(("over", "code", "data"), [
        ({"template_id": "sankey.v9"}, co.UNKNOWN_TEMPLATE, DATA),
        ({}, co.MISSING_FIGURE_DATA, {})])
    def test_an_uncompilable_figure_is_refused(self, over: dict[str, Any], code: str,
                                               data: dict[str, Any]) -> None:
        broken = plan((entry("fig_1", FOREST, (PRIMARY,)).model_copy(update=over),))
        with pytest.raises(pc.PresentationError) as excinfo:
            co.compile_figures(broken, CATALOG, RCT, data, LINEAGE)
        assert excinfo.value.code == code

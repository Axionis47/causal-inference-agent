"""CausalGraphView rendering: DOT, legend, accessible text, fidelity (T-013 §1.4; PRD-002 §12.3)."""

from __future__ import annotations

import json
import shutil
from typing import Any
from xml.etree import ElementTree

import pytest

from causal.design import renderer
from causal.design.renderer import CausalGraphViewV1
from causal.design.semantics import (
    CausalContextV1,
    CausalEdgeV1,
    ConceptStatus,
    ConceptV1,
    GraphAlternativeV1,
    MeasurementMapV1,
    RoleClaimV1,
    RoleLedgerV1,
    RoleName,
    TimingClass,
)
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef
from causal.shared.envelope import CausalFrameV1, EpistemicStatus, SupportClass

# D-042: a missing Graphviz binary skips the full-render tests; nothing is ever substituted.
requires_dot = pytest.mark.skipif(shutil.which("dot") is None, reason="no local Graphviz binary")
FRAME = CausalFrameV1(treatment="ad_spend", outcome="sales", population="stores", timeframe="2024")
PARENTS = (ArtifactRef(artifact_id="causal-context-1", content_hash="a" * 64),)


def edge(edge_id: str, source: str, target: str, status: EpistemicStatus) -> CausalEdgeV1:
    return CausalEdgeV1(edge_id=edge_id, source_concept_id=source, target_concept_id=target,
                        timeframe="2024", mechanism_summary="", supporting_evidence_ids=(),
                        contrary_evidence_ids=(), status=status, differing_alternative_ids=())


def claim(role: RoleName, concept_id: str) -> RoleClaimV1:
    return RoleClaimV1(role=role, concept_id=concept_id, column_refs=(), evidence_ids=(),
                       timing=TimingClass.UNKNOWN, graph_edge_ids=(), alternatives=(), methods=(),
                       support_class=SupportClass.DIRECT_SOURCE_STATEMENT,
                       status=EpistemicStatus.EVIDENCED)


def concept(concept_id: str, status: ConceptStatus) -> ConceptV1:
    return ConceptV1(concept_id=concept_id, name=concept_id, description="", status=status)


# One small design carrying all four edge statuses and all three measurement statuses.
EDGES = (edge("e-spend-sales", "ad_spend", "sales", EpistemicStatus.EVIDENCED),
         edge("e-season-spend", "season", "ad_spend", EpistemicStatus.HYPOTHESIS),
         edge("e-season-sales", "season", "sales", EpistemicStatus.DISPUTED),
         edge("e-trust-sales", "brand_trust", "sales", EpistemicStatus.UNKNOWN))
ALTERNATIVE = GraphAlternativeV1(
    alternative_id="alt-collider", label="brand trust as a collider",
    edges=(edge("e-spend-trust", "ad_spend", "brand_trust", EpistemicStatus.HYPOTHESIS), EDGES[0]))
CONTEXT = CausalContextV1(
    frame=FRAME, concept_ids=("sales", "ad_spend", "season", "brand_trust"), edges=EDGES,
    alternatives=(ALTERNATIVE,), selection_notes="")
LEDGER = RoleLedgerV1(frame=FRAME, claims=(
    claim(RoleName.TREATMENT, "ad_spend"), claim(RoleName.OUTCOME, "sales"),
    claim(RoleName.CONFOUNDER_CANDIDATE, "season"), claim(RoleName.PRECISION_COVARIATE, "season"),
    claim(RoleName.COLLIDER, "brand_trust")))
MAP = MeasurementMapV1(links=(), concepts=(
    concept("ad_spend", ConceptStatus.OBSERVED), concept("sales", ConceptStatus.OBSERVED),
    concept("season", ConceptStatus.PROXY_MEASURED),
    concept("brand_trust", ConceptStatus.UNMEASURED)))


def spec_of(**kwargs: Any) -> renderer.GraphSpec:
    return renderer.build_graph_spec(CONTEXT, LEDGER, MAP, **kwargs)


def render(alternative: str | None = None, **kwargs: Any) -> CausalGraphViewV1:
    return renderer.render_causal_graph(context=CONTEXT, ledger=LEDGER, measurement_map=MAP,
                                        parents=PARENTS, selected_alternative_id=alternative,
                                        **kwargs)


def dot_line(dot: str, prefix: str) -> str:
    return next(line.strip() for line in dot.splitlines() if line.strip().startswith(prefix))


class TestSpecWithoutTheBinary:
    def test_nodes_are_sorted_and_labelled_with_seat_status_and_roles(self) -> None:
        nodes, _, _, _ = spec_of()
        assert [n.concept_id for n in nodes] == ["ad_spend", "brand_trust", "sales", "season"]
        labels = {n.concept_id: n.label for n in nodes}
        assert labels["ad_spend"] == "ad_spend [treatment] (observed) roles: treatment"
        assert labels["sales"] == "sales [outcome] (observed) roles: outcome"
        assert labels["season"] == "season (proxy) roles: confounder_candidate, precision_covariate"
        assert labels["brand_trust"] == "brand_trust (unmeasured) roles: collider"

    def test_the_frame_seats_are_boxed_and_unmeasured_concepts_are_dashed(self) -> None:
        _, _, _, dot = spec_of()
        assert "graph [rankdir=TB]" in dot
        assert dot_line(dot, "ad_spend [").endswith("shape=box]")
        assert dot_line(dot, "sales [").endswith("shape=box]")
        assert dot_line(dot, "brand_trust [").endswith("style=dashed]")
        assert "shape=box" not in dot_line(dot, "season [")

    def test_every_edge_status_gets_its_own_line_style(self) -> None:
        _, edges, _, dot = spec_of()
        assert dot_line(dot, "ad_spend -> sales") == "ad_spend -> sales [style=solid]"
        assert dot_line(dot, "season -> ad_spend") == "season -> ad_spend [style=dashed]"
        assert dot_line(dot, "season -> sales") == "season -> sales [style=dotted]"
        assert dot_line(dot, "brand_trust -> ") == 'brand_trust -> sales [label="?" style=dotted]'
        assert [e.edge_id for e in edges] == [
            "e-spend-sales", "e-trust-sales", "e-season-spend", "e-season-sales"]

    def test_the_spec_hash_is_stable_and_moves_with_the_layout(self) -> None:
        first, second = spec_of(), spec_of()
        assert first[2] == second[2] and first[3] == second[3]
        assert content_hash(first[2]) == content_hash(second[2])
        _, _, sideways, dot = spec_of(layout_direction="LR", renderer_profile="other.v1")
        assert "graph [rankdir=LR]" in dot and sideways["renderer_profile"] == "other.v1"
        assert content_hash(sideways) != content_hash(first[2])

    def test_an_alternative_replaces_the_whole_edge_set(self) -> None:
        _, edges, spec, dot = spec_of(alternative_id="alt-collider")
        assert [e.edge_id for e in edges] == ["e-spend-trust", "e-spend-sales"]
        assert spec["selected_alternative_id"] == "alt-collider"
        assert "season -> sales" not in dot

    def test_an_unknown_alternative_id_is_a_fidelity_fault(self) -> None:
        with pytest.raises(renderer.RendererError) as error:
            spec_of(alternative_id="alt-missing")
        assert error.value.code == "graph_view_infidelity"

    def test_a_concept_the_map_never_names_is_unmeasured(self) -> None:
        thin = MAP.model_copy(update={"concepts": MAP.concepts[:2]})
        nodes, _, _, _ = renderer.build_graph_spec(CONTEXT, LEDGER, thin)
        assert {n.concept_id: n.status for n in nodes}["season"] is ConceptStatus.UNMEASURED


def test_the_legend_names_every_status_as_text_never_as_color() -> None:
    for status in EpistemicStatus:
        assert status.value in renderer.LEGEND_TEXT
    for text in ("solid", "dashed", "dotted", "box", "(observed)", "(proxy)", "(unmeasured)",
                 "color carries none"):
        assert text in renderer.LEGEND_TEXT


class TestAccessibleAlternative:
    def test_the_summary_states_every_edge_with_its_status_in_sorted_order(self) -> None:
        nodes, edges, _, _ = spec_of()
        _, summary, _ = renderer._texts(CONTEXT, None, nodes, edges)
        assert summary == " ".join(["4 concepts and 4 causal edges.", *sorted([
            "ad_spend causes sales (evidenced).", "brand_trust causes sales (unknown).",
            "season causes ad_spend (hypothesis).", "season causes sales (disputed)."])])

    def test_the_table_carries_one_row_per_node_then_one_per_edge(self) -> None:
        nodes, edges, _, _ = spec_of()
        _, _, table = renderer._texts(CONTEXT, None, nodes, edges)
        rows = table.splitlines()
        assert rows[:3] == ["concept | status | roles", "ad_spend | observed | treatment",
                            "brand_trust | unmeasured | collider"]
        assert rows[4] == "season | proxy_measured | confounder_candidate, precision_covariate"
        assert rows[5] == "edge | status"
        assert rows[6:] == ["ad_spend -> sales | evidenced", "brand_trust -> sales | unknown",
                            "season -> ad_spend | hypothesis", "season -> sales | disputed"]

    def test_the_disclosure_names_the_view_and_counts_the_alternatives(self) -> None:
        nodes, edges, _, _ = spec_of()
        base, _, _ = renderer._texts(CONTEXT, None, nodes, edges)
        assert "the base graph" in base and "1 alternative graph view(s)" in base
        assert "kept and rendered separately, never merged into this diagram" in base
        chosen, _, _ = renderer._texts(CONTEXT, "alt-collider", nodes, edges)
        assert "alternative 'brand trust as a collider'" in chosen


class TestRender:
    def test_an_absent_dot_binary_is_a_typed_blocker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(renderer.RendererError) as error:
            render()
        assert error.value.code == "renderer_unavailable"

    def test_a_lost_node_fails_the_fidelity_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(renderer, "_render_svg", lambda _d: "<svg>ad_spend sales season</svg>")
        with pytest.raises(renderer.RendererError) as error:
            render()
        assert error.value.code == "graph_view_infidelity"
        assert "brand_trust" in str(error.value)

    def test_xml_escaped_node_identity_survives_the_fidelity_check(self) -> None:
        node = renderer.GraphNodeViewV1(
            concept_id="full-time & wages", label="employment", status=ConceptStatus.OBSERVED,
            roles=())
        spec: renderer.GraphSpec = ((node,), (), {}, "")
        renderer._check_fidelity(
            '<svg><g class="node"><title>full&#45;time &amp; wages</title></g></svg>', "",
            "full-time & wages | observed | none", spec)

    @requires_dot
    def test_the_rendered_view_validates_and_records_the_actual_graphviz_version(self) -> None:
        view = render()
        assert isinstance(view, CausalGraphViewV1) and "<svg" in view.svg
        assert view.renderer_version == f"graphviz-{renderer.graphviz_version()}"
        assert view.spec_hash == content_hash(spec_of()[2])
        assert view.legend_text == renderer.LEGEND_TEXT and view.parents == PARENTS
        assert (view.validation_status, view.theme_version, view.validator_version) == (
            "validated", "design-graph-theme.v1", "design-validators.v1")

    @requires_dot
    def test_an_alternative_renders_as_its_own_labelled_view(self) -> None:
        view = render("alt-collider")
        assert view.selected_alternative_id == "alt-collider"
        assert [e.edge_id for e in view.edges] == ["e-spend-trust", "e-spend-sales"]
        assert "brand trust as a collider" in view.disclosure_text


def test_the_version_probe_is_none_without_a_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert renderer.graphviz_version() is None


@requires_dot
def test_the_version_probe_reports_the_actual_local_version() -> None:
    version = renderer.graphviz_version()
    assert version is not None and version[0].isdigit()


@requires_dot
@pytest.mark.parametrize("alternative", (None, "alt-collider"))
def test_colon_and_space_concept_ids_render_literal_edges_in_base_and_alternative(
        alternative: str | None) -> None:
    mapping = {name: f"c:{name} with space" for name in CONTEXT.concept_ids}

    def renamed(value: Any) -> Any:
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, dict):
            return {key: renamed(item) for key, item in value.items()}
        if isinstance(value, list):
            return [renamed(item) for item in value]
        return value

    context = CausalContextV1.model_validate_json(json.dumps(
        renamed(CONTEXT.model_dump(mode="json"))))
    ledger = RoleLedgerV1.model_validate_json(json.dumps(
        renamed(LEDGER.model_dump(mode="json"))))
    measurements = MeasurementMapV1.model_validate_json(json.dumps(
        renamed(MAP.model_dump(mode="json"))))
    view = renderer.render_causal_graph(
        context=context, ledger=ledger, measurement_map=measurements, parents=PARENTS,
        selected_alternative_id=alternative)
    root = ElementTree.fromstring(view.svg)
    actual = {kind: [group.find("{*}title").text for group in root.iter()  # type: ignore[union-attr]
                     if group.attrib.get("class") == kind] for kind in ("node", "edge")}
    assert set(actual["node"]) == set(mapping.values())
    assert sorted(actual["edge"]) == sorted(
        f"{edge.source_concept_id}->{edge.target_concept_id}" for edge in view.edges)
    assert all(node.concept_id in node.label for node in view.nodes)
    # Corrupt the real SVG connectivity while leaving every node label and DOT unchanged.
    first = next(group for group in root.iter() if group.attrib.get("class") == "edge")
    first.find("{*}title").text = "c->c"  # type: ignore[union-attr]
    spec = renderer.build_graph_spec(context, ledger, measurements, alternative_id=alternative)
    with pytest.raises(renderer.RendererError, match="svg_edge_set"):
        renderer._check_fidelity(ElementTree.tostring(root, encoding="unicode"), spec[3],
                                 view.node_edge_table, spec)

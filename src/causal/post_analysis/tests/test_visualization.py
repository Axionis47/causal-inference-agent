"""Evidence fidelity, honest encodings, and exact visual exports."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from pydantic import ValidationError

from causal.post_analysis.visualization import (
    CausalDiagram,
    Column,
    DataTable,
    DiagramEdge,
    DiagramNode,
    VisualSpec,
    render_dag,
    render_visual,
)
from causal.post_analysis.visualization.compile import compile_visual, validate_visual
from causal.shared.contracts import ArtifactRef


@pytest.fixture
def table() -> DataTable:
    return DataTable(
        table_id="effect",
        source=ArtifactRef(artifact_id="analysis_result", content_hash="a" * 64),
        selector="/results/primary",
        columns=(
            Column(
                name="group",
                label="Population",
                kind="nominal",
                quantity="population",
                role="dimension",
            ),
            Column(
                name="estimate",
                label="Estimated effect",
                kind="quantitative",
                quantity="effect",
                units="points",
            ),
            Column(
                name="low",
                label="Lower bound",
                kind="quantitative",
                quantity="effect",
                units="points",
                role="lower",
            ),
            Column(
                name="high",
                label="Upper bound",
                kind="quantitative",
                quantity="effect",
                units="points",
                role="upper",
            ),
        ),
        rows=(("All participants", 2.125, 0.375, 3.875), ("Site B", -1.25, -2.5, 0.0)),
    )


def spec(**changes: object) -> VisualSpec:
    return VisualSpec.model_validate(
        {
            "visual_id": "primary_effect",
            "table_id": "effect",
            "kind": "point",
            "x": "estimate",
            "y": "group",
            "interval_lower": "low",
            "interval_upper": "high",
            "interval_axis": "x",
            "title": "Estimated effect by population",
            "caption": "Intervals are supplied by the analysis.",
        }
        | changes
    )


def test_literal_values_exact_artifact_binding_and_same_svg_png(
    table: DataTable, tmp_path: Path
) -> None:
    result = render_visual(spec(), {table.table_id: table}, tmp_path)
    assert result.status == "rendered", result.error_codes
    for name, path in result.objects.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == result.object_hashes[name]
    assert json.loads(Path(result.objects["data.json"]).read_text()) == table.model_dump(
        mode="json"
    )
    chart = json.loads(Path(result.objects["chart.json"]).read_text())
    assert chart["data"]["values"][0] == {
        "c0": "All participants",
        "c1": 2.125,
        "c2": 0.375,
        "c3": 3.875,
        "row_order": 0,
    }
    assert not any(word in chart for word in ("transform", "aggregate", "calculate"))
    svg = ET.fromstring(Path(result.objects["figure.svg"]).read_text())
    png = Path(result.objects["figure.png"]).read_bytes()
    assert struct.unpack(">II", png[16:24]) == tuple(
        round(float(svg.attrib[name]) * 2) for name in ("width", "height")
    )
    assert result.source == table.source and result.selector == table.selector
    assert render_visual(spec(), {table.table_id: table}, tmp_path) == result
    revised = render_visual(spec(caption="Revised caption."), {table.table_id: table}, tmp_path)
    assert revised.objects["figure.svg"] != result.objects["figure.svg"]
    Path(result.objects["figure.svg"]).write_text("changed")
    with pytest.raises(ValueError, match="immutable_visual_output_changed"):
        render_visual(spec(), {table.table_id: table}, tmp_path)


def test_uncertainty_is_bound_to_quantity_and_roles(table: DataTable) -> None:
    assert "interval_quantity_mismatch" in validate_visual(spec(interval_lower="estimate"), table)
    altered = table.model_copy(
        update={
            "columns": (
                *table.columns[:2],
                table.columns[2].model_copy(update={"units": "percent"}),
                table.columns[3],
            )
        }
    )
    assert validate_visual(spec(), altered) == ("interval_quantity_mismatch",)
    assert "interval_bound_is_not_a_point_estimate" in validate_visual(spec(x="low"), table)
    reversed_bounds = table.model_copy(update={"rows": (("A", 2.0, 4.0, 1.0),)})
    assert validate_visual(spec(), reversed_bounds) == ("reversed_interval",)
    one_bound = table.model_copy(update={"rows": (("A", 2.0, None, 4.0),)})
    assert validate_visual(spec(), one_bound) == ("incomplete_interval",)


def test_missing_values_preserved_and_never_zero_filled(table: DataTable, tmp_path: Path) -> None:
    partial = table.model_copy(
        update={"rows": (*table.rows, ("Unavailable group", None, None, None))}
    )
    result = render_visual(spec(), {"effect": partial}, tmp_path)
    assert result.status == "rendered", result.error_codes
    assert result.warnings == ("missing_coordinates", "missing_intervals")
    assert json.loads(Path(result.objects["data.json"]).read_text())["rows"][-1] == [
        "Unavailable group",
        None,
        None,
        None,
    ]
    missing = table.model_copy(update={"rows": (("Unavailable group", None, None, None),)})
    unavailable = render_visual(spec(visual_id="missing"), {"effect": missing}, tmp_path)
    assert unavailable.status == "unavailable" and "figure.png" not in unavailable.objects


@pytest.mark.parametrize("value", ["1.2", True, float("nan"), float("inf")])
def test_numeric_table_rejects_non_numeric_or_non_finite_values(
    table: DataTable, value: object
) -> None:
    with pytest.raises(ValidationError):
        DataTable.model_validate(table.model_dump() | {"rows": (("A", value, 0, 1),)})


def test_literal_field_names_and_no_implicit_bar_aggregation(table: DataTable) -> None:
    weird = "datum['value'].x"
    renamed = table.model_copy(
        update={
            "columns": (
                table.columns[0],
                table.columns[1].model_copy(update={"name": weird}),
                *table.columns[2:],
            )
        }
    )
    document = compile_visual(spec(x=weird), renamed)
    assert document["layer"][-1]["encoding"]["x"]["field"] == "c1"
    bar = spec(kind="bar", x="group", y="estimate", interval_axis="y")
    assert compile_visual(bar, table)["layer"][-1]["encoding"]["y"]["stack"] is None
    duplicates = table.model_copy(update={"rows": (table.rows[0], table.rows[0])})
    assert "ambiguous_coordinate_requires_explicit_series" in validate_visual(bar, duplicates)
    with pytest.raises(ValidationError):
        spec(visual_id="../overwrite")
    assert validate_visual(spec(x="absent"), table) == ("unknown_field:absent",)


def test_table_escapes_labels_and_preserves_values(table: DataTable, tmp_path: Path) -> None:
    result = render_visual(
        spec(
            kind="table",
            x=None,
            y=None,
            interval_lower=None,
            interval_upper=None,
            title="Effect <estimate> & evidence",
        ),
        {"effect": table},
        tmp_path,
    )
    assert result.status == "rendered", result.error_codes
    svg = Path(result.objects["figure.svg"]).read_text()
    assert "&lt;estimate&gt; &amp; evidence" in svg
    assert "2.125" in svg and "0.375" in svg
    crowded = table.model_copy(update={"rows": table.rows * 26})
    result = render_visual(
        spec(kind="table", x=None, y=None, interval_lower=None, interval_upper=None),
        {"effect": crowded},
        tmp_path,
    )
    assert result.error_codes == ("table_display_capacity",)


def test_supplied_dag_edge_status_and_refs_preserved(table: DataTable, tmp_path: Path) -> None:
    diagram = CausalDiagram(
        source=table.source,
        selector="/causal_graph",
        nodes=(
            DiagramNode(node_id="u", label="Underlying conditions"),
            DiagramNode(node_id="t", label="Treatment <received>"),
            DiagramNode(node_id="y", label="Outcome"),
        ),
        edges=(
            DiagramEdge(edge_id="e1", source_node="u", target_node="t", status="hypothesis"),
            DiagramEdge(
                edge_id="e2",
                source_node="t",
                target_node="y",
                status="disputed",
                evidence_ids=("claim-1",),
                contrary_evidence_ids=("claim-2",),
            ),
        ),
    )
    result = render_dag("causal_graph", diagram, tmp_path)
    assert result.status == "rendered", result.error_codes
    assert json.loads(Path(result.objects["data.json"]).read_text()) == diagram.model_dump(
        mode="json"
    )
    assert "does not prove causal identification" in result.caption
    assert "#ae3434" in Path(result.objects["figure.svg"]).read_text()
    cyclic = diagram.model_copy(
        update={
            "edges": (
                *diagram.edges,
                DiagramEdge(edge_id="e3", source_node="y", target_node="u", status="unknown"),
            )
        }
    )
    assert render_dag("cyclic", cyclic, tmp_path).error_codes == ("causal_graph_has_cycle",)


@pytest.mark.parametrize("kind", ["line", "bar"])
def test_grouped_series_draw_supplied_intervals_without_aggregation(
    table: DataTable, tmp_path: Path, kind: str
) -> None:
    time = Column(name="time", label="Period", kind="ordinal", quantity="period", role="dimension")
    grouped = table.model_copy(
        update={
            "columns": (*table.columns, time),
            "rows": (
                ("A", 1.2, 0.8, 1.6, "before"),
                ("A", 1.8, 1.1, 2.5, "after"),
                ("B", 0.6, 0.2, 1.0, "before"),
                ("B", 1.0, 0.5, 1.5, "after"),
            ),
        }
    )
    chosen = spec(kind=kind, x="time", y="estimate", series="group", interval_axis="y")
    result = render_visual(chosen, {"effect": grouped}, tmp_path)
    assert result.status == "rendered", result.error_codes
    assert json.loads(Path(result.objects["data.json"]).read_text())["rows"] == [
        list(row) for row in grouped.rows
    ]
    document = json.loads(Path(result.objects["chart.json"]).read_text())
    assert all("aggregate" not in json.dumps(layer) for layer in document["layer"])


def test_scientific_axis_units_and_per_row_sources_survive_export(tmp_path: Path) -> None:
    from causal.post_analysis.visualization import TableRowSource

    source = ArtifactRef(artifact_id="density_measurements", content_hash="b" * 64)
    supplied = DataTable(
        table_id="density",
        source=source,
        selector="/measurements",
        columns=(
            Column(
                name="side", label="Limit", kind="nominal", quantity="cutoff side", role="series"
            ),
            Column(
                name="score",
                label="Poverty rate",
                kind="quantitative",
                quantity="poverty rate",
                units="percent",
            ),
            Column(
                name="density",
                label="Density at cutoff",
                kind="quantitative",
                quantity="density",
                units="1 / percent",
            ),
        ),
        rows=(("below cutoff", 59.1984, 0.01001), ("above cutoff", 59.1984, 0.00837)),
        row_sources=tuple(
            TableRowSource(source=source, selector=f"/density_{side}") for side in ("left", "right")
        ),
    )
    chosen = VisualSpec(
        visual_id="density",
        table_id="density",
        kind="point",
        x="score",
        y="density",
        series="side",
        title="Supplied density limits at the cutoff",
    )
    result = render_visual(chosen, {"density": supplied}, tmp_path)
    assert result.status == "rendered", result.error_codes
    document = json.loads(Path(result.objects["chart.json"]).read_text())
    encoding = document["layer"][-1]["encoding"]
    assert encoding["x"]["title"] == "Poverty rate (percent)"
    assert encoding["y"]["title"] == "Density at cutoff (1 / percent)"
    copied = json.loads(Path(result.objects["data.json"]).read_text())
    assert copied == supplied.model_dump(mode="json")
    assert [row["selector"] for row in copied["row_sources"]] == ["/density_left", "/density_right"]
    with pytest.raises(ValidationError, match="row_sources must bind every supplied row"):
        DataTable.model_validate(supplied.model_dump() | {"row_sources": supplied.row_sources[:1]})

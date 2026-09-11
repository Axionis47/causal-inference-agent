"""Final review sees the exact paginated delivery, with complete citations and values."""

from __future__ import annotations

import base64
import hashlib
import json
import struct
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from causal.post_analysis.contracts import Citation, ReportDraft, Section, Statement
from causal.post_analysis.presentation.build import build_report
from causal.post_analysis.visualization import (
    Column,
    DataTable,
    TableRowSource,
    VisualSpec,
    render_visual,
)
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef


def draft(
    *,
    text: str = "The supplied effect is 2.125 points; identification assumptions remain.",
    visual_ids: tuple[str, ...] = ("effect",),
) -> ReportDraft:
    return ReportDraft(
        title="Causal evidence <review> & limitations",
        sections=(
            Section(
                title="What the evidence supports",
                statements=(
                    Statement(
                        text=text,
                        citations=(Citation(evidence_id="primary", selector="/estimate"),),
                    ),
                ),
                visual_ids=visual_ids,
            ),
        ),
        coverage={
            "primary": "Estimate considered.",
            "overlap": "Unavailable diagnostic; overlap has not been established.",
        },
    )


def rendered(tmp_path: Path):
    source = ArtifactRef(artifact_id="analysis_result", content_hash="a" * 64)
    table = DataTable(
        table_id="effect",
        source=source,
        selector="/primary",
        columns=(
            Column(name="group", label="Population", kind="nominal", quantity="population"),
            Column(
                name="effect",
                label="Estimated effect",
                kind="quantitative",
                quantity="effect",
                units="points",
            ),
        ),
        rows=(("All participants", 2.125),),
        row_sources=(TableRowSource(source=source, selector="/primary/0"),),
    )
    spec = VisualSpec(
        visual_id="effect",
        table_id="effect",
        kind="point",
        x="effect",
        y="group",
        title="Estimated effect",
        caption="No interval was supplied; precision is unavailable.",
    )
    return render_visual(spec, {"effect": table}, tmp_path)


def test_exact_preview_pages_selected_image_and_accessible_rows(tmp_path: Path) -> None:
    visual = rendered(tmp_path / "figures")
    assert visual.status == "rendered", visual.error_codes
    authored = draft()
    result = build_report(authored, {"effect": visual}, tmp_path / "report")
    assert result["draft_hash"] == content_hash(authored.model_dump(mode="json"))
    html = Path(result["objects"]["html"]).read_text()
    assert "&lt;review&gt; &amp; limitations" in html
    assert "primary/estimate" in html and "Unavailable diagnostic" in html
    assert "2.125" in html and "Source selector" in html and "/primary/0" in html
    assert "<script" not in html and "https://" not in html
    found_image = False
    for key, path in result["objects"].items():
        raw = Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == result["object_hashes"][key]
        if key.endswith("_svg"):
            assert raw.decode() in html
            root = ET.fromstring(raw)
            for image in root.findall("{http://www.w3.org/2000/svg}image"):
                assert (
                    base64.b64decode(image.attrib["href"].split(",", 1)[1])
                    == Path(visual.objects["figure.png"]).read_bytes()
                )
                assert float(image.attrib["y"]) + float(image.attrib["height"]) <= 1040
                found_image = True
        elif key.endswith("_png"):
            assert key in result["preview_keys"]
            assert struct.unpack(">II", raw[16:24]) == (1275, 1650)
    assert found_image
    assert build_report(authored, {"effect": visual}, tmp_path / "report") == result


def test_pagination_preserves_every_statement_and_citation(tmp_path: Path) -> None:
    statements = tuple(
        Statement(
            text=f"Finding {i}: " + "Full evidence explanation. " * 90,
            citations=(Citation(evidence_id=f"evidence_{i}", selector="/result"),),
        )
        for i in range(8)
    )
    authored = ReportDraft(
        title="Detailed evidence",
        sections=(Section(title="Results", statements=statements),),
        coverage={},
    )
    result = build_report(authored, {}, tmp_path)
    assert 2 < len(result["preview_keys"]) <= 24
    text = " ".join(
        " ".join(element.text or "" for element in ET.fromstring(Path(path).read_bytes()).findall(
            "{http://www.w3.org/2000/svg}text") if float(element.attrib["y"]) < 1060)
        for key, path in result["objects"].items()
        if key.endswith("_svg")
    )
    for index in range(8):
        assert f"Finding {index}:" in text
        assert f"evidence_{index}/result" in text
    assert text.count("Full evidence explanation.") == 720


def test_source_byte_change_and_unknown_visual_rejected(tmp_path: Path) -> None:
    visual = rendered(tmp_path / "figures")
    with pytest.raises(ValueError, match="unknown_report_visual"):
        build_report(draft(), {}, tmp_path / "report")
    Path(visual.objects["data.json"]).write_text(json.dumps({"rows": [[999]]}))
    with pytest.raises(ValueError, match="visual_object_hash_mismatch"):
        build_report(draft(), {"effect": visual}, tmp_path / "report")


def test_report_capacity_does_not_silently_truncate(tmp_path: Path) -> None:
    statement = Statement(text="W" * 4000, citations=(Citation(evidence_id="source"),))
    authored = ReportDraft(
        title="Long evidence",
        sections=(Section(title="Findings", statements=(statement,) * 30),),
        coverage={},
    )
    with pytest.raises(ValueError, match="report_page_capacity_exceeded"):
        build_report(authored, {}, tmp_path)

# The §13 renderer: the vendored font or a blocker, one SVG and one PNG from the same document
# at the one pinned display profile, the §14 accessible table, and gate 4 (T-031 §2).

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from causal.presentation import contracts as pc
from causal.presentation import render as rd
from tests.presentation import test_compile as fx

FONT = fx.REPO / fx.CATALOG.font_file_path
needs_font = pytest.mark.skipif(not FONT.exists(), reason="the vendored Noto Sans is absent")
PROFILE = fx.CATALOG.display_profile
# §17: the four label shapes the one profile must draw, and the one that cannot stay inside it.
LABELS = {"short": ["a", "b"], "sparse": ["only_arm"],
          "long": ["treatment arm b versus the pooled control group"],
          "dense": [f"arm_{index:02d}" for index in range(8)]}


def series_data(names: list[str]) -> dict[str, Any]:
    return dict(fx.DATA) | {fx.PRIMARY: fx.figure_data(fx.PRIMARY, [
        fx.point(name, "primary", x_value=0.1 + index / 100, interval_lower=0.0, interval_upper=0.3)
        for index, name in enumerate(names)])}


def drawn(shape: str, tmp_path: Path) -> tuple[pc.FigureSpecV1, rd.RenderOutput]:
    rd.register_font(fx.CATALOG, fx.REPO)
    spec, document = fx.compiled(fx.PRIMARY, fx.FOREST, data=series_data(LABELS[shape]))
    return spec, rd.render(spec, document, fx.CATALOG, tmp_path, fx.LINEAGE)


class TestFontAndFingerprint:
    def test_a_missing_or_mismatched_vendored_font_is_a_blocker(self, tmp_path: Path) -> None:
        swapped = fx.CATALOG.model_copy(update={"font_sha256": fx.digest("another face")})
        with pytest.raises(pc.PresentationError) as absent:
            rd.register_font(fx.CATALOG, tmp_path)
        with pytest.raises(pc.PresentationError) as mismatched:
            rd.register_font(swapped, fx.REPO)
        assert absent.value.code == mismatched.value.code == rd.FONT_BLOCKER

    def test_the_fingerprint_names_every_replay_input(self) -> None:
        found = rd.fingerprint(fx.CATALOG)
        assert set(found) >= {"python", "altair", "vl_convert", "vega_lite_schema", "os", "arch",
                              "theme_hash", "font_sha256", "display_profile", "compiler",
                              "renderer"}
        assert found["display_profile"] == PROFILE.display_profile_version
        assert found["font_sha256"] == fx.CATALOG.font_sha256 == (
            rd.register_font(fx.CATALOG, fx.REPO) if FONT.exists() else fx.CATALOG.font_sha256)


@needs_font
class TestRender:
    @pytest.mark.parametrize("shape", sorted(LABELS))
    def test_each_label_fixture_renders_one_svg_and_one_png_in_the_one_profile(
            self, shape: str, tmp_path: Path) -> None:
        spec, output = drawn(shape, tmp_path)
        codes = rd.render_codes(spec, output, fx.TEMPLATES[fx.FOREST], fx.CATALOG)
        assert output.svg.startswith("<svg") and output.png[:8] == b"\x89PNG\r\n\x1a\n"
        assert all(Path(where).read_bytes() for where in output.artifact.objects.values())
        assert output.artifact.spec_hash == spec.spec_hash()
        assert set(output.artifact.object_hashes) == {"svg", "png"}
        if shape == "long":
            # §13: the one label fixture that cannot stay inside the pinned canvas.
            assert any(code.startswith(rd.CLIPPED) for code in codes)
            assert rd.render_status(codes) == "needs_layout_revision"
            return
        assert codes == () and rd.render_status(codes) == "complete"
        assert rd._canvas(output.svg)[0] == PROFILE.logical_width
        assert rd._png_width(output.png) == PROFILE.png_width

    def test_a_specification_outside_its_bounds_or_its_render_fails_gate_four(
            self, tmp_path: Path) -> None:
        spec, out = drawn("short", tmp_path)
        over = (spec.model_copy(update={"logical_height": 9000}), f"{rd.CLIPPED}:logical_height")
        wrong = (spec.model_copy(update={"figure_id": "fig_other"}), rd.PARITY_BROKEN)
        for broken, code in (over, wrong):
            assert any(row.startswith(code)
                       for row in rd.render_codes(broken, out, fx.TEMPLATES[fx.FOREST], fx.CATALOG))


class TestAccessibility:
    def test_the_frozen_value_table_matches_the_specification_data_exactly(self) -> None:
        data = series_data(LABELS["dense"])
        spec, _ = fx.compiled(fx.PRIMARY, fx.FOREST, data=data)
        row = fx.entry(f"fig_{fx.PRIMARY}", fx.FOREST, (fx.PRIMARY,))
        table = rd.accessible_table(spec, row, data, fx.CATALOG.limits.accessible_table_max_rows)
        frozen = data[fx.PRIMARY]["points"]
        assert table["spec_hash"] == spec.spec_hash() and len(table["rows"]) == len(frozen)
        assert [found["x_value"] for found in table["rows"]] == [row["x_value"] for row in frozen]
        assert {found["panel_id"] for found in table["rows"]} == {spec.panels[0].panel_id}
        assert rd.table_codes(table, spec) == ()
        assert rd.table_codes(rd.accessible_table(spec, row, data, 2), spec) != ()

    def test_a_description_that_omits_what_was_drawn_is_inaccurate(self) -> None:
        spec, _ = fx.compiled(fx.PRIMARY, fx.FOREST)
        assert rd.description_codes(spec) == ()
        thin = spec.model_copy(update={"text": dict(spec.text) | {
            "accessible_description": "a chart of the result"}})
        assert {f"{rd.DESCRIPTION_INACCURATE}:interval",
                f"{rd.DESCRIPTION_INACCURATE}:null_effect"} <= set(rd.description_codes(thin))
        qualified = spec.model_copy(update={"qualification_ids": ("attrition_above_threshold",)})
        assert rd.description_codes(qualified) == (
            f"{rd.DESCRIPTION_INACCURATE}:attrition_above_threshold",)
